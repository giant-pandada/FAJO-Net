import math
import torch
import numpy as np
from torch.nn import init
import torch.nn.functional as F
from torch import nn
from einops import rearrange
import numbers
import torch.fft as fft
import cv2
from torchvision.transforms import Resize, functional

class FAJONet(nn.Module):
    def __init__(self, config):
        super(FAJONet, self).__init__()
        self.config = config
        self.phi_size = 32
        points = self.phi_size ** 2
        self.point = int(np.ceil(config.ratio * points))
        self.phi = nn.Parameter(init.xavier_normal_(torch.Tensor(self.point, 1, self.phi_size, self.phi_size)))
        self.channels = 32
        self.recon = CA(channels=self.channels)
        self.conv_start = nn.Conv2d(1, self.channels, kernel_size=1, stride=1, bias=False)
        self.conv_end = nn.Conv2d(self.channels, 1, kernel_size=3, stride=1, padding=1, bias=False)
        self.split = Split_freq()

    def forward(self, inputs):
        out_split, _ = self.split(inputs)
        y = F.conv2d(inputs, self.phi, stride=self.phi_size, padding=0, bias=None)
        recon_init = F.conv_transpose2d(y, self.phi, stride=self.phi_size)
        rec = self.conv_start(recon_init)
        rec = self.recon(rec, recon_init, self.phi, self.phi_size)
        rec = self.conv_end(rec)
        return rec

class UshapedNet(nn.Module):
    def __init__(self, dim, bias=True):
        super(UshapedNet, self).__init__()
        self.rb1 = RB(dim=dim)
        self.rb2 = RB(dim=dim)
        self.down1 = nn.Sequential(
            nn.Conv2d(dim, dim * 2, kernel_size=2, stride=2, bias=bias),
            RB(dim=dim*2),
        )
        self.down2 = nn.Sequential(
            nn.Conv2d(dim * 2, dim * 4, kernel_size=2, stride=2, bias=bias),
            RB(dim=dim * 4),
        )
        self.up2 = nn.Sequential(
            nn.ConvTranspose2d(dim * 4, dim * 2, kernel_size=2, stride=2, bias=bias),
            RB(dim=dim * 2),
        )
        self.up1 = nn.Sequential(
            nn.ConvTranspose2d(dim * 2, dim, kernel_size=2, stride=2, bias=bias),
            RB(dim=dim),
        )

    def forward(self, x):
        f = self.rb1(x)
        fd1 = self.down1(f)
        fd2 = self.down2(fd1)
        fu2 = self.up2(fd2)+fd1
        fu1 = self.up1(fu2)+f
        out = self.rb2(fu1)
        return out

def to_3d(x):
    return rearrange(x, 'b c h w -> b (h w) c')

def to_4d(x,h,w):
    return rearrange(x, 'b (h w) c -> b c h w',h=h,w=w)

class BiasFree_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(BiasFree_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)
        assert len(normalized_shape) == 1
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return x / torch.sqrt(sigma+1e-5) * self.weight

class WithBias_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(WithBias_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)
        assert len(normalized_shape) == 1
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        mu = x.mean(-1, keepdim=True)
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return (x - mu) / torch.sqrt(sigma+1e-5) * self.weight + self.bias

class Split_freq(nn.Module):
    def __init__(self, channel_num=2):
        super(Split_freq, self).__init__()
        self.channel_num = channel_num
        self.mask = self.generate_freq_mask(1024, 1024)

    def generate_freq_mask(self, H, W):
        length = math.sqrt((H / 2) ** 2 + (W / 2) ** 2)
        length_interval = length / self.channel_num
        pf_chunk = []
        for i in range(self.channel_num):
            pf = np.zeros((H, W))
            cv2.circle(pf, (W // 2, H // 2), math.ceil((i + 1) * length_interval), (1), -1)
            pf = torch.from_numpy(pf).float().unsqueeze(0)
            if i == 0:
                pass
            else:
                for prev in pf_chunk:
                    pf = pf - prev
            pf_chunk.append(pf)
        pf_chunk = torch.cat(pf_chunk, dim=0)
        return pf_chunk

    def forward(self, x):
        B, C, H, W = x.size()
        x_list = torch.split(x, 1, dim=1)
        mask = Resize([H, W], interpolation=functional.InterpolationMode.BICUBIC)(self.mask).to(x.device)
        out_list = []
        for x in x_list:
            f = fft.fftn(x, dim=(2, 3))
            f = fft.fftshift(f, dim=(2, 3))
            f_split = f * mask
            f_split = fft.ifftshift(f_split, dim=(2, 3))
            out = fft.ifftn(f_split, dim=(2, 3)).real
            out_list.append(out.unsqueeze(-1))
        out = torch.cat(out_list, dim=-1)
        out = torch.split(out, 1, dim=1)
        outs = []
        for f in out:
            outs.append(f.squeeze(1).permute(0, 3, 1, 2).contiguous())
        return outs, mask

class LayerNorm1(nn.Module):
    def __init__(self, dim, LayerNorm_type):
        super(LayerNorm1, self).__init__()
        if LayerNorm_type =='BiasFree':
            self.body = BiasFree_LayerNorm(dim)
        else:
            self.body = WithBias_LayerNorm(dim)

    def forward(self, x):
        h, w = x.shape[-2:]
        return to_4d(self.body(to_3d(x)), h, w)

class U(torch.nn.Module):
    def __init__(self, channels, num_layers=1):
        super(U, self).__init__()
        self.channels = channels
        self.num_layers = num_layers
        self.res = nn.ModuleList()
        self.conv321 = nn.ModuleList()
        self.conv132 = nn.ModuleList()
        self.conv324 = nn.ModuleList()
        self.conv432 = nn.ModuleList()
        for i in range(self.num_layers):
            self.conv132.append(nn.Sequential(
                nn.Conv2d(1, self.channels, kernel_size=3, stride=1, padding=1, bias=False),
            ))
            self.conv432.append(nn.Sequential(
                nn.Conv2d(4, self.channels, kernel_size=3, stride=1, padding=1, bias=False),
            ))
            self.conv321.append(nn.Sequential(
                nn.Conv2d(self.channels, 1, kernel_size=3, stride=1, padding=1, bias=False),
            ))
            self.conv324.append(nn.Sequential(
                RB(dim=self.channels),
                nn.Conv2d(self.channels, 4, kernel_size=3, stride=1, padding=1, bias=False),
            ))
        for i in range(self.num_layers):
            self.res.append(
                UshapedNet(dim=self.channels)
            )

    def forward(self, rec1, recon_init, phi, phi_size,j):
        # 1
        for i in range(self.num_layers):
            rec = self.conv321[i](rec1.contiguous())
            temp = F.conv2d(rec, phi, padding=0, stride=phi_size, bias=None)
            temp = F.conv_transpose2d(temp, phi, stride=phi_size)
            rec = (temp - recon_init)
            rec = self.conv132[i](rec)
            rec4 = self.conv324[i](rec1)
            b, c, h, w = rec4.shape
            temp = F.conv2d(rec4.reshape(-1, 1, h, w), phi, padding=0, stride=phi_size, bias=None)
            temp = F.conv_transpose2d(temp, phi, stride=phi_size).reshape(b, c, h, w)
            rec4 = self.conv432[i](temp - recon_init)
            rec1 = rec1 - (rec + rec4)
            rec1 = self.res[i](rec1)
        return rec1


class CA(torch.nn.Module):
    def __init__(self, channels, num_layers=8):
        super(CA, self).__init__()
        self.channels = channels
        self.num_layers = num_layers
        self.U1 = nn.ModuleList()
        self.U2 = nn.ModuleList()
        self.U3 = nn.ModuleList()
        self.FCAM = nn.ModuleList()
        for i in range(self.num_layers):
            self.U1.append(U(channels=self.channels))
            self.U2.append(U(channels=self.channels))
            self.U3.append(U(channels=self.channels))
            self.FCAM.append(FCAM(dim=self.channels))
        self.split = Split_freq()

    def forward(self, rec, recon_init, phi, phi_size):
        for i in range(self.num_layers):

            rec, _ = self.split(rec.contiguous())
            rec1 = self.U1[i](rec[0], recon_init, phi, phi_size,i)
            rec2 = self.U2[i](rec[1], recon_init, phi, phi_size,i)
            rec = self.U3[i](rec[0]+rec[1], recon_init, phi, phi_size,i)
            rec = self.FCAM[i](rec,rec1,rec2,i)

        return rec

class FCAM(nn.Module):
    def __init__(self, dim, bias=True):
        super(FCAM, self).__init__()
        self.qq = nn.Sequential(nn.Conv2d(dim, dim * 2, kernel_size=1, bias=bias),
            nn.Conv2d(dim * 2, dim * 2, kernel_size=3, stride=1, padding=1, groups=dim * 2, bias=bias))
        self.kv1 = nn.Sequential(nn.Conv2d(dim, dim * 2, kernel_size=1, bias=bias),
            nn.Conv2d(dim * 2, dim * 2, kernel_size=3, stride=1, padding=1, groups=dim * 2,
                                           bias=bias))
        self.kv2 = nn.Sequential(nn.Conv2d(dim, dim * 2, kernel_size=1, bias=bias),
                                 nn.Conv2d(dim * 2, dim * 2, kernel_size=3, stride=1, padding=1, groups=dim * 2,
                                           bias=bias))
        self.conv = nn.Conv2d(dim*2, dim, kernel_size=3, stride=1, padding=1, bias=bias)
        self.conv1 = nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=1, bias=bias)
        self.conv2 = nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=1, bias=bias)
        self.norm = LayerNorm1(dim, "WithBias")
        self.norm1 = LayerNorm1(dim, "WithBias")
        self.norm2 = LayerNorm1(dim, "WithBias")

        self.split = Split_freq()
    def forward(self, x,x1,x2,i):
        b, c, h, w = x.shape
        qq = self.qq(self.norm2(x))
        q, qq = qq.chunk(2, dim=1)

        kv1 = self.kv1(self.norm(x1))
        k1, v1 = kv1.chunk(2, dim=1)

        kv2 = self.kv2(self.norm1(x2))
        k2, v2 = kv2.chunk(2, dim=1)

        q = rearrange(q, 'b c h w -> b c (h w)')
        k1 = rearrange(k1, 'b c h w -> b c (h w)')
        v1 = rearrange(v1, 'b c h w -> b c (h w)')

        qq = rearrange(qq, 'b c h w -> b c (h w)')
        k2 = rearrange(k2, 'b c h w -> b c (h w)')
        v2 = rearrange(v2, 'b c h w -> b c (h w)')

        q = torch.nn.functional.normalize(q, dim=-1)
        k1 = torch.nn.functional.normalize(k1, dim=-1)

        qq = torch.nn.functional.normalize(qq, dim=-1)
        k2 = torch.nn.functional.normalize(k2, dim=-1)

        attn = (q @ k1.transpose(-2, -1)).softmax(dim=-1)
        attn1 = (qq @ k2.transpose(-2, -1)).softmax(dim=-1)

        out = (attn @ v1)
        out = rearrange(out, 'b c (h w) -> b c h w', h=h, w=w)
        out1 = (attn1 @ v2)
        out1 = rearrange(out1, 'b c (h w) -> b c h w', h=h, w=w)
        out = self.conv(torch.cat([self.conv1(out1) + x2 + self.conv2(out)+x1,x],dim=1))

        return out

class RB(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(dim, dim, 3, padding=1),
            nn.ReLU(True),
            nn.Conv2d(dim, dim, 3, padding=1),
        )

    def forward(self, x):
        return x + self.conv(x)




