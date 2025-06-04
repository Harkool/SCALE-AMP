import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp import autocast
from functools import partial

def get_same_padding(kernel_size):
    return kernel_size // 2 if isinstance(kernel_size, int) else tuple(k // 2 for k in kernel_size)

def val2tuple(x, min_len=1):
    return x if isinstance(x, tuple) else tuple([x] * min_len)

class LayerNorm2d(nn.LayerNorm):
    def forward(self, x):
        out = x - torch.mean(x, dim=1, keepdim=True)
        out = out / torch.sqrt(torch.mean(out ** 2, dim=1, keepdim=True) + self.eps)
        if self.elementwise_affine:
            out = out * self.weight.view(1, -1, 1, 1) + self.bias.view(1, -1, 1, 1)
        return out

REGISTERED_ACT_DICT = {
    "relu": nn.ReLU,
    "gelu": partial(nn.GELU, approximate="tanh"),
    "silu": nn.SiLU
}

def build_act(name):
    return REGISTERED_ACT_DICT[name]() if name in REGISTERED_ACT_DICT else None

REGISTERED_NORM_DICT = {
    "bn2d": nn.BatchNorm2d,
    "ln2d": LayerNorm2d
}

def build_norm(name, num_features):
    if name == "ln2d":
        return LayerNorm2d([num_features])
    else:
        return REGISTERED_NORM_DICT[name](num_features)

class ConvLayer(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, norm="bn2d", act_func="relu"):
        super().__init__()
        padding = get_same_padding(kernel_size)
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, padding=padding, bias=False)
        self.norm = build_norm(norm, out_channels)
        self.act = build_act(act_func)

    def forward(self, x):
        x = self.conv(x)
        x = self.norm(x)
        x = self.act(x)
        return x

class LiteMLA(nn.Module):
    def __init__(self, in_channels, out_channels, dim=8, heads_ratio=1.0, kernel_func="relu", scales=(5,), eps=1e-12):
        super().__init__()
        self.eps = eps
        heads = int(in_channels // dim * heads_ratio)
        total_dim = heads * dim

        self.dim = dim
        self.qkv = ConvLayer(in_channels, 3 * total_dim, kernel_size=1)

        self.aggreg = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(3 * total_dim, 3 * total_dim, kernel_size=scale, padding=get_same_padding(scale), groups=3 * total_dim),
                nn.Conv2d(3 * total_dim, 3 * total_dim, kernel_size=1, groups=3 * heads)
            ) for scale in scales
        ])

        self.kernel_func = build_act(kernel_func)
        self.proj = ConvLayer(total_dim * (1 + len(scales)), out_channels, kernel_size=1)

    @autocast(enabled=False)
    def relu_linear_att(self, qkv):
        B, _, H, W = qkv.size()
        qkv = qkv.float().view(B, -1, 3 * self.dim, H * W).transpose(-1, -2)
        q, k, v = qkv[..., :self.dim], qkv[..., self.dim:2 * self.dim], qkv[..., 2 * self.dim:]

        q, k = self.kernel_func(q), self.kernel_func(k)
        k_t = k.transpose(-1, -2)
        v = F.pad(v, (0, 1), mode="constant", value=1)

        kv = torch.matmul(k_t, v)
        out = torch.matmul(q, kv)
        out = out[..., :-1] / (out[..., -1:] + self.eps)
        out = out.transpose(-1, -2).reshape(B, -1, H, W)
        return out

    def forward(self, x):
        qkv = self.qkv(x)
        multi_scale = [qkv] + [agg(qkv) for agg in self.aggreg]
        out = torch.cat(multi_scale, dim=1)
        out = self.relu_linear_att(out)
        out = self.proj(out)
        return out
