import torch
import torch.nn as nn
import torch.nn.functional as F

class GroupBatchnorm2d(nn.Module):
    def __init__(self, c_num: int, group_num: int = 16, eps: float = 1e-10):
        super().__init__()
        assert c_num >= group_num
        self.group_num = group_num
        self.weight = nn.Parameter(torch.randn(c_num, 1, 1))
        self.bias = nn.Parameter(torch.zeros(c_num, 1, 1))
        self.eps = eps

    def forward(self, x):
        N, C, L, _ = x.size()  # Treat sequence as H, dummy W=1
        x = x.permute(0, 2, 3, 1).contiguous()  # [N, L, 1, C]
        x = x.view(N, L, self.group_num, C // self.group_num)
        x = x.permute(0, 2, 1, 3).contiguous().view(N, self.group_num, -1)
        mean = x.mean(dim=2, keepdim=True)
        std = x.std(dim=2, keepdim=True)
        x = (x - mean) / (std + self.eps)
        x = x.view(N, C, L, 1)
        return x * self.weight + self.bias


class SRU(nn.Module):
    def __init__(self, oup_channels: int, group_num: int = 16, gate_treshold: float = 0.5):
        super().__init__()
        self.gn = GroupBatchnorm2d(c_num=oup_channels, group_num=group_num)
        self.gate_treshold = gate_treshold
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        gn_x = self.gn(x)
        w_gamma = self.gn.weight / torch.sum(self.gn.weight)
        w_gamma = w_gamma.view(1, -1, 1, 1)
        reweights = self.sigmoid(gn_x * w_gamma)
        info_mask = reweights >= self.gate_treshold
        noninfo_mask = reweights < self.gate_treshold
        x_1 = info_mask * gn_x
        x_2 = noninfo_mask * gn_x
        return self.reconstruct(x_1, x_2)

    def reconstruct(self, x_1, x_2):
        x_11, x_12 = torch.chunk(x_1, 2, dim=1)
        x_21, x_22 = torch.chunk(x_2, 2, dim=1)
        return torch.cat([x_11 + x_22, x_12 + x_21], dim=1)


class CRU(nn.Module):
    def __init__(self, op_channel: int, alpha: float = 0.5, squeeze_radio: int = 2,
                 group_size: int = 2, group_kernel_size: int = 3):
        super().__init__()
        self.up_channel = int(alpha * op_channel)
        self.low_channel = op_channel - self.up_channel

        self.squeeze1 = nn.Conv2d(self.up_channel, self.up_channel // squeeze_radio, kernel_size=1, bias=False)
        self.squeeze2 = nn.Conv2d(self.low_channel, self.low_channel // squeeze_radio, kernel_size=1, bias=False)

        self.GWC = nn.Conv2d(self.up_channel // squeeze_radio, op_channel,
                             kernel_size=(group_kernel_size, 1),
                             stride=1, padding=(group_kernel_size // 2, 0),
                             groups=group_size)
        self.PWC1 = nn.Conv2d(self.up_channel // squeeze_radio, op_channel, kernel_size=1, bias=False)
        self.PWC2 = nn.Conv2d(self.low_channel // squeeze_radio,
                              op_channel - self.low_channel // squeeze_radio, kernel_size=1, bias=False)
        self.advavg = nn.AdaptiveAvgPool2d(1)

    def forward(self, x):
        up, low = torch.split(x, [self.up_channel, self.low_channel], dim=1)
        up, low = self.squeeze1(up), self.squeeze2(low)
        Y1 = self.GWC(up) + self.PWC1(up)
        Y2 = torch.cat([self.PWC2(low), low], dim=1)
        out = torch.cat([Y1, Y2], dim=1)
        weights = F.softmax(self.advavg(out), dim=1)
        out = weights * out
        out1, out2 = torch.chunk(out, 2, dim=1)
        return out1 + out2


class ScConv(nn.Module):
    def __init__(self, op_channel: int,
                 group_num: int = 4,
                 gate_treshold: float = 0.5,
                 alpha: float = 0.5,
                 squeeze_radio: int = 2,
                 group_size: int = 2,
                 group_kernel_size: int = 3):
        super().__init__()
        self.SRU = SRU(op_channel, group_num=group_num, gate_treshold=gate_treshold)
        self.bridge = nn.Sequential(
            nn.Conv2d(op_channel, op_channel, kernel_size=1),
            nn.GELU()
        )

        self.CRU = CRU(op_channel, alpha=alpha, squeeze_radio=squeeze_radio,
                       group_size=group_size, group_kernel_size=group_kernel_size)

    def forward(self, x):
        x = self.SRU(x)
        x = self.bridge(x)  
        x = self.CRU(x)
        return x

