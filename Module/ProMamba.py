import torch
import torch.nn as nn
import torch.nn.functional as F


class GatedConv1D(nn.Module):
    """蛋白质状态路径：门控卷积 + 残差 + 输出投影"""
    def __init__(self, dim, kernel_size=3, expansion=1):  # expansion=1 表示不升维
        super().__init__()
        hidden_dim = dim * expansion
        self.norm = nn.LayerNorm(dim)
        self.conv = nn.Conv1d(dim, hidden_dim * 2, kernel_size, padding=kernel_size // 2)
        self.proj_out = nn.Linear(hidden_dim, dim) if expansion != 1 else nn.Identity()
        self.expansion = expansion

    def forward(self, x):  # [B, L, D]
        residual = x
        x = self.norm(x)
        x = self.conv(x.transpose(1, 2)).transpose(1, 2)  # [B, L, 2H]
        v, g = x.chunk(2, dim=-1)                         # [B, L, H], [B, L, H]
        out = v * torch.sigmoid(g)                        # [B, L, H]
        out = self.proj_out(out)                          # [B, L, D]
        return residual + out

class LePELinearAttention(nn.Module):
    """局部位置增强的线性注意力"""
    def __init__(self, dim, num_heads=4, qkv_bias=True):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.qk_proj = nn.Linear(dim, dim * 2, bias=qkv_bias)
        self.v_proj = nn.Linear(dim, dim)
        self.elu = nn.ELU()
        self.lepe = nn.Conv1d(self.head_dim, self.head_dim, 3, padding=1, groups=self.head_dim)

    def forward(self, x):  # [B, L, C]
        B, L, C = x.shape
        H, D = self.num_heads, self.head_dim

        qk = self.qk_proj(x).reshape(B, L, 2, H, D).permute(2, 0, 3, 1, 4)
        q, k = qk[0], qk[1]  # [B, H, L, D]
        v = self.v_proj(x).reshape(B, L, H, D).permute(0, 2, 1, 3)

        q = self.elu(q) + 1.0
        k = self.elu(k) + 1.0

        # LePE: 每个 head 使用独立深度卷积增强位置信息
        q = q + self.lepe(q.reshape(B * H, D, L)).reshape(B, H, D, L).permute(0, 1, 3, 2)
        k = k + self.lepe(k.reshape(B * H, D, L)).reshape(B, H, D, L).permute(0, 1, 3, 2)

        k_mean = k.mean(dim=2, keepdim=True)  # [B, H, 1, D]
        z = 1 / (torch.matmul(q, k_mean.transpose(-2, -1)) + 1e-6)
        kv = torch.matmul(k.transpose(-2, -1), v) / L
        out = torch.matmul(q, kv) * z

        return out.transpose(1, 2).reshape(B, L, C)


class ProMambaBlock(nn.Module):
    def __init__(self, dim=256, num_heads=4, drop=0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = LePELinearAttention(dim, num_heads)
        self.drop1 = nn.Dropout(drop)

        self.norm2 = nn.LayerNorm(dim)
        self.state_path = GatedConv1D(dim)
        self.drop2 = nn.Dropout(drop)

        self.norm3 = nn.LayerNorm(dim)
        self.ffn = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.GELU(),
            nn.Linear(dim * 4, dim),
            nn.Dropout(drop)
        )

    def forward(self, x):
        x = x + self.drop1(self.attn(self.norm1(x)))
        x = x + self.drop2(self.state_path(self.norm2(x)))
        x = x + self.ffn(self.norm3(x))
        return x
