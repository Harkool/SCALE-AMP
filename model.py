import torch
import torch.nn as nn
#from torch.nn.utils import weight_norm
from Module.litemla import LiteMLA
from Module.ScConv import ScConv
from Module.ProMamba import ProMambaBlock
import torch.nn.functional as F



class ResidualClassifier(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_classes, dropout=0.5):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, num_classes)
        self.dropout = nn.Dropout(dropout)
        self.act = nn.LeakyReLU()

    def forward(self, x):
        x = self.act(self.norm1(self.fc1(x)))
        x = self.dropout(x)
        x = self.act(self.norm2(self.fc2(x)))
        x = self.dropout(x)
        out = self.fc3(x)
        return torch.clamp(out, min=-10.0, max=10.0)  


class ESMAMP(nn.Module):
    def __init__(self, 
                 linsize: int = 1024, 
                 lindropout: float = 0.8,
                 num_labels: int = 14):
        super().__init__()

        self.hidden_size = 480

        self.scconv = ScConv(op_channel=self.hidden_size)
        self.mlla = ProMambaBlock(dim=self.hidden_size)

        self.norm_esm = nn.LayerNorm(self.hidden_size)  
        self.norm_scconv = nn.LayerNorm(self.hidden_size)
        self.norm_mlla = nn.LayerNorm(self.hidden_size)
        self.norm_litemla = nn.LayerNorm(self.hidden_size)
        self.norm_pool = nn.LayerNorm(self.hidden_size)

        self.litemla = LiteMLA(
            in_channels=self.hidden_size, 
            out_channels=self.hidden_size,
            dim=4,
            heads_ratio=0.5,
            kernel_func="relu", 
            scales=(3, 5, 7), 
            eps=1.0e-15
        )

        self.classify = ResidualClassifier(
            input_dim=self.hidden_size,
            hidden_dim=linsize,
            num_classes=num_labels,  
            dropout=lindropout
        )



    def forward(self, input_ids, input_mask=None, return_embedding=False):
        esm_output = input_ids  
        esm_output = self.norm_esm(esm_output)
        scconv_input = esm_output.permute(0, 2, 1).unsqueeze(-1)  
        scconv_output = self.scconv(scconv_input).squeeze(-1).permute(0, 2, 1) 
        scconv_output = self.norm_scconv(scconv_output)
        mlla_output = self.mlla(esm_output)  
        mlla_output = self.norm_mlla(mlla_output)

        residual_output = scconv_output + mlla_output + esm_output 

        mla_input = residual_output.permute(0, 2, 1).unsqueeze(-1) 
        mla_output = self.litemla(mla_input).squeeze(-1).permute(0, 2, 1)  
        mla_output = self.norm_litemla(mla_output)

        mla_output = F.dropout(mla_output, p=0.3, training=self.training)
        pooled_output = mla_output.mean(dim=1)
        pooled_output = self.norm_pool(pooled_output)
        pooled_output = F.dropout(pooled_output, p=0.3, training=self.training)

        logits = self.classify(pooled_output)
        logits = torch.clamp(logits, min=-10.0, max=10.0)

        return (logits, pooled_output, mla_output) if return_embedding else logits






