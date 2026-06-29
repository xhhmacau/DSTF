import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import einops
from timm.models.layers import to_2tuple, trunc_normal_


class LayerNormProxy(nn.Module):
    """
    Layer normalization proxy for 1D convolution features
    """
    def __init__(self, dim):
        super().__init__()
        self.norm = nn.LayerNorm(dim)

    def forward(self, x):
        # x: [B, C, L] -> [B, L, C] -> LayerNorm -> [B, C, L]
        x = x.transpose(-1, -2)
        x = self.norm(x)
        x = x.transpose(-1, -2)
        return x


class TransformerMLP(nn.Module):
    """
    Standard Transformer MLP
    """
    def __init__(self, dim_embed, expansion, drop, local_kernel_size=None):
        super().__init__()
        dim_feedforward = dim_embed * expansion
        self.linear1 = nn.Conv1d(dim_embed, dim_feedforward, 1)
        self.activation = nn.GELU()
        self.dropout1 = nn.Dropout(drop)
        self.linear2 = nn.Conv1d(dim_feedforward, dim_embed, 1)
        self.dropout2 = nn.Dropout(drop)

    def forward(self, x):
        x = self.linear1(x)
        x = self.activation(x)
        x = self.dropout1(x)
        x = self.linear2(x)
        x = self.dropout2(x)
        return x


class TransformerMLPWithConv(nn.Module):
    """
    Transformer MLP with depth-wise convolution
    """
    def __init__(self, dim_embed, expansion, drop, local_kernel_size):
        super().__init__()
        dim_feedforward = dim_embed * expansion
        self.linear1 = nn.Conv1d(dim_embed, dim_feedforward, 1)
        self.dwconv = nn.Conv1d(
            dim_feedforward, dim_feedforward, 
            kernel_size=local_kernel_size, 
            stride=1, 
            padding=local_kernel_size//2,
            groups=dim_feedforward
        )
        self.activation = nn.GELU()
        self.dropout1 = nn.Dropout(drop)
        self.linear2 = nn.Conv1d(dim_feedforward, dim_embed, 1)
        self.dropout2 = nn.Dropout(drop)

    def forward(self, x):
        x = self.linear1(x)
        x = self.dwconv(x)
        x = self.activation(x)
        x = self.dropout1(x)
        x = self.linear2(x)
        x = self.dropout2(x)
        return x