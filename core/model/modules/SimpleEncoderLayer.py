import torch
from torch import nn
import torch.nn.functional as F

class SimpleEncoderLayer(nn.Module):
    def __init__(self,
                 d_model,
                 nhead,
                 dim_feedforward,
                 dropout):
        super(SimpleEncoderLayer, self).__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout)

        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.dropout = nn.Dropout(dropout)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

        self.activation = F.relu

    def forward(self, x, src_mask=None, src_padding_mask=None):
        x = self.norm1(x + self._sa_block(x, src_mask, src_padding_mask))
        x = self.norm2(x + self._ff_block(x))
        return x

    def _sa_block(self, x, src_mask, src_padding_mask):
        sa = self.self_attn(x, x, x, attn_mask=src_mask, key_padding_mask=src_padding_mask)[0]
        return self.dropout1(sa)

    def _ff_block(self, x):
        x = self.linear2(self.dropout(self.activation(self.linear1(x))))
        return self.dropout2(x)


