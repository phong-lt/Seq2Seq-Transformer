import torch
from torch import nn
import torch.nn.functional as F

class SimpleDecoderLayer(nn.Module):
    def __init__(self,
                 d_model,
                 nhead,
                 dim_feedforward,
                 dropout):
        super(SimpleDecoderLayer, self).__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout)
        self.cross_attn = nn.MultiheadAttention(d_model, nhead, dropout)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)

        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.dropout = nn.Dropout(dropout)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

        self.activation = F.relu

    def forward(self, x, memory, trg_mask=None, memory_mask=None, trg_padding_mask=None, memory_key_padding_mask=None):
        x = self.norm1(x + self._sa_block(x, trg_mask, trg_padding_mask))
        x = self.norm2(x + self._ca_block(x, memory, memory_mask, memory_key_padding_mask))
        x = self.norm3(x + self._ff_block(x))

        return x

    def _sa_block(self, x, trg_mask=None, trg_padding_mask=None):
        sa = self.self_attn(x, x, x, attn_mask=trg_mask, key_padding_mask=trg_padding_mask)[0]
        return self.dropout1(sa)

    def _ca_block(self, x, memory, memory_mask=None, memory_key_padding_mask=None):
        ca = self.cross_attn(x, memory, memory, attn_mask=memory_mask, key_padding_mask=memory_key_padding_mask)[0]
        return self.dropout2(ca)

    def _ff_block(self, x):
        out = self.linear2(self.dropout(self.activation(self.linear1(x))))
        return self.dropout3(out)