import math
import torch
from torch import nn
import torch.nn.functional as F

class MHA(nn.Module):
  def __init__(self,
               d_model,
               nhead,
               dropout):
      super(MHA, self).__init__()
      self.d_model = d_model

      self.nhead = nhead
      self.head_dim = d_model // nhead

      assert self.head_dim * nhead == self.d_model, "d_model must be divisible by nhead"

      self.query = nn.Linear(d_model, d_model)
      self.key = nn.Linear(d_model, d_model)
      self.value = nn.Linear(d_model, d_model)

      self.out = nn.Linear(d_model, d_model)

      self.dropout = nn.Dropout(dropout)

  def forward(self, q, k, v, attn_mask=None, key_padding_mask=None):  # batch first is False
      N = q.shape[1]

      q = self.query(q).view(-1, N, self.nhead, self.head_dim).transpose(0,2)
      k = self.key(k).view(-1, N, self.nhead, self.head_dim).transpose(0,2)
      v = self.value(v).view(-1, N, self.nhead, self.head_dim).transpose(0,2)



      attn_scores = torch.matmul(q, k.transpose(-1, -2)) / math.sqrt(self.head_dim)

      if attn_mask is not None:
            attn_scores += attn_mask

      att_probs = self.dropout(F.softmax(attn_scores, dim=-1))

      context = torch.matmul(att_probs, v).transpose(0,2).contiguous().view(-1, N, self.d_model)

      out = self.out(context)
      return out
