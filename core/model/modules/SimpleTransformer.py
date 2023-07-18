from torch import nn, Tensor
from . import SimpleEncoderLayer, SimpleDecoderLayer

class SimpleEncoder(nn.Module):
    def __init__(self,
                d_model,
                nhead,
                num_encoder_layers,
                dim_feedforward,
                dropout):
      super(SimpleEncoder, self).__init__()
      self.ens = nn.ModuleList(
            [SimpleEncoderLayer(d_model,
                                nhead,
                                dim_feedforward,
                                dropout)
             for _ in range(num_encoder_layers)]
        )
    def forward(self, x, src_mask=None, src_padding_mask=None):
      for en in self.ens:
        x = en(x, src_mask, src_padding_mask)
      return x

class SimpleDecoder(nn.Module):
    def __init__(self,
                d_model,
                nhead,
                num_decoder_layers,
                dim_feedforward,
                dropout):
      super(SimpleDecoder, self).__init__()
      self.des = nn.ModuleList(
            [SimpleDecoderLayer(d_model,
                                nhead,
                                dim_feedforward,
                                dropout)
             for _ in range(num_decoder_layers)]
        )
    def forward(self, x, memory, trg_mask=None, memory_mask=None, trg_padding_mask=None, memory_key_padding_mask=None):
      for de in self.des:
        x = de(x, memory, trg_mask, memory_mask, trg_padding_mask, memory_key_padding_mask)
      return x

class SimpleTransformer(nn.Module):
  def __init__(self,
              d_model,
              nhead,
              num_encoder_layers,
              num_decoder_layers,
              dim_feedforward,
              dropout):
      super(SimpleTransformer, self).__init__()
      self.encoder = SimpleEncoder(d_model,
                                   nhead,
                                   num_encoder_layers,
                                   dim_feedforward,
                                   dropout)
      self.decoder = SimpleDecoder(d_model,
                                   nhead,
                                   num_decoder_layers,
                                   dim_feedforward,
                                   dropout)
  def forward(self, src_emb, trg_emb, src_mask=None, trg_mask=None, memory_mask=None,
                                src_padding_mask=None, trg_padding_mask=None, memory_key_padding_mask=None):
      memory = self.encoder(src_emb, src_mask, src_padding_mask)
      return self.decoder(trg_emb, memory, trg_mask, memory_mask, trg_padding_mask, memory_key_padding_mask)
