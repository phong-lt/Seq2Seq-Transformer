from torch import nn, Tensor
from .modules import SimpleTransformer, SimplePositionalEncoding, SimpleEmbedding


class Seq2SeqTransformer(nn.Module):
    def __init__(self,
                 num_encoder_layers: int,
                 num_decoder_layers: int,
                 emb_size: int,
                 nhead: int,
                 src_vocab_size: int,
                 trg_vocab_size: int,
                 dim_feedforward: int = 512,
                 dropout: float = 0.1):
        super(Seq2SeqTransformer, self).__init__()
        self.transformer = SimpleTransformer( d_model=emb_size,
                                              nhead=nhead,
                                              num_encoder_layers=num_encoder_layers,
                                              num_decoder_layers=num_decoder_layers,
                                              dim_feedforward=dim_feedforward,
                                              dropout=dropout)
        self.generator = nn.Linear(emb_size, trg_vocab_size)
        self.src_tok_emb = SimpleEmbedding(src_vocab_size, emb_size)
        self.trg_tok_emb = SimpleEmbedding(trg_vocab_size, emb_size)
        self.positional_encoding = SimplePositionalEncoding(
            emb_size, dropout=dropout)

    def forward(self,
                src: Tensor,
                trg: Tensor,
                src_mask: Tensor,
                trg_mask: Tensor,
                src_padding_mask: Tensor,
                trg_padding_mask: Tensor,
                memory_key_padding_mask: Tensor):
        src_emb = self.positional_encoding(self.src_tok_emb(src))
        trg_emb = self.positional_encoding(self.trg_tok_emb(trg))
        outs = self.transformer(src_emb, trg_emb, src_mask, trg_mask, None,
                                src_padding_mask, trg_padding_mask, memory_key_padding_mask)
        return self.generator(outs)

    def encode(self, src: Tensor, src_mask: Tensor):
        return self.transformer.encoder(self.positional_encoding(
                            self.src_tok_emb(src)), src_mask)

    def decode(self, trg: Tensor, memory: Tensor, trg_mask: Tensor):
        return self.transformer.decoder(self.positional_encoding(
                          self.trg_tok_emb(trg)), memory,
                          trg_mask)

