import math
from torch import nn, Tensor

class SimpleEmbedding(nn.Module):
    def __init__(self,
                 vocab_size: int,
                 emb_size):
        super(SimpleEmbedding, self).__init__()
        self.embedding = nn.Embedding(vocab_size, emb_size)
        self.emb_size = emb_size

    def forward(self, tokens: Tensor):
        return self.embedding(tokens.long()) * math.sqrt(self.emb_size)
