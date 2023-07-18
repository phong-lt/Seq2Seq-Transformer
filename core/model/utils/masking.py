import torch

def generate_square_subsequent_mask(sz, DEVICE):
    mask = (torch.triu(torch.ones((sz, sz), device=DEVICE)) == 1).transpose(0, 1)
    mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
    return mask


def create_mask(src, trg, Vocab, DEVICE):
    src_seq_len = src.shape[0]
    trg_seq_len = trg.shape[0]

    trg_mask = generate_square_subsequent_mask(trg_seq_len)
    src_mask = torch.zeros((src_seq_len, src_seq_len),device=DEVICE).type(torch.bool)

    src_padding_mask = (src == Vocab.PAD).transpose(0, 1)
    trg_padding_mask = (trg == Vocab.PAD).transpose(0, 1)
    return src_mask, trg_mask, src_padding_mask, trg_padding_mask