import os
import torch

from torch.utils.data import DataLoader
from .data.Dataset import Seq2SeqDataset

from model.Seq2SeqTransformer import Seq2SeqTransformer 

from model.utils.masking import create_mask
from .data.utils import SimpleVocab, tokenizing, seq2seq_collate

from timeit import default_timer as timer

class Executor():
    def __init__(self, config):
        self.config = config
        self.tokenize = tokenizing
        self.collate_fn = seq2seq_collate
        
        self.eval_epoch = config.EVAL_EPOCH
        if config.SEP_VOCAB: 
            src_vocab_size, trg_vocab_size = self._create_dataloader()
        
        src_vocab_size = trg_vocab_size = self._create_dataloader()

        self.model = Seq2SeqTransformer(num_encoder_layers = config.NUM_ENCODER_LAYERS,
                                        num_decoder_layers = config.NUM_DECODER_LAYERS,
                                        emb_size = config.EMB_SIZE,
                                        nhead = config.NHEAD,
                                        src_vocab_size = src_vocab_size,
                                        trg_vocab_size = trg_vocab_size,
                                        dim_feedforward = config.DIM_FF)
        for p in self.model.parameters():
            if p.dim() > 1:
                torch.nn.init.xavier_uniform_(p)
        self.model = self.model.to(self.config.DEVICE)

        self.optim = torch.optim.Adam(self.model.parameters(), lr=config.LR, betas=config.BETAS, eps=config.EPS)

        self.loss_fn = torch.nn.CrossEntropyLoss(ignore_index=SimpleVocab.PAD)
        
        self.SAVE = config.SAVE
        self.save_model_epoch = config.SAVE_EPOCHS
        self.save_best = config.SAVE_BEST

    def train(self):
        if not self.config.SAVE_PATH:
            folder = 'Seq2Seq-Transformer/models_param'
        else:
            folder = self.config.SAVE_PATH

        if self.SAVE:
            if self.config.SEP_VOCAB:
                self.train_data.src_vocab.save_to_file(os.path.join(folder, "src_vocab.pth"))
                self.train_data.trg_vocab.save_to_file(os.path.join(folder, "trg_vocab.pth"))
            else:
                self.train_data.vocab.save_to_file(os.path.join(folder, "vocab.pth"))

        s_train_time = timer()

        for epoch in range(1, self.config.NUM_EPOCHS+1):
            start_time = timer()
            train_loss = self._train_epoch()
            end_time = timer()
            if self.eval_epoch:
                val_loss = self._evaluate_epoch()
                print((f"Epoch: {epoch}, Train loss: {train_loss:.3f}, Val loss: {val_loss:.3f}, "f"Epoch time = {(end_time - start_time):.3f}s"))
            else:
                print((f"Epoch: {epoch}, Train loss: {train_loss:.3f}, "f"Epoch time = {(end_time - start_time):.3f}s"))
            
            if self.SAVE:
                if epoch in self.save_model_epoch:
                    statedict = {
                        "state_dict": self.model.state_dict(),
                        "optimizer": self.optim.state_dict(),
                        "epoch": epoch
                    }

                    filename = f"E{epoch}_ckp.pth"
                    torch.save(statedict, os.path.join(folder,filename))
                    print(f"!---------Saved {filename}----------!")

        e_train_time = timer()

        print(f"#----------- TRAINING END-Time: { e_train_time-s_train_time} -----------------#")
    
    def evaluate(self):
        pass 

    def run(self, mode = 'train'):
        if mode =='train':
            self.train()
        elif mode == 'eval':
            self.evaluate()
        else:
            exit(-1)

    def _create_dataloader(self):
        if self.config.TRAIN_PATH != "":
            self.train_data = Seq2SeqDataset(filename = self.config.TRAIN_PATH,
                                            max_rows = self.config.TRAIN_MAX_ROW,
                                            min_freq = self.config.MIN_FREQ,
                                            truncate_src = self.config.TRUNCATE_SRC,
                                            max_src_len = self.config.MAX_SRC_LEN,
                                            truncate_trg = self.config.TRUNCATE_TRG,
                                            max_trg_len = self.config.MAX_TRG_LEN,
                                            tokenize = self.tokenize,
                                            sep_vocab = self.config.SEP_VOCAB)
            self.train_loader = DataLoader(dataset = self.train_data, batch_size=self.config.BATCH_SIZE, shuffle=True, collate_fn = self.collate_fn)
        
        if self.config.VAL_PATH != "":
            self.val_data = Seq2SeqDataset(filename = self.config.VAL_PATH,
                                            max_rows = self.config.VAL_MAX_ROW,
                                            min_freq = self.config.MIN_FREQ,
                                            truncate_src = self.config.TRUNCATE_SRC,
                                            max_src_len = self.config.MAX_SRC_LEN,
                                            truncate_trg = self.config.TRUNCATE_TRG,
                                            max_trg_len = self.config.MAX_TRG_LEN,
                                            tokenize = self.tokenize,
                                            sep_vocab = self.config.SEP_VOCAB)
            self.val_loader = DataLoader(dataset = self.train_data, batch_size=self.config.BATCH_SIZE, shuffle=True, collate_fn = self.collate_fn) 
        
        if self.config.SEP_VOCAB:
            return len(self.train_data.src_vocab), len(self.train_data.src_vocab)
        
        return len(self.train_data.vocab)

    def _train_epoch(self):
        self.model.train()
        losses = 0

        for batch in self.train_loader:
            src = batch['x'].to(self.config.DEVICE)
            trg = batch['y'].to(self.config.DEVICE)


            trg_input = trg[:-1, :]

            src_mask, trg_mask, src_padding_mask, trg_padding_mask = create_mask(src, trg_input)

            logits = self.model(src, trg_input, src_mask, trg_mask,src_padding_mask, trg_padding_mask, src_padding_mask)

            self.optim.zero_grad()

            trg_out = trg[1:, :]
            loss = self.loss_fn(logits.reshape(-1, logits.shape[-1]), trg_out.reshape(-1))
            loss.backward()

            self.optim.step()
            losses += loss.item()

        return losses / len(list(self.save_besttrain_loader))


    def _evaluate_epoch(self):
        self.model.eval()
        losses = 0

        for batch in self.val_loader:
            src = batch['x'].to(self.config.DEVICE)
            trg = batch['y'].to(self.config.DEVICE)

            trg_input = trg[:-1, :]

            src_mask, trg_mask, src_padding_mask, trg_padding_mask = create_mask(src, trg_input)

            logits = self.model(src, trg_input, src_mask, trg_mask, src_padding_mask, trg_padding_mask, src_padding_mask)

            trg_out = trg[1:, :]
            loss = self.loss_fn(logits.reshape(-1, logits.shape[-1]), trg_out.reshape(-1))
            losses += loss.item()

        return losses / len(list(self.val_loader))