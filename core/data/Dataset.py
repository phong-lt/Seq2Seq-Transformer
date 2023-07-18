from torch.utils.data import Dataset
import pandas as pd
from collections import Counter
from utils import SimpleVocab

class Seq2SeqDataset(Dataset):
  def __init__(self,
               filename,
               max_rows,
               min_freq,
               truncate_src,
               max_src_len,
               truncate_trg,
               max_trg_len,
               sep_vocab=False,
               name_vocab="data_vocab",
               save_vocab=False,
               save_dir="",
               tokenize = None):
      super(Seq2SeqDataset, self).__init__()
      print("Reading dataset {}...".format(filename))

      assert tokenize!=None, "Need a function for tokenizing"

      self.filename = filename
      self.pairs = []
      self.max_len = 0
      self.trg_len = 0
      self.max_rows = max_rows
      self.min_freq = min_freq
      self.sep_vocab = sep_vocab


      if self.max_rows is None:
        df = pd.read_excel(self.filename)
      else:
        df = pd.read_excel(self.filename, nrows=self.max_rows)

      sources = df['original'].apply(lambda x: tokenize(x))

      if truncate_src:
        sources = [src[:max_src_len] if len(src)>max_src_len else src for src in sources]

      targets = df['normalized'].apply(lambda x: tokenize(x))

      if truncate_trg:
        targets = [trg[:max_trg_len] if len(trg)>max_trg_len else trg for trg in targets]

      src_length = [len(src) for src in sources]
      trg_length = [len(trg) for trg in targets]

      max_src = max(src_length)
      max_trg = max(trg_length)

      self.max_len = max_src
      self.trg_len = max_trg

      self.pairs.append([(src, trg, src_len, trg_len) for src, trg, src_len, trg_len in zip(sources, targets, src_length, trg_length)])
      self.pairs = self.pairs[0]
      print(f"{len(self.pairs)} pairs were built.", flush=True)
      if sep_vocab:
        self.build_sep_vocab(name_vocab+"_src", name_vocab+"_trg")
        self.src_vocab.max_len = max_src
        self.trg_vocab.max_len = max_trg
      else:
        self.build_vocab(name_vocab, save_vocab, save_dir)
        self.vocab.max_len = max(max_src, max_trg)

  def build_vocab(self, name, save_vocab, save_dir)->SimpleVocab:
      total_words = [src+trg for src,trg, len_src, len_trg in self.pairs]
      total_words = [item for sublist in total_words for item in sublist]
      word_counts = Counter(total_words)
      vocab = SimpleVocab(name)
      for word, count in word_counts.items():
        if(count > self.min_freq):
          vocab.add_by_words([word])

      print(f"Vocab {vocab.name} of dataset {self.filename} was created.")

      if save_vocab:
        vocab.save_to_file(save_dir)
        print(f"Saved vocab {vocab.name} to {save_dir}.")

      self.vocab=vocab
      return vocab

  def build_sep_vocab(self, name_src, name_trg, save_vocab_src=False, save_dir_src="", save_vocab_trg=False, save_dir_trg=""):
      src_words = [src for src, trg, len_src, len_trg in self.pairs]
      src_words = [item for sublist in src_words for item in sublist]
      src_word_counts = Counter(src_words)

      src_vocab = SimpleVocab(name_src)
      for word, count in src_word_counts.items():
        if(count > self.min_freq):
          src_vocab.add_by_words([word])

      print(f"Vocab {src_vocab.name} of dataset {self.filename} was created.")

      if save_vocab_src:
        src_vocab.save_to_file(save_dir_src)
        print(f"Saved vocab {src_vocab.name} to {save_dir_src}.")

      self.src_vocab=src_vocab



      trg_words = [trg for src,trg, len_src, len_trg in self.pairs]
      trg_words = [item for sublist in trg_words for item in sublist]
      trg_word_counts = Counter(trg_words)

      trg_vocab = SimpleVocab(name_trg)
      for word, count in trg_word_counts.items():
        if(count > self.min_freq):
          trg_vocab.add_by_words([word])

      print(f"Vocab {trg_vocab.name} of dataset {self.filename} was created.")

      if save_vocab_trg:
        trg_vocab.save_to_file(save_dir_trg)
        print(f"Saved vocab {trg_vocab.name} to {save_dir_trg}.")

      self.trg_vocab=trg_vocab


      return src_vocab, trg_vocab

  def vectorize(self, tokens):
      return [self.vocab[token] for token in tokens]

  def unvectorize(self, indices):
      return [self.vocab[i] for i in indices]

  def __getitem__(self, index):
      if self.sep_vocab:
          return {'x':[self.src_vocab[token] for token in self.pairs[index][0]],
                  'y':[self.trg_vocab[token] for token in self.pairs[index][1]],
                  'src':self.pairs[index][0],
                  'trg':self.pairs[index][1],
                  'x_len':len(self.pairs[index][0]),
                  'y_len':len(self.pairs[index][1])}

      return {'x':self.vectorize(self.pairs[index][0]),
              'y':self.vectorize(self.pairs[index][1]),
              'src':self.pairs[index][0],
              'trg':self.pairs[index][1],
              'x_len':len(self.pairs[index][0]),
              'y_len':len(self.pairs[index][1])}

  def __len__(self):
      return len(self.pairs)