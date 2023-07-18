import re
import string
import unidecode
import torch
from Vocab import SimpleVocab

def remove_emoji(string):
    emoji_pattern = re.compile("["
                               u"\U0001F600-\U0001F64F"  # emoticons
                               u"\U0001F300-\U0001F5FF"  # symbols & pictographs
                               u"\U0001F680-\U0001F6FF"  # transport & map symbols
                               u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
                               u"\U00002500-\U00002BEF"  # chinese char
                               u"\U00002702-\U000027B0"
                               u"\U00002702-\U000027B0"
                               u"\U000024C2-\U0001F251"
                               u"\U0001f926-\U0001f937"
                               u"\U00010000-\U0010ffff"
                               u"\u2640-\u2642"
                               u"\u2600-\u2B55"
                               u"\u200d"
                               u"\u23cf"
                               u"\u23e9"
                               u"\u231a"
                               u"\ufe0f"  # dingbats
                               u"\u3030"
                               "]+", flags=re.UNICODE)
    return emoji_pattern.sub(r'', string)

def remove_punctuation(text): # remove all punctuations in text
    punctuationfree="".join([i for i in text if i not in string.punctuation])

    return punctuationfree

def reconstruct_punctuation(text):
    text = re.sub(r'([a-zA-Z])([(),.!?])', '\\1 \\2' , text)
    text = re.sub(r'([(),.!?])([a-zA-Z])', '\\1 \\2' , text)

    text = re.sub(r'([0-9])([()!?])', '\\1 \\2' , text)
    text = re.sub(r'([()!?])([0-9])', '\\1 \\2' , text)

    text = re.sub(r' ([.,])([0-9])', ' \\1 \\2' , text)
    text = re.sub(r'([0-9])([.,]) ', '\\1 \\2 ' , text)

    text = re.sub(r'([0-9]) ([0-9])', '\\1_\\2' , text)

    return text

def preprocessing(text, remove_diacritics = False):
    text = text.lower()                   # lowercase
    text = remove_emoji(text)             # remove emojis
    if remove_diacritics:
      text  = unidecode.unidecode(text)   # remove diacritics
    else:
      text = reconstruct_punctuation(text)

    return text


#Tokenization

def tokenize_sentences(text): # add <SEP>
    alphabets= "([A-Za-z])"
    prefixes = "(Mr|St|Mrs|Ms|Dr|Miss)[.]"
    suffixes = "(Inc|Ltd|Jr|Sr|Co)"
    starters = "(Mr|Mrs|Ms|Dr|Prof|Capt|Cpt|Lt)"
    websites = "[.](com|net|org|io|gov|edu|me)"
    digits = "([0-9])"

    text = " " + text + "  "
    text = text.replace("\n"," ")
    text = re.sub(prefixes,"\\1<prd>",text)
    text = re.sub(websites,"<prd>\\1",text)
    text = re.sub(digits + "[.]" + digits,"\\1<prd>\\2",text)

    if "..." in text: text = text.replace("...","<prd><prd><prd> <SEP>")
    if "Ph.D" in text: text = text.replace("Ph.D.","Ph<prd>D<prd>")
    text = re.sub("\s" + alphabets + "[.] "," \\1<prd> ",text)
    text = re.sub(alphabets + "[.]" + alphabets + "[.]" + alphabets + "[.]","\\1<prd>\\2<prd>\\3<prd>",text)
    text = re.sub(alphabets + "[.]" + alphabets + "[.]","\\1<prd>\\2<prd>",text)
    text = re.sub(" "+suffixes+"[.] "+starters," \\1<stop> \\2",text)
    text = re.sub(" "+suffixes+"[.]"," \\1<prd>",text)
    text = re.sub(" " + alphabets + "[.]"," \\1<prd>",text)
    if "”" in text: text = text.replace(".”","”.")
    if "\"" in text: text = text.replace(".\"","\".")
    if "!" in text: text = text.replace("!\"","\"!")
    if "?" in text: text = text.replace("?\"","\"?")
    text = text.replace(".",". <SEP>")
    text = text.replace("?","? <SEP>")
    text = text.replace("!","! <SEP>")
    text = text.replace("<prd>",".")

    tokens = re.split(" +", text.strip())

    return tokens


def tokenize_words(text):
    tokens = re.split(" +", text.strip())

    return tokens


def tokenizing(text, have_sep:bool = False, remove_diacritics:bool = False):
    text = preprocessing(text, remove_diacritics)

    if have_sep:
        return tokenize_sentences(text)

    return tokenize_words(text)

def seq2seq_collate(batch): # collate function for Dataloader

    src_batch = [torch.Tensor([SimpleVocab.BOS] + item['x'] + [SimpleVocab.EOS]) for item in batch]
    trg_batch = [torch.Tensor([SimpleVocab.BOS] + item['y'] + [SimpleVocab.EOS]) for item in batch]

    x = torch.nn.utils.rnn.pad_sequence(src_batch, padding_value=SimpleVocab.PAD)
    y = torch.nn.utils.rnn.pad_sequence(trg_batch, padding_value=SimpleVocab.PAD)
    return {'x': x.type(torch.LongTensor),
            'y': y.type(torch.LongTensor),
            'x_len': [item['x_len']+2 for item in batch],
            'y_len': [item['y_len']+2 for item in batch]
            }

