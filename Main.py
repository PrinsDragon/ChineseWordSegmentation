import pickle

import torch
from torch.utils.data import Dataset, DataLoader

text_file = open("data/proc/text.pl", "rb")
tag_file = open("data/proc/tag.pl", "rb")

text = pickle.load(text_file)
tag = pickle.load(tag_file)

def data_padding(seq):
    len_list = [len(s) for s in seq]
    max_len = max(len_list)
    for i in range(len(seq)):
        cur_len = len(seq[i])
        pad = [0 for _ in range(max_len-cur_len)]
        seq[i] = seq[i] + pad
    return seq, len_list

class TextDataset(Dataset):
    def __init__(self, text_, tag_):
        self.text, self.len_list = data_padding(text_)
        self.tag, _ = data_padding(tag_)

        self.len = len(text_)

    def __len__(self):
        return self.len

    def __getitem__(self, index):
        return torch.Tensor(self.text[index]), torch.Tensor([self.len_list[index]]).long(), torch.Tensor(self.tag[index])


data_loader = DataLoader(TextDataset(text, tag), batch_size=5)

for t in data_loader:
    pass
