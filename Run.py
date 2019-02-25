import pickle
import numpy as np
import copy

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from Model import Model

checkpoint_path = "checkpoint_12-8/EPOCH_5_model.pkl"

dict_file = open("data/proc/dict.pl", "rb")
text_file = open("data/proc/test/text.pl", "rb")
tag_file = open("data/proc/test/tag.pl", "rb")
len_file = open("data/proc/test/len.pl", "rb")

text = pickle.load(text_file)
tag = pickle.load(tag_file)
length = pickle.load(len_file)

word_dict = pickle.load(dict_file)
rev_word_dict = {}
for word in word_dict:
    word_id = word_dict[word]
    rev_word_dict[word_id] = word

batch_size = 100
vocabulary_size = len(word_dict)
embedding_dim = 128
fc_dim = 512
hidden_size = 128


class TextDataset(Dataset):
    def __init__(self, text_, tag_, length_):
        self.text = text_
        self.tag = tag_
        self.len_list = length_

        self.word_num = sum(length_)
        self.len = len(text_)

    def __len__(self):
        return self.len

    def __getitem__(self, index):
        return torch.Tensor(self.text[index]).long(), \
               torch.Tensor([self.len_list[index]]).long(), \
               torch.Tensor(self.tag[index]).long()


test_data_set = TextDataset(text, tag, length)
test_data_loader = DataLoader(test_data_set, batch_size=batch_size, shuffle=False)

model = Model(vocab_size=vocabulary_size, embedding_dim=embedding_dim, fc_dim=fc_dim, hidden_size=hidden_size).cuda()
model.load_state_dict(torch.load(checkpoint_path))

model.eval()

result = []

for batch_index, batch in enumerate(test_data_loader):
    text_batch = batch[0].cuda()
    length_batch = batch[1].view(-1)
    tag_batch = batch[2].cuda()

    score = model(text_batch, length_batch)

    for i in range(len(score)):
        cur_length = int(length_batch[i])
        cur_score = score[i][:cur_length]
        cur_predict = cur_score.max(1)[1]

        softmax_cur_score = F.softmax(cur_score, dim=0)
        softmax_cur_score = softmax_cur_score.cpu().detach().numpy()
        log_cur_score = -np.log(softmax_cur_score)
        nodes = [dict(zip(['b', 'm', 'e', 's'], i)) for i in log_cur_score]

        """
        Begin = 0
        Middle = 1
        End = 2
        Single = 3
        """
        cur_result = []
        for j in range(cur_length):
            char_id = int(text_batch[i][j])
            char = rev_word_dict[char_id]
            pred = cur_predict[j]
            if pred == 0 or pred == 3:
                cur_result.append(char)
            else:
                try:
                    cur_result[-1] += char
                except:
                    cur_result.append(char)
        result.append(cur_result)

save_path = "data/result.txt"
save_file = open(save_path, "w", encoding="utf-8")
for sentence in result:
    for word in sentence:
        save_file.write("{}  ".format(word))
    save_file.write("\n")

print("Finish!")
