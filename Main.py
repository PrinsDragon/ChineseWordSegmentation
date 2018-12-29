import pickle
import os

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

from Model import Model, ModelCRF

directory = "checkpoint"
OUTPUT_FILE = open("{}/output.txt".format(directory), "w", encoding="utf-8")
def output(string, end="\n"):
    print(string, end=end)
    print(string, end=end, file=OUTPUT_FILE)

dict_file = open("data/proc/dict.pl", "rb")
matrix_file = open("data/proc/matrix.pl", "rb")
text_file = [open("data/proc/train/text.pl", "rb"), open("data/proc/test/text.pl", "rb")]
tag_file = [open("data/proc/train/tag.pl", "rb"), open("data/proc/test/tag.pl", "rb")]
len_file = [open("data/proc/train/len.pl", "rb"), open("data/proc/test/len.pl", "rb")]

word_dict = pickle.load(dict_file)
word_matrix = pickle.load(matrix_file)
text = [pickle.load(text_file[0]), pickle.load(text_file[1])]
tag = [pickle.load(tag_file[0]), pickle.load(tag_file[1])]
length = [pickle.load(len_file[0]), pickle.load(len_file[1])]

epoch = 50
batch_size = 100
vocabulary_size = len(word_dict)
embedding_dim = 128
fc_dim = 512
hidden_size = 128


class TextDataset(Dataset):
    def __init__(self, text_, tag_, length_):
        super(TextDataset, self).__init__()
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


train_data_set = TextDataset(text[0], tag[0], length[0])
train_data_loader = DataLoader(train_data_set, batch_size=batch_size, shuffle=True, drop_last=True)

test_data_set = TextDataset(text[1], tag[1], length[1])
test_data_loader = DataLoader(test_data_set, batch_size=batch_size, shuffle=True, drop_last=True)

model = ModelCRF(vocab_size=vocabulary_size, embedding_dim=embedding_dim, fc_dim=fc_dim, hidden_size=hidden_size,
                 word_vec_matrix=word_matrix).cuda()

optimizer = optim.Adam(model.parameters(), lr=0.001)
loss_func = nn.CrossEntropyLoss(reduce=True, size_average=False)


plot_loss = {"train": []}

def train(epoch_id):
    model.train()
    running_loss = 0.
    word_num = 0.
    for batch_index, batch in enumerate(train_data_loader):
        if batch_index % 100 == 0:
            output("{} sentence finish...".format(batch_index * batch_size))

        text_batch = batch[0].cuda()
        length_batch = batch[1].view(-1)
        tag_batch = batch[2].cuda()

        optimizer.zero_grad()

        loss = model.neg_log_likelihood(text_batch, length_batch, tag_batch)

        loss = sum(loss) / batch_size

        output(float(loss))

        plot_loss["train"].append(float(loss))

        loss.backward()
        optimizer.step()

        running_loss += float(loss)
        word_num += int(sum(length_batch))

    running_loss /= word_num
    output("Train: [%d/%d] Loss: %.5f" % (epoch_id + 1, epoch, running_loss))

def evaluate(epoch_id):
    model.eval()
    running_acc = 0.
    word_num = 0.
    for batch_index, batch in enumerate(test_data_loader):
        text_batch = batch[0].cuda()
        length_batch = batch[1].view(-1)
        tag_batch = batch[2].cuda()

        score_batch, predict_batch = model(text_batch, length_batch)

        correct_num = 0.
        for i in range(batch_size):
            cur_length = int(length_batch[i])
            cur_tag = tag_batch[i][:cur_length]
            cur_predict = predict_batch[i]
            correct_num += sum([1 if cur_predict[j] == int(cur_tag[j]) else 0 for j in range(cur_length)])
            word_num += cur_length

        running_acc += correct_num

    running_acc /= word_num
    output("Evaluate: [%d/%d] Acc: %.5f" % (epoch_id + 1, epoch, running_acc))


for i in range(epoch):
    train(i)
    evaluate(i)
    torch.save(model.state_dict(), "checkpoint/EPOCH_{}_model.pkl".format(i))
