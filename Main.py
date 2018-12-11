import pickle
import sys

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

from Model import Model

dict_file = open("data/proc/dict.pl", "rb")
text_file = [open("data/proc/train/text.pl", "rb"), open("data/proc/test/text.pl", "rb")]
tag_file = [open("data/proc/train/tag.pl", "rb"), open("data/proc/test/tag.pl", "rb")]
len_file = [open("data/proc/train/len.pl", "rb"), open("data/proc/test/len.pl", "rb")]

word_dict = pickle.load(dict_file)
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

model = Model(vocab_size=vocabulary_size, embedding_dim=embedding_dim, fc_dim=fc_dim, hidden_size=hidden_size).cuda()
optimizer = optim.Adam(model.parameters(), lr=0.001)
loss_func = nn.CrossEntropyLoss(reduce=True, size_average=False)

def train(epoch_id):
    model.train()
    running_loss = 0.
    running_acc = 0.
    word_num = 0.
    for batch_index, batch in enumerate(train_data_loader):
        if batch_index % 100 == 0:
            print("{} sentence finish...".format(batch_index * batch_size))

        text_batch = batch[0].cuda()
        length_batch = batch[1].view(-1)
        tag_batch = batch[2].cuda()

        optimizer.zero_grad()

        loss, predict = model(text_batch, length_batch, tag_batch)

        correct_num = 0.

        for i in range(batch_size):
            cur_length = int(length_batch[i])
            cur_tag = tag_batch[i][:cur_length]
            cur_tag = [int(t) for t in cur_tag]
            cur_predict = predict[i]
            correct_num += sum([1 if t == p else 0 for (t, p) in zip(cur_tag, cur_predict)])
            word_num += cur_length

        # loss = 0.
        # correct_num = 0.
        # for i in range(batch_size):
        #     cur_length = int(length_batch[i])
        #     cur_score = score[i][:cur_length]
        #     cur_tag = tag_batch[i][:cur_length]
        #     cur_predict = cur_score.max(1)[1]
        #     loss += loss_func(cur_score, cur_tag)
        #     correct_num += (cur_predict == cur_tag).sum()
        #     word_num += cur_length

        loss.backward()
        optimizer.step()

        running_loss += loss.data
        running_acc += correct_num

    running_loss /= word_num
    running_acc /= word_num
    print("Train: [%d/%d] Loss: %.5f, Acc: %.5f" % (epoch_id + 1, epoch, running_loss, running_acc))

    return running_acc

def evaluate(epoch_id):
    model.eval()
    running_loss = 0.
    running_acc = 0.
    word_num = 0.
    for batch_index, batch in enumerate(train_data_loader):
        text_batch = batch[0].cuda()
        length_batch = batch[1].view(-1)
        tag_batch = batch[2].cuda()

        loss, predict = model(text_batch, length_batch, tag_batch)

        correct_num = 0.

        for i in range(batch_size):
            cur_length = int(length_batch[i])
            cur_tag = tag_batch[i][:cur_length]
            cur_tag = [int(t) for t in cur_tag]
            cur_predict = predict[i]
            correct_num += sum([1 if t == p else 0 for (t, p) in zip(cur_tag, cur_predict)])
            word_num += cur_length

        running_loss += loss.data
        running_acc += correct_num

    running_loss /= word_num
    running_acc /= word_num
    print("Evaluate: [%d/%d] Loss: %.5f, Acc: %.5f" % (epoch_id + 1, epoch, running_loss, running_acc))

    return running_acc

def save_model(epoch_id, train_acc, eval_acc):
    file = open("checkpoint/EPOCH_{}_Detail.txt".format(epoch_id), "w", encoding="utf-8")
    file.write("Train ACC: {:.5f}, Test ACC: {:.5f}".format(train_acc, eval_acc))
    torch.save(model.state_dict(), "checkpoint/EPOCH_{}_model.pkl".format(epoch_id))


for i in range(epoch):
    eval_acc = evaluate(i)
    sys.stdout.flush()
    train_acc = train(i)
    sys.stdout.flush()
    if (i + 1) % 5 == 0:
        save_model(i, train_acc, 0)
