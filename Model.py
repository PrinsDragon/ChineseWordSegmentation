import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_packed_sequence, pack_padded_sequence

"""Init: input_size, hidden_size, bidirectional=True, batch_first=True"""
"""Forward: vec_seq, len_seq"""


class LSTM(nn.Module):
    def __init__(self, input_size, hidden_size, bidirectional=True, batch_first=True):
        super(LSTM, self).__init__()
        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size, bidirectional=bidirectional,
                            batch_first=batch_first)
        # self.hidden = self.init_hidden()
        #
        # self.batch_size = batch_size
        # self.layer_num = 2 if bidirectional else 1

    def init_hidden(self):
        # Before we've done anything, we dont have any hidden state.
        # Refer to the Pytorch documentation to see exactly
        # why they have this dimensionality.
        # The axes semantics are (num_layers, minibatch_size, hidden_dim)
        return (torch.zeros(self.layer_num, self.batch_size, self.hidden_dim),
                torch.zeros(self.layer_num, self.batch_size, self.hidden_dim))

    def forward(self, vec_seq, len_seq):
        """
        :param vec_seq: tensor max_len x vec_dim with padding
        :param len_seq: tensor max_len
        :return: lstm out
        """
        # sort
        _, index = torch.sort(len_seq, descending=True)
        _, unsort_index = torch.sort(index, descending=False)
        vec_seq = vec_seq[index]
        len_seq = len_seq[index]

        # pack and pad
        vec_seq = pack_padded_sequence(vec_seq, len_seq.data.numpy(), batch_first=True)
        lstm_out, _ = self.lstm(vec_seq)
        # lstm_out, self.hidden = self.lstm(vec_seq, self.hidden)
        lstm_out, _ = pad_packed_sequence(lstm_out, batch_first=True)

        # unsort
        lstm_out = lstm_out[unsort_index]

        return lstm_out


# class DependenceNet(nn.Module):
#     def __init__(self, window_r, input_size, hidden_size, bidirectional=True, batch_first=True):
#         super(DependenceNet, self).__init__()
#         self.basic_lstm = LSTM(input_size=input_size, hidden_size=hidden_size,
#                                bidirectional=bidirectional, batch_first=batch_first)
#
#     def forward(self):
#         pass

class Model(nn.Module):
    def __init__(self, vocab_size, embedding_dim, fc_dim, hidden_size, tag_num=4,
                 bidirectional=True, batch_first=True, dropout=0.5):
        super(Model, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.basic_lstm = LSTM(input_size=embedding_dim, hidden_size=hidden_size,
                               bidirectional=bidirectional, batch_first=batch_first)

        self.classifier = nn.Sequential(
            nn.Linear(2 * hidden_size, fc_dim),
            nn.Dropout(dropout),
            nn.ReLU(),
            nn.Linear(fc_dim, fc_dim),
            nn.Dropout(dropout),
            nn.ReLU(),
            nn.Linear(fc_dim, tag_num)
        )

    def forward(self, text, length):
        embed = self.embedding(text)
        lstm_out = self.basic_lstm(embed, length)
        score = self.classifier(lstm_out)
        return score
