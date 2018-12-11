import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
from torch.nn.utils.rnn import pad_packed_sequence, pack_padded_sequence

from CRF import CRF


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

class Model(nn.Module):
    def __init__(self, vocab_size, embedding_dim, fc_dim, hidden_size, word_vec_matrix=None, tag_num=4,
                 bidirectional=True, batch_first=True, dropout=0.5):
        super(Model, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)

        if word_vec_matrix is not None:
            self.embedding.weight.data.copy_(word_vec_matrix)

        self.basic_lstm = LSTM(input_size=embedding_dim, hidden_size=hidden_size,
                               bidirectional=bidirectional, batch_first=batch_first)

        # self.CRF = CRF(tag_num)

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

# class BasicAttention(nn.Module):
#     def __init__(self, tensor_dim, dropout=0.1):
#         super(BasicAttention, self).__init__()
#         self.scale_constant = np.power(tensor_dim, 0.5)
#         self.dropout_layer = nn.Dropout(dropout)
#         self.softmax_layer = nn.Softmax(dim=2)
#
#     def forward(self, q, k, v, mask=None):
#         attention_matrix = q.bmm(k.transpose(1, 2)) / self.scale_constant
#
#         if mask is not None:
#             attention_matrix.masked_fill_(mask, -float('inf'))
#
#         attention_matrix = self.softmax_layer(attention_matrix)
#         attention_matrix = self.dropout_layer(attention_matrix)
#
#         output = attention_matrix.bmm(v)
#
#         return output

# class ProjectAttention(nn.Module):
#     def __init__(self, tensor_dim, k_dim, v_dim, dropout=0.1):
#         super(ProjectAttention, self).__init__()
#
#         self.k_dim = k_dim
#         self.v_dim = v_dim
#
#         self.w_qs = nn.Parameter(torch.FloatTensor(tensor_dim, k_dim))
#         self.w_ks = nn.Parameter(torch.FloatTensor(tensor_dim, k_dim))
#         self.w_vs = nn.Parameter(torch.FloatTensor(tensor_dim, v_dim))
#
#         self.basic_attention = BasicAttention(tensor_dim)
#         self.norm = LayerNormalization(tensor_dim)
#
#         self.dropout_layer = nn.Dropout(dropout)
#
#         init.xavier_normal_(self.w_qs)
#         init.xavier_normal_(self.w_ks)
#         init.xavier_normal_(self.w_vs)
#
#     def forward(self, q, k, v, attention_mask=None):
#
#         res = q
#
#         multi_q = q.mm(self.w_qs)
#         multi_k = k.mm(self.w_ks)
#         multi_v = v.mm(self.w_vs)
#
#         attention_output = self.basic_attention(multi_q, multi_k, multi_v)
#
#         dropout_output = self.dropout_layer(attention_output)
#
#         return self.norm(dropout_output + res)
#
#
# class LayerNormalization(nn.Module):
#     def __init__(self, d_hid, eps=1e-3):
#         super(LayerNormalization, self).__init__()
#
#         self.eps = eps
#         self.a_2 = nn.Parameter(torch.ones(d_hid), requires_grad=True)
#         self.b_2 = nn.Parameter(torch.zeros(d_hid), requires_grad=True)
#
#     def forward(self, z):
#         if z.size(1) == 1:
#             return z
#
#         mu = torch.mean(z, keepdim=True, dim=-1)
#         sigma = torch.std(z, keepdim=True, dim=-1)
#         ln_out = (z - mu.expand_as(z)) / (sigma.expand_as(z) + self.eps)
#         ln_out = ln_out * self.a_2.expand_as(ln_out) + self.b_2.expand_as(ln_out)
#
#         return ln_out

