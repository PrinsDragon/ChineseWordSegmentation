import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_packed_sequence, pack_padded_sequence

Begin = 0
Middle = 1
End = 2
Single = 3
SOS = 4
EOS = 5

"""Init: input_size, hidden_size, bidirectional=True, batch_first=True"""
"""Forward: vec_seq, len_seq"""


class LSTM(nn.Module):
    def __init__(self, input_size, hidden_size, layer_num=1, bidirectional=True, batch_first=True):
        super(LSTM, self).__init__()
        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size, bidirectional=bidirectional,
                            num_layers=layer_num, batch_first=batch_first)
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


class ModelCRF(nn.Module):
    def __init__(self, vocab_size, embedding_dim, fc_dim, hidden_size, layer_num=2, word_vec_matrix=None, tag_num=6,
                 batch_first=True, bidirectional=True, dropout=0.5):
        super(ModelCRF, self).__init__()

        self.tagset_size = tag_num

        self.embedding = nn.Embedding(vocab_size, embedding_dim)

        if word_vec_matrix is not None:
            self.embedding.weight.data.copy_(word_vec_matrix)

        self.basic_lstm = LSTM(input_size=embedding_dim, hidden_size=hidden_size, batch_first=batch_first,
                               layer_num=layer_num, bidirectional=bidirectional)

        self.classifier = nn.Sequential(
            nn.Linear(2 * hidden_size, fc_dim),
            nn.Dropout(dropout),
            nn.ReLU(),
            nn.Linear(fc_dim, fc_dim),
            nn.Dropout(dropout),
            nn.ReLU(),
            nn.Linear(fc_dim, tag_num)
        )

        # for CRF
        # Matrix of transition parameters.  Entry i,j is the score of
        # transitioning *to* i *from* j.
        self.transitions = nn.Parameter(torch.Tensor(self.tagset_size, self.tagset_size))
        init.xavier_normal_(self.transitions.data)

        # These two statements enforce the constraint that we never transfer
        # to the start tag and we never transfer from the stop tag
        self.transitions.data[SOS, :] = -10000
        self.transitions.data[:, EOS] = -10000

    def _get_lstm_features(self, text, length):
        embed = self.embedding(text)
        lstm_out = self.basic_lstm(embed, length)
        score = self.classifier(lstm_out)
        return score

    def _forward_alg(self, feats, length):
        # Do the forward algorithm to compute the partition function
        init_alphas = torch.full((1, self.tagset_size), -10000.).cuda()
        # START_TAG has all of the score.
        init_alphas[0][SOS] = 0.

        # Wrap in a variable so that we will get automatic backprop
        forward_var = init_alphas

        # Iterate through the sentence
        for i in range(length):
            feat = feats[i]
            alphas_t = []  # The forward tensors at this timestep
            for next_tag in range(self.tagset_size):
                # broadcast the emission score: it is the same regardless of
                # the previous tag
                emit_score = feat[next_tag].view(
                    1, -1).expand(1, self.tagset_size)
                # the ith entry of trans_score is the score of transitioning to
                # next_tag from i
                trans_score = self.transitions[next_tag].view(1, -1)
                # The ith entry of next_tag_var is the value for the
                # edge (i -> next_tag) before we do log-sum-exp
                next_tag_var = forward_var + trans_score + emit_score
                # The forward variable for this tag is log-sum-exp of all the
                # scores.
                alphas_t.append(log_sum_exp(next_tag_var).view(1))
            forward_var = torch.cat(alphas_t).view(1, -1)
        terminal_var = forward_var + self.transitions[EOS]
        alpha = log_sum_exp(terminal_var)
        return alpha

    def _score_sentence(self, feats, length, tags):
        # Gives the score of a provided tag sequence
        score = torch.zeros(1).cuda()
        tags = torch.cat([torch.Tensor([SOS]).long().cuda(), tags])
        for i in range(length):
            feat = feats[i]
            score = score + \
                self.transitions[tags[i + 1], tags[i]] + feat[tags[i + 1]]
        score = score + self.transitions[EOS, tags[-1]]
        return score

    def neg_log_likelihood(self, text, length, tags):
        feats = self._get_lstm_features(text, length)
        forward_score = torch.cat([self._forward_alg(feats[i], int(length[i])).view(1, -1) for i in range(feats.shape[0])], 0)
        gold_score = torch.cat([self._score_sentence(feats[i], int(length[i]), tags[i]).view(1, -1) for i in range(feats.shape[0])], 0)
        return forward_score - gold_score

    def _viterbi_decode(self, feats, length):
        backpointers = []

        # Initialize the viterbi variables in log space
        init_vvars = torch.full((1, self.tagset_size), -10000.).cuda()
        init_vvars[0][SOS] = 0

        # forward_var at step i holds the viterbi variables for step i-1
        forward_var = init_vvars
        for i in range(length):
            feat = feats[i]
            bptrs_t = []  # holds the backpointers for this step
            viterbivars_t = []  # holds the viterbi variables for this step

            for next_tag in range(self.tagset_size):
                # next_tag_var[i] holds the viterbi variable for tag i at the
                # previous step, plus the score of transitioning
                # from tag i to next_tag.
                # We don't include the emission scores here because the max
                # does not depend on them (we add them in below)
                next_tag_var = forward_var + self.transitions[next_tag]
                best_tag_id = argmax(next_tag_var)
                bptrs_t.append(best_tag_id)
                viterbivars_t.append(next_tag_var[0][best_tag_id].view(1))
            # Now add in the emission scores, and assign forward_var to the set
            # of viterbi variables we just computed
            forward_var = (torch.cat(viterbivars_t) + feat).view(1, -1)
            backpointers.append(bptrs_t)

        # Transition to STOP_TAG
        terminal_var = forward_var + self.transitions[EOS]
        best_tag_id = argmax(terminal_var)
        path_score = terminal_var[0][best_tag_id]

        # Follow the back pointers to decode the best path.
        best_path = [best_tag_id]
        for bptrs_t in reversed(backpointers):
            best_tag_id = bptrs_t[best_tag_id]
            best_path.append(best_tag_id)
        # Pop off the start tag (we dont want to return that to the caller)
        start = best_path.pop()
        assert start == SOS  # Sanity check
        best_path.reverse()
        return path_score, best_path

    def forward(self, text, length):  # dont confuse this with _forward_alg above.
        # Get the emission scores from the BiLSTM
        lstm_feats = self._get_lstm_features(text, length)

        # Find the best path, given the features.
        score_batch = []
        tag_seq_batch = []
        batch_size = text.shape[0]
        for i in range(batch_size):
            score, tag_seq = self._viterbi_decode(lstm_feats[i], int(length[i]))
            score_batch.append(score)
            tag_seq_batch.append(tag_seq)
        return score_batch, tag_seq_batch

def argmax(vec):
    # return the argmax as a python int
    _, idx = torch.max(vec, 1)
    return idx.item()

# Compute log sum exp in a numerically stable way for the forward algorithm
def log_sum_exp(vec):
    max_score = vec[0, argmax(vec)]
    max_score_broadcast = max_score.view(1, -1).expand(1, vec.size()[1])
    return max_score + \
        torch.log(torch.sum(torch.exp(vec - max_score_broadcast)))
