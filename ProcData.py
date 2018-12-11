import pickle
import numpy as np
from gensim.models import Word2Vec

def get_id(char):
    if char in character_dict:
        return character_dict[char]
    else:
        new_id = len(character_dict) + 1
        character_dict[char] = new_id
        return new_id

def data_padding(seq):
    len_list = [len(s) for s in seq]
    max_len = max(len_list)
    for i in range(len(seq)):
        cur_len = len(seq[i])
        pad = [0 for _ in range(max_len-cur_len)]
        seq[i] = seq[i] + pad
    return seq, len_list

def proc_line(line):
    line.replace("“", "")
    line.replace("”", "")


def proc_file(file_name):
    path = "data/{}.txt".format(file_name)
    file = open(path, "r", encoding="utf-8")
    text_list_str = []
    text_list = []
    tag_list = []

    for i, line in enumerate(file):
        if i % 10000 == 0:
            print("{} lines finish...".format(i))

        line = line[: -1]
        part_list = line.split("  ")

        now_text_str = []
        now_text = []
        now_tag = []
        for part in part_list:
            part_len = len(part)
            if part_len == 0:
                continue
            if part_len == 1:
                now_text_str.append(part)
                now_text.append(get_id(part))
                now_tag.append(Single)
            else:
                now_text_str.append(part[0])
                now_text.append(get_id(part[0]))
                now_tag.append(Begin)
                for j in range(1, part_len - 1):
                    now_text_str.append(part[j])
                    now_text.append(get_id(part[j]))
                    now_tag.append(Middle)
                now_text_str.append(part[-1])
                now_text.append(get_id(part[-1]))
                now_tag.append(End)
        text_list_str.append(now_text_str)
        text_list.append(now_text)
        tag_list.append(now_tag)

    model = Word2Vec(text_list_str, min_count=1, size=128)

    text_list, len_list = data_padding(text_list)
    tag_list, _ = data_padding(tag_list)

    text_file = open("data/proc/{}/text.pl".format(file_name), "wb")
    tag_file = open("data/proc/{}/tag.pl".format(file_name), "wb")
    len_file = open("data/proc/{}/len.pl".format(file_name), "wb")

    pickle.dump(text_list, text_file)
    pickle.dump(tag_list, tag_file)
    pickle.dump(len_list, len_file)

    return model

if __name__ == "__main__":
    # tag:
    # {B:begin, M:middle, E:end, S:single}
    Begin = 0
    Middle = 1
    End = 2
    Single = 3

    character_dict = {}

    train_model = proc_file("train")
    test_model = proc_file("test")

    word_num = len(character_dict)
    word_vec_dim = 128
    word_vec_matrix = 0.5 * np.random.random_sample((word_num + 1, word_vec_dim)) - 0.25

    for word in character_dict:
        cur_id = character_dict[word]
        if word in train_model.wv:
            cur_vec = train_model.wv[word]
        else:
            cur_vec = test_model.wv[word]
        word_vec_matrix[cur_id] = cur_vec

    matrix_file = open("data/proc/matrix.pl", "wb")
    pickle.dump(word_vec_matrix, matrix_file)
    dict_file = open("data/proc/dict.pl", "wb")
    pickle.dump(character_dict, dict_file)

