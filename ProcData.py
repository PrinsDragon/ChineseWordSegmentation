import pickle

path = "data/train.txt"
file = open(path, "r", encoding="utf-8")
character_dict = {}
text_list = []
tag_list = []

# tag:
# {B:begin, M:middle, E:end, S:single}
Begin = 0
Middle = 1
End = 2
Single = 3


def get_id(char):
    if char in character_dict:
        return character_dict[char]
    else:
        new_id = len(character_dict) + 1
        character_dict[char] = new_id
        return new_id


for i, line in enumerate(file):
    if i % 10000 == 0:
        print("{} lines finish...".format(i))
    line = line[: -1]
    part_list = line.split("  ")
    now_text = []
    now_tag = []
    for part in part_list:
        part_len = len(part)
        if part_len == 0:
            continue
        if part_len == 1:
            now_text.append(get_id(part))
            now_tag.append(Single)
        else:
            now_text.append(get_id(part[0]))
            now_tag.append(Begin)
            for j in range(1, part_len - 1):
                now_text.append(get_id(part[j]))
                now_tag.append(Middle)
            now_text.append(get_id(part[-1]))
            now_tag.append(End)
    text_list.append(now_text)
    tag_list.append(now_tag)

dict_file = open("data/proc/dict.pl", "wb")
text_file = open("data/proc/text.pl", "wb")
tag_file = open("data/proc/tag.pl", "wb")

pickle.dump(character_dict, dict_file)
pickle.dump(text_list, text_file)
pickle.dump(tag_list, tag_file)
