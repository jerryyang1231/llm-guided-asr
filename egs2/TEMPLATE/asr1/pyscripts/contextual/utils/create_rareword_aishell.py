import os
import numpy as np
import matplotlib.pyplot as plt

from tqdm import tqdm
from dataio import read_file
from dataio import read_json
from dataio import write_file

plt.rcParams['font.sans-serif'] = ['Microsoft JhengHei']
plt.rcParams['axes.unicode_minus'] = False

def get_word_count(datas):
    counts = {}
    for uid, words in datas:
        for word in words:
            if len(word) < 2:
                continue
            counts[word] = counts[word] + 1 if word in counts else 1
    counts = sorted([[counts[key], key] for key in counts], reverse=True)
    counts = {data[1]: data[0] for i, data in enumerate(counts)}
    return counts

def plot_word_count(output_path, data, tag=''):
    plt.figure(figsize=(20, 8))
    plt.bar(data.keys(), data.values(), color='skyblue')
    plt.title(f'Word Counts - {tag}')
    plt.xlabel('Word')
    plt.ylabel('Counts')
    # plt.xticks(range(max(data.keys()) + 1), rotation=90)
    plt.xticks(rotation=90)

    out_path = os.path.join(output_path, f'word_counts_{tag}.png')
    plt.savefig(out_path)

train_text_path    = "/share/nas169/yuchunliu/espnet_contextual/espnet/egs2/aishell/asr1_contextual/dump/raw/zh_train_sp/text"
dev_text_path      = "/share/nas169/yuchunliu/espnet_contextual/espnet/egs2/aishell/asr1_contextual/dump/raw/zh_dev/text"
test_text_path     = "/share/nas169/yuchunliu/espnet_contextual/espnet/egs2/aishell/asr1_contextual/dump/raw/zh_test/text"
dump_path          = "/share/nas169/yuchunliu/espnet_contextual/espnet/egs2/aishell/asr1_contextual/exp/test"
dump_rareword_path = "/share/nas169/yuchunliu/espnet_contextual/espnet/egs2/aishell/asr1_contextual/local/contextual/rarewords"

train_text_datas = [[d[0], d[1:]] for d in read_file(train_text_path, sp=' ') if not d[0].startswith("sp")]
dev_text_datas   = [[d[0], d[1:]] for d in read_file(dev_text_path, sp=' ') if not d[0].startswith("sp")]
test_text_datas  = [[d[0], d[1:]] for d in read_file(test_text_path, sp=' ') if not d[0].startswith("sp")]

text_datas = train_text_datas + dev_text_datas
counts = get_word_count(text_datas)

# plot_word_count(dump_path, counts, 'aishell')

word_occurance = 2 ** 10

rarewords = []
for word in counts:
    count = counts[word]
    if (count < word_occurance):
        rarewords.append([word])

print(f'rareword length: {len(rarewords)}')
output_path = os.path.join(dump_rareword_path, f'rareword_f{word_occurance}_train.txt')
print(dump_rareword_path)
write_file(output_path, rarewords, sp=' ')
print(output_path)
# text_datas = test_text_datas
# counts = get_word_count(text_datas)
# rarewords = []
# for word in counts:
#     count = counts[word]
#     if (count < 10):
#         rarewords.append([word])

# print(f'rareword length: {len(rarewords)}')
# output_path = os.path.join(dump_rareword_path, 'rareword_f10_test.txt')
# write_file(output_path, rarewords, sp=' ')