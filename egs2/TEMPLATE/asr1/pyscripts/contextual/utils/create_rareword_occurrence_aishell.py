import os
import numpy as np

from tqdm import tqdm
from dataio import read_file
from dataio import read_json
from dataio import write_file

TRAIN_DEV_BLIST_PATH = "/share/nas169/yuchunliu/espnet_contextual/espnet/egs2/aishell/asr2/id_ner_list/aishell_ner_train_dev.txt"
TEST_BLIST_PATH      = "/share/nas169/yuchunliu/espnet_contextual/espnet/egs2/aishell/asr2/id_ner_list/aishell_ner_test.txt"

def occurrence(texts, bwords):
    bword_occurrence = {word: 0 for word in bwords}
    oov = 0
    for uid, words in texts:
        for word in words:
            if word not in bword_occurrence:
                oov += 1
                continue
            bword_occurrence[word] += 1
    return list(bword_occurrence.values()) + [oov]

if __name__ == '__main__':
    text_path  = '/share/nas169/yuchunliu/espnet_contextual/espnet/egs2/aishell/asr1/dump/raw/zh_train_sp/text'
    dev_path   = '/share/nas169/yuchunliu/espnet_contextual/espnet/egs2/aishell/asr1/dump/raw/zh_dev/text'
    dump_path  = '/share/nas169/yuchunliu/espnet_contextual/espnet/egs2/aishell/asr1/local/contextual/rarewords'
    name       = TRAIN_DEV_BLIST_PATH.split('/')[-1].replace('.txt', '')

    text_datas = [[d[0], d[1:]] for d in read_file(text_path, sp=' ')]
    dev_datas  = [[d[0], d[1:]] for d in read_file(dev_path, sp=' ')]
    blist      = [b[0] for b in read_file(TRAIN_DEV_BLIST_PATH, sp=' ')]
    
    text_datas = text_datas + dev_datas
    counts     = list(map(lambda x: [str(x)], occurrence(text_datas, blist)))

    output_path = os.path.join(dump_path, f'{name}_occurrence.txt')
    write_file(output_path, counts)
print(output_path)
print("done")