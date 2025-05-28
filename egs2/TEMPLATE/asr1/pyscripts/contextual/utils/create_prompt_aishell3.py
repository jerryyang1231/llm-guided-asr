import os
import random
from tqdm import tqdm
from dataio import read_file
from dataio import write_file

TRAIN_DEV_BLIST_PATH = "/share/nas169/yuchunliu/espnet_contextual/espnet/egs2/aishell/asr1/local/contextual/rarewords/rareword_f1000_train.txt"
# /share/nas169/yuchunliu/espnet_contextual/espnet/egs2/esun/asr1/local/contextual/rarewords/rareword_f1000_train.txtesun300_entity.txt
TEST_BLIST_PATH      = "/share/nas169/yuchunliu/espnet_contextual/espnet/egs2/aishell/asr1/local/contextual/rarewords/rareword_f10_test.txt"
# /share/nas169/yuchunliu/espnet_contextual/espnet/egs2/esun/asr1/local/contextual/rarewords/rareword_f10_test.txt

PROMPT_TEMPLATE         = '''主題為: {}. 開始吧.'''
PROMPT_NON_ENT_TEMPLATE = '''開始吧.'''

def get_uttblist(words, blist):
    matched = []
    for word in words:
        for entity in blist:
            if entity in word:  # 使用 in 判斷是否匹配子字串
                matched.append([str(word2idx[entity]), entity])
    # 根據 word2idx 的索引排序，確保順序一致
    # matched.sort(key=lambda x: int(x[0]))
    return matched

def generate_prompt(uttblist, shuffle_blist=False):
    # Shuffle the uttblist if shuffle_blist is True
    if shuffle_blist and len(uttblist) > 0:
        random.shuffle(uttblist)

    # Generate prompt using the specified template
    if len(uttblist) > 0:
        return PROMPT_TEMPLATE.format(", ".join(uttblist))
    else:
        return PROMPT_NON_ENT_TEMPLATE

if __name__ == '__main__':
    datas_path = '/share/nas169/yuchunliu/espnet_contextual/espnet/egs2/aishell/asr1/dump/raw'
    shuffle_blist = False  # Set to True to enable random sorting
    for folder in os.listdir(datas_path):
        path = os.path.join(datas_path, folder)
        if not os.path.isfile(os.path.join(path, 'wav.scp')):
            continue
        if 'test' not in path:
            blist_path = TRAIN_DEV_BLIST_PATH
        else:
            blist_path = TEST_BLIST_PATH
        blist = [b[0] for b in read_file(blist_path, sp=' ')]
        word2idx = {word: i for i, word in enumerate(blist)}

        print(f'processing {path}...')
        text_path = os.path.join(path, 'text')
        text_datas = read_file(text_path, sp=' ')

        rareword_datas = []
        for data in tqdm(text_datas):
            uttid = data[0]
            results = get_uttblist(data[1:], blist)
            uttblist = [d[1] for d in results]

            # Generate the prompt with optional shuffling
            prompt = generate_prompt(uttblist, shuffle_blist=False)
            rareword_datas.append([uttid, prompt.upper()])

        output_path_uttblist = os.path.join(path, 'prompt_rareword_gt')
        write_file(output_path_uttblist, rareword_datas)