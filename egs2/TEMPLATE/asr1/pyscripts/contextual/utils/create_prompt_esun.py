import os
import random
from tqdm import tqdm
from dataio import read_file
from dataio import write_file

TRAIN_DEV_BLIST_PATH = "/share/nas169/yuchunliu/espnet_contextual/espnet/egs2/esun/asr1/local/contextual/rarewords/esun_earningcall.entity.txt"
# /share/nas169/yuchunliu/espnet_contextual/espnet/egs2/esun/asr1/local/contextual/rarewords/rareword_f1000_train.txtesun300_entity.txt
TEST_BLIST_PATH      = "/share/nas169/yuchunliu/espnet_contextual/espnet/egs2/esun/asr1/local/contextual/rarewords/esun_earningcall.entity.txt"
# /share/nas169/yuchunliu/espnet_contextual/espnet/egs2/esun/asr1/local/contextual/rarewords/rareword_f10_test.txt
# PROMPT_TEMPLATE         = '''玉山銀行法人說明會是一個定期舉辦的活動，向投資者和法人機構介紹公司財務狀況、業務策略及未來展望。該活動旨在增進透明度，讓法人了解公司的經營績效與市場動態。有出現: {}. 開始吧.'''
# PROMPT_NON_ENT_TEMPLATE = '''開始吧.'''
PROMPT_TEMPLATE         = '''主題為: {}. 開始吧.'''
PROMPT_NON_ENT_TEMPLATE = '''開始吧.'''

def get_uttblist(words, blist):
    # Filter words that are in the blist and remove duplicates
    unique_words = list(dict.fromkeys(words))  # Remove duplicates while preserving order
    return [[str(word2idx[word]), word] for word in unique_words if word in blist]

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
    datas_path = '/share/nas169/yuchunliu/espnet_contextual/espnet/egs2/esun/asr2_tts/dump/raw'
    shuffle_blist = False  # Set to True to enable random sorting
    for folder in os.listdir(datas_path):
        path = os.path.join(datas_path, folder)
        if not os.path.isfile(os.path.join(path, 'wav.scp')):
            continue
        if 'test' not in path:
            blist_path = TEST_BLIST_PATH
        else:
            blist_path = TRAIN_DEV_BLIST_PATH
        blist = [b[0] for b in read_file(blist_path, sp=' ')]

        # Create a mapping from word to its index
        word2idx = {word: i for i, word in enumerate(blist)}

        print(f'processing {path}...')
        text_path = os.path.join(path, 'text')
        text_datas = read_file(text_path, sp=' ')

        rareword_datas = []
        for data in tqdm(text_datas):
            uttid = data[0]

            # Filter words and remove duplicates
            results = get_uttblist(data[1:], blist)
            uttblist = [d[1] for d in results]

            # Generate the prompt with optional shuffling
            prompt = generate_prompt(uttblist, shuffle_blist=shuffle_blist)
            rareword_datas.append([uttid, prompt.upper()])

        # Save the prompts
        output_path_uttblist = os.path.join(path, 'prompt_only_entity')
        write_file(output_path_uttblist, rareword_datas)