import os
import random
from tqdm import tqdm
from dataio import read_file
from dataio import write_file

TRAIN_DEV_BLIST_PATH = "/share/nas169/yuchunliu/espnet_contextual/espnet/egs2/esun/asr1/local/contextual/rarewords/esun_earningcall.entity.txt"
TEST_BLIST_PATH      = "/share/nas169/yuchunliu/espnet_contextual/espnet/egs2/esun/asr1/local/contextual/rarewords/esun_earningcall.entity.txt"

# PROMPT_TEMPLATE         = '''玉山銀行法人說明會是一個定期舉辦的活動，向投資者和法人機構介紹公司財務狀況、業務策略及未來展望。該活動旨在增進透明度，讓法人了解公司的經營績效與市場動態。有出現: {}. 開始吧.'''
# PROMPT_NON_ENT_TEMPLATE = '''開始吧.'''
PROMPT_TEMPLATE         = '''主題為: {}. 開始吧.'''
PROMPT_NON_ENT_TEMPLATE = '''開始吧.'''

def get_uttblist(words, blist):
    return [[str(word2idx[word]), word] for word in words if word in blist]

def supplement_rarewords(uttblist, blist, num_words=10):
    # 如果稀有詞少於 10 個，隨機從 blist 中補充
    if len(uttblist) < num_words and len(uttblist)>0:
        remaining_words = list(set(blist) - set(uttblist))  # 排除已存在的詞
        additional_words = random.sample(remaining_words, num_words - len(uttblist))
        uttblist += additional_words
    return uttblist

if __name__ == '__main__':
    datas_path = '/share/nas169/yuchunliu/espnet_contextual/espnet/egs2/esun/asr1/dump/raw'
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
        text_path  = os.path.join(path, 'text')
        text_datas = read_file(text_path, sp=' ')
        
        rareword_datas = []
        rareword_idxs  = []
        for data in tqdm(text_datas):
            uttid    = data[0]
            results  = get_uttblist(data[1:], blist)
            uttblist = [d[1] for d in results]
            
            # 補充稀有詞到 10 個
            uttblist = supplement_rarewords(uttblist, blist, num_words=10)

            if len(uttblist) > 0:
                prompt = PROMPT_TEMPLATE.format(", ".join(uttblist))
            else:
                prompt = PROMPT_NON_ENT_TEMPLATE
            rareword_datas.append([uttid, prompt.upper()])

        output_path_uttblist = os.path.join(path, 'prompt_only_entity_random10')
        write_file(output_path_uttblist, rareword_datas)
