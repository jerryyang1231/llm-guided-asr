import os
import random
from tqdm import tqdm
from dataio import read_file
from dataio import write_file

TRAIN_BLIST_PATH = "/share/nas169/yuchunliu/espnet_contextual/espnet/egs2/aishell/asr1/id_ner_list/aishell_ner_train.txt"
DEV_BLIST_PATH   = "/share/nas169/yuchunliu/espnet_contextual/espnet/egs2/aishell/asr1/id_ner_list/aishell_ner_dev.txt"   
TEST_BLIST_PATH  = "/share/nas169/yuchunliu/espnet_contextual/espnet/egs2/aishell/asr1/id_ner_list/aishell_ner_test.txt"


PROMPT_TEMPLATE         = '''有出現: {}. 開始吧.'''
PROMPT_NON_ENT_TEMPLATE = '''開始吧.'''

def get_uttblist(words, blist):
    matched = []
    for word in words:
        for entity in blist:
            if entity in word:  # 使用 in 判斷是否匹配子字串
                matched.append([str(word2idx[entity]), entity])
    # 根據 word2idx 的索引排序，確保順序一致
    matched.sort(key=lambda x: int(x[0]))
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
    supplement_entity = True  # 若為 True，將 entity 補充到 10~20 個
    for folder in os.listdir(datas_path):
        path = os.path.join(datas_path, folder)
        if not os.path.isfile(os.path.join(path, 'wav.scp')):
            continue
        if 'test' not in path and 'dev' not in path:
            blist_path = TRAIN_BLIST_PATH
        elif 'test' not in path and 'train' not in path:
            blist_path = DEV_BLIST_PATH
        else:
            blist_path = TEST_BLIST_PATH
        blist = [b[0] for b in read_file(blist_path, sp=' ')]
        # 過濾掉只有一個字的 entity
        blist = [b for b in blist if len(b) > 1]
        word2idx = {word: i for i, word in enumerate(blist)}

        print(f'processing {path}...with {blist_path}...')
        text_path = os.path.join(path, 'text')
        text_datas = read_file(text_path, sp=' ')

        rareword_datas = []
        for data in tqdm(text_datas):
            uttid = data[0]
            results = get_uttblist(data[1:], blist)
            uttblist = [d[1] for d in results]

            # 若 supplement_entity 為 True 且 uttblist 有 entity，則補充到 10~20 個
            if supplement_entity and len(uttblist) > 0:
                target_num = random.randint(10, 20)
                if len(uttblist) < target_num:
                    # 從 blist 裡隨機補充不重複且不在 uttblist 的 entity
                    candidates = list(set(blist) - set(uttblist))
                    supplement = random.sample(candidates, min(target_num - len(uttblist), len(candidates)))
                    uttblist += supplement
                    random.shuffle(uttblist)

            # Generate the prompt with optional shuffling
            prompt = generate_prompt(uttblist, shuffle_blist=False)
            rareword_datas.append([uttid, prompt.upper()])

        output_path_uttblist = os.path.join(path, 'prompt_gtner_entity_word2_random')
        write_file(output_path_uttblist, rareword_datas)