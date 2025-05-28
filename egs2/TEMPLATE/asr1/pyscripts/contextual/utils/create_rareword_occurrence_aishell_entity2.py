import os
import numpy as np
from tqdm import tqdm
from dataio import read_file, write_file

# 定義檔案路徑
TRAIN_DEV_BLIST_PATH = "/share/nas169/yuchunliu/espnet_contextual/espnet/egs2/aishell/asr2/id_ner_list/aishell_ner_full.txt"
TEST_BLIST_PATH = "/share/nas169/yuchunliu/espnet_contextual/espnet/egs2/aishell/asr2/id_ner_list/aishell_ner_test.txt"

def occurrence(texts, bwords):
    """
    計算指定字詞列表在文本中的出現次數
    Args:
        texts: [[uid, [word1, word2, ...]], ...]
        bwords: [bword1, bword2, ...]
    Returns:
        List[int]: 每個字詞的出現次數 + 未知字詞數量
    """
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
    # 設定輸入與輸出路徑
    train_text_path = '/share/nas169/yuchunliu/espnet_contextual/espnet/egs2/aishell/asr1/dump/raw/zh_train_sp/text'
    dev_text_path = '/share/nas169/yuchunliu/espnet_contextual/espnet/egs2/aishell/asr1/dump/raw/zh_dev/text'
    test_text_path = '/share/nas169/yuchunliu/espnet_contextual/espnet/egs2/aishell/asr1/dump/raw/zh_test/text'
    dump_path = '/share/nas169/yuchunliu/espnet_contextual/espnet/egs2/aishell/asr1/local/contextual/rarewords'

    # 載入文本數據
    train_texts = [[d[0], d[1:]] for d in read_file(train_text_path, sp=' ')]
    dev_texts = [[d[0], d[1:]] for d in read_file(dev_text_path, sp=' ')]
    test_texts = [[d[0], d[1:]] for d in read_file(test_text_path, sp=' ')]

    # 載入實體黑名單
    blist = [b[0] for b in read_file(TRAIN_DEV_BLIST_PATH, sp=' ')]

    # 計算訓練與開發集中的實體出現次數
    train_dev_texts = train_texts + dev_texts
    train_dev_counts = list(map(lambda x: [str(x)], occurrence(train_dev_texts, blist)))

    # 計算測試集中的實體出現次數
    test_counts = list(map(lambda x: [str(x)], occurrence(test_texts, blist)))

    # 儲存結果
    train_dev_output_path = os.path.join(dump_path, 'train_dev_occurrence.txt')
    write_file(train_dev_output_path, train_dev_counts)
    print(f"Train/Dev occurrence file saved to: {train_dev_output_path}")

    test_output_path = os.path.join(dump_path, 'test_occurrence.txt')
    write_file(test_output_path, test_counts)
    print(f"Test occurrence file saved to: {test_output_path}")

    print("Done.")
