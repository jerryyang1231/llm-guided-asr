import os
import numpy as np

from tqdm import tqdm
from dataio import read_file
from dataio import read_json
from dataio import write_file
import matplotlib.pyplot as plt
from collections import Counter
from matplotlib import font_manager, rc

# 設置字型
font_path = '/share/nas169/yuchunliu/miniconda3/envs/espnet_context0113/lib/python3.10/site-packages/matplotlib/mpl-data/fonts/ttf/MicrosoftJhengHei.ttf'

if os.path.isfile(font_path):
    print(f"Font file found at {font_path}")
    font_manager.fontManager.addfont(font_path)
    prop = font_manager.FontProperties(fname=font_path)
    plt.rcParams['font.family'] = prop.get_name()
    plt.rcParams['axes.unicode_minus'] = False
else:
    print(f"Font file not found at {font_path}")

# 計算實體出現次數
def occurrence(texts, bwords):
    bword_occurrence = Counter({word: 0 for word in bwords})
    oov = 0
    for _, words in texts:
        for word in words:
            if word not in bword_occurrence:
                oov += 1
                continue
            bword_occurrence[word] += 1
    return bword_occurrence, oov

# 生成頻率圖
def plot_frequency(bword_occurrence, output_path, title, max_y=None, top_n=None, reverse=False, x_spacing=1.2):
    # 取實體名稱與頻率
    entities, counts = zip(*bword_occurrence.items())
    
    # 排序
    sorted_indices = sorted(range(len(counts)), key=lambda k: counts[k], reverse=not reverse)
    sorted_entities = [entities[i] for i in sorted_indices]
    sorted_counts = [counts[i] for i in sorted_indices]

    # 限制顯示的實體數量
    if top_n:
        sorted_entities = sorted_entities[:top_n]
        sorted_counts = sorted_counts[:top_n]

    # 畫圖
    plt.figure(figsize=(len(sorted_entities) * x_spacing / 10, 6))
    bar_positions = range(len(sorted_counts))
    plt.bar(bar_positions, sorted_counts, tick_label=sorted_entities, width=0.6)
    plt.xticks(bar_positions, sorted_entities, rotation=90, fontsize=8)
    plt.xlabel("Entities", fontsize=12)
    plt.ylabel("Occurrences", fontsize=12)
    plt.title(title, fontsize=14)
    
    # 設定 Y 軸範圍
    if max_y:
        plt.ylim(0, max_y)
    
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
    print(f"頻率圖已保存至: {output_path}")


if __name__ == '__main__':
    # 路徑
    train_text_path = '/share/nas169/yuchunliu/espnet_contextual/espnet/egs2/aishell/asr1/dump/raw/zh_train_sp/text'
    dev_text_path   = '/share/nas169/yuchunliu/espnet_contextual/espnet/egs2/aishell/asr1/dump/raw/zh_dev/text'
    test_text_path  = '/share/nas169/yuchunliu/espnet_contextual/espnet/egs2/aishell/asr1/dump/raw/zh_test/text'
    train_blist_path = "/share/nas169/yuchunliu/espnet_contextual/espnet/egs2/aishell/asr2/id_ner_list/aishell_ner_train.txt"
    test_blist_path  = "/share/nas169/yuchunliu/espnet_contextual/espnet/egs2/aishell/asr2/id_ner_list/aishell_ner_test.txt"
    dump_path        = '/share/nas169/yuchunliu/espnet_contextual/espnet/egs2/aishell/asr1/local/contextual/rarewords'

    # 讀取數據
    train_texts = [[d[0], d[1:]] for d in read_file(train_text_path, sp=' ')]
    dev_texts = [[d[0], d[1:]] for d in read_file(dev_text_path, sp=' ')]
    test_texts = [[d[0], d[1:]] for d in read_file(test_text_path, sp=' ')]
    
    train_blist = [b[0] for b in read_file(train_blist_path, sp=' ')]
    test_blist = [b[0] for b in read_file(test_blist_path, sp=' ')]

    # 合併 train 和 dev 數據
    train_dev_texts = train_texts + dev_texts

    # 計算出現次數
    train_occurrence, train_oov = occurrence(train_dev_texts, train_blist)
    test_occurrence, test_oov = occurrence(test_texts, test_blist)

    # 寫出計算結果
    train_output_path = os.path.join(dump_path, 'train_dev_occurrence.txt')
    test_output_path = os.path.join(dump_path, 'test_occurrence.txt')

    # 格式化數據
    train_data = [f"{word} {count}" for word, count in train_occurrence.items()]
    test_data = [f"{word} {count}" for word, count in test_occurrence.items()]

    write_file(train_output_path, train_data)
    write_file(test_output_path, test_data)
    print(f"計算結果已保存：\n - {train_output_path}\n - {test_output_path}")

    # 畫圖
    train_plot_path = os.path.join(dump_path, 'train_dev_frequency.png')
    test_plot_path = os.path.join(dump_path, 'test_frequency.png')

    # Train 頻率倒數圖
    train_reverse_plot_path = os.path.join(dump_path, 'train_dev_frequency_reverse.png')
    plot_frequency(train_occurrence, train_reverse_plot_path, "Train Entity Frequency (Bottom 100)", top_n=100, reverse=True, x_spacing=1.5)

    # Test 頻率倒數圖
    test_reverse_plot_path = os.path.join(dump_path, 'test_frequency_reverse.png')
    plot_frequency(test_occurrence, test_reverse_plot_path, "Test Entity Frequency (Bottom 100)", top_n=100, reverse=True, x_spacing=1.5)

