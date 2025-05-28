import os
import re
from collections import Counter
import matplotlib.pyplot as plt
from matplotlib import font_manager
from dataio import read_file, write_file

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
def compute_occurrence(texts, entities):
    """
    計算實體出現次數，支持中英混雜、空格處理與小寫匹配
    :param texts: [[ID, [分詞列表]], ...]
    :param entities: 實體列表
    :return: Counter 實例，包含每個實體的出現次數，與未匹配的文本列表
    """
    # 移除實體中的空格並轉為小寫
    entities = [entity.replace(' ', '').lower() for entity in entities]
    entity_counts = Counter({entity: 0 for entity in entities})
    unmatched_texts = []
    
    for text_id, words in texts:
        # 將文本拼接並移除空格後轉為小寫
        original_text = ''.join(words).lower()
        matched = False

        for entity in entities:
            # 使用正則匹配完整實體
            if re.search(rf'{re.escape(entity)}', original_text):
                entity_counts[entity] += 1  # 累計匹配次數
                matched = True
        
        if not matched:
            unmatched_texts.append((text_id, original_text))  # 記錄未匹配文本

    return entity_counts, unmatched_texts

# 繪製頻率圖
def plot_frequency(entity_counts, entity_mapping, output_path, title, top_n=None, reverse=False, x_spacing=1.2):
    entities, counts = zip(*entity_counts.items())
    
    # 排序
    sorted_indices = sorted(range(len(counts)), key=lambda k: counts[k], reverse=not reverse)
    sorted_entities = [entities[i] for i in sorted_indices]
    sorted_counts = [counts[i] for i in sorted_indices]

    # 還原實體名稱
    sorted_entities = [entity_mapping[entity] for entity in sorted_entities]

    # 限制顯示數量
    if top_n:
        sorted_entities = sorted_entities[:top_n]
        sorted_counts = sorted_counts[:top_n]

    # 繪圖
    plt.figure(figsize=(len(sorted_entities) * x_spacing / 10, 6))
    bar_positions = range(len(sorted_counts))
    plt.bar(bar_positions, sorted_counts, tick_label=sorted_entities, width=0.6)
    plt.xticks(bar_positions, sorted_entities, rotation=90, fontsize=8)
    plt.xlabel("Entities", fontsize=12)
    plt.ylabel("Occurrences", fontsize=12)
    plt.title(title, fontsize=14)
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
    print(f"頻率圖已保存至: {output_path}")


if __name__ == '__main__':
    # 路徑
    train_text_path = '/share/nas169/yuchunliu/espnet_contextual/espnet/egs2/esun/asr1/dump/raw/train_sp/text'
    test_text_path = '/share/nas169/yuchunliu/espnet_contextual/espnet/egs2/esun/asr1/dump/raw/test/text'
    entity_list_path = "/share/nas169/yuchunliu/espnet_contextual/espnet/egs2/esun/asr1/local/contextual/rarewords/esun_earningcall.entity.txt"
    dump_path = '/share/nas169/yuchunliu/espnet_contextual/espnet/egs2/esun/asr1/local/contextual/rarewords/'

    # 讀取數據
    train_texts = [[d[0], d[1:]] for d in read_file(train_text_path, sp=' ')]
    test_texts = [[d[0], d[1:]] for d in read_file(test_text_path, sp=' ')]
    # 讀取實體列表並建立映射
    raw_entities = read_file(entity_list_path, sp='\n')
    if isinstance(raw_entities[0], list):  # 檢查是否為嵌套列表
        raw_entities = [line[0] for line in raw_entities]

    # 建立實體映射字典
    entity_mapping = {entity.replace(' ', '').lower(): entity for entity in raw_entities}
    entities = list(entity_mapping.keys())

    print("Entities loaded:", list(entity_mapping.values())[:5])  # 確認載入的實體

    # 計算出現次數與未匹配文本
    train_counts, train_unmatched = compute_occurrence(train_texts, entities)
    test_counts, test_unmatched = compute_occurrence(test_texts, entities)

    # 保存未分詞的文本
    train_unmatched_path = os.path.join(dump_path, 'train_unmatched_texts.txt')
    test_unmatched_path = os.path.join(dump_path, 'test_unmatched_texts.txt')
    write_file(train_unmatched_path, [f"{tid} {text}" for tid, text in train_unmatched])
    write_file(test_unmatched_path, [f"{tid} {text}" for tid, text in test_unmatched])

    # 保存出現次數清單
    train_count_path = os.path.join(dump_path, 'train_entity_counts.txt')
    test_count_path = os.path.join(dump_path, 'test_entity_counts.txt')
    write_file(train_count_path, [f"{entity_mapping[entity]} {count}" for entity, count in train_counts.items()])
    write_file(test_count_path, [f"{entity_mapping[entity]} {count}" for entity, count in test_counts.items()])

    # 保存出現次數清單（排序後）
    train_sorted_count_path = os.path.join(dump_path, 'train_entity_counts_sort.txt')
    test_sorted_count_path = os.path.join(dump_path, 'test_entity_counts_sort.txt')

    # 生成排序後的清單
    train_sorted_counts = sorted(train_counts.items(), key=lambda x: x[1], reverse=True)
    test_sorted_counts = sorted(test_counts.items(), key=lambda x: x[1], reverse=True)

    # 將排序後的清單寫入文件
    write_file(train_sorted_count_path, [f"{entity_mapping[entity]} {count}" for entity, count in train_sorted_counts])
    write_file(test_sorted_count_path, [f"{entity_mapping[entity]} {count}" for entity, count in test_sorted_counts])

    print(f"排序後的清單已保存：\n - {train_sorted_count_path}\n - {test_sorted_count_path}")

    # 繪製頻率圖
    train_plot_path = os.path.join(dump_path, 'train_entity_frequency.png')
    test_plot_path = os.path.join(dump_path, 'test_entity_frequency.png')
    plot_frequency(train_counts, entity_mapping, train_plot_path, "Train Entity Frequency", top_n=100, reverse=False, x_spacing=1.5)
    plot_frequency(test_counts, entity_mapping, test_plot_path, "Test Entity Frequency", top_n=100, reverse=False, x_spacing=1.5)
