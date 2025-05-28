from collections import defaultdict

def read_counts(file_path):
    """讀取 entity_count.txt 檔案"""
    counts = {}
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 2:  # 確保至少有實體名稱和計數
                entity = ' '.join(parts[:-1])  # 實體名稱可能包含空格
                count = int(parts[-1])
                counts[entity] = count
    return counts

def merge_counts(file1_path, file2_path, output_path):
    """合併兩個計數檔案"""
    # 讀取兩個檔案的計數
    counts1 = read_counts(file1_path)
    counts2 = read_counts(file2_path)
    
    # 合併計數
    merged_counts = defaultdict(int)
    for entity, count in counts1.items():
        merged_counts[entity] += count
    for entity, count in counts2.items():
        merged_counts[entity] += count
    
    # 按照計數排序並寫入檔案
    sorted_entities = sorted(merged_counts.items(), key=lambda x: (-x[1], x[0]))
    with open(output_path, 'w', encoding='utf-8') as f:
        for entity, count in sorted_entities:
            f.write(f"{entity} {count}\n")
    
    return len(sorted_entities)

def main():
    # 設定檔案路徑
    file1_path = '/share/nas169/yuchunliu/espnet_contextual/espnet/egs2/esun/asr1/local/contextual/rarewords/analyz2/counts.txt'  # 第一個實體計數檔案
    file2_path = '/share/nas169/yuchunliu/espnet_contextual/espnet/egs2/esun/asr1/local/contextual/rarewords/analyz/counts.txt'  # 第二個實體計數檔案
    output_path = 'merged_counts.txt'  # 合併後的輸出檔案
    
    try:
        print("=== 開始合併實體計數 ===")
        
        # 合併計數
        total_entities = merge_counts(file1_path, file2_path, output_path)
        
        print(f"\n合併完成！")
        print(f"合併後的實體總數：{total_entities}")
        print(f"結果已儲存至：{output_path}")
        
    except Exception as e:
        print(f"\n錯誤：{str(e)}")
        raise

if __name__ == "__main__":
    main()