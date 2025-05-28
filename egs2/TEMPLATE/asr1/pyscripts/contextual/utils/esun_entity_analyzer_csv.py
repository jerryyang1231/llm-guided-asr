import re
import csv
from collections import defaultdict, Counter

# 設定檔案路徑
TEXT_FILE = '/share/nas169/yuchunliu/espnet_contextual/espnet/egs2/esun/asr1/dump/raw/train_sp/text'                 # 輸入的文本檔案
ENTITY_FILE = '/share/nas169/yuchunliu/espnet_contextual/espnet/egs2/esun/asr1/local/contextual/rarewords/esun_earningcall.entity.txt'         # 輸入的 entity list 檔案
COUNT_OUTPUT = '/share/nas169/yuchunliu/espnet_contextual/espnet/egs2/esun/asr1/local/contextual/rarewords/analyz/counts.txt'             # 輸出的計數結果檔案
LOCATION_OUTPUT = '/share/nas169/yuchunliu/espnet_contextual/espnet/egs2/esun/asr1/local/contextual/rarewords/analyz/locations.txt'        # 輸出的位置結果檔案
PROCESSED_TEXT_OUTPUT = '/share/nas169/yuchunliu/espnet_contextual/espnet/egs2/esun/asr1/local/contextual/rarewords/analyz/processed_text.txt'    # 輸出處理過的文本檔案
PROCESSED_ENTITY_OUTPUT = '/share/nas169/yuchunliu/espnet_contextual/espnet/egs2/esun/asr1/local/contextual/rarewords/analyz/processed_entities.txt'  # 輸出處理過的實體列表
DEBUG_OUTPUT = '/share/nas169/yuchunliu/espnet_contextual/espnet/egs2/esun/asr1/local/contextual/rarewords/analyz/debug_info.txt'         # 輸出除錯資訊
# 新增 KWS 相關路徑
WAV_SCP = '/share/nas169/yuchunliu/espnet_contextual/espnet/egs2/esun/asr1/data/train_sp/wav.scp'
KWS_CSV_OUTPUT = '/share/nas169/yuchunliu/espnet_contextual/espnet/egs2/esun/asr1/local/contextual/rarewords/analyz/kws_data.csv'

def read_wav_scp(wav_scp_file):
    """讀取 wav.scp 檔案"""
    wav_paths = {}
    with open(wav_scp_file, 'r', encoding='utf-8') as f:
        for line in f:
            parts = line.strip().split(maxsplit=1)
            if len(parts) == 2:
                wav_paths[parts[0]] = parts[1]
    return wav_paths

def process_text_for_matching(text):
    """處理文本以準備進行匹配"""
    return text.replace(' ', '').lower()

def read_text_from_file(text_file):
    """讀取文本檔案"""
    total_lines = 0
    valid_lines = 0
    invalid_lines = []
    content = []
    
    with open(text_file, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f, 1):
            total_lines += 1
            line = line.strip()
            if line:
                # 檢查是否符合 "{ID} {TEXT}" 格式
                parts = line.split(maxsplit=1)  # 只分割一次，分成ID和剩餘文本
                if len(parts) == 2:  # 如果能分成兩部分
                    valid_lines += 1
                    content.append(line)
                else:
                    invalid_lines.append((line_num, line))
            
            if total_lines % 10000 == 0:
                print(f"已處理 {total_lines} 行...")
    
    print(f"\n文件統計：")
    print(f"總行數：{total_lines}")
    print(f"有效行數：{valid_lines}")
    print(f"無效行數：{total_lines - valid_lines}")
    
    if invalid_lines:
        print("\n前5個無效行的範例：")
        for line_num, line in invalid_lines[:5]:
            print(f"行號 {line_num}: {line[:100]}")
    
    return '\n'.join(content)

def read_and_process_entities(entity_file):
    """讀取和處理實體列表，處理重複和大小寫問題"""
    entities = []
    processed_to_original = {}  # 處理後形式 -> 原始形式的映射
    original_to_processed = {}  # 原始形式 -> 處理後形式的映射
    duplicates = []  # 記錄重複項
    
    with open(entity_file, 'r', encoding='utf-8') as f:
        for line in f:
            original_entity = line.strip()
            if not original_entity:
                continue
                
            processed_entity = process_text_for_matching(original_entity)
            
            # 檢查是否已存在相同的處理後形式
            if processed_entity in processed_to_original:
                duplicates.append((original_entity, processed_to_original[processed_entity]))
                continue
                
            entities.append(original_entity)
            processed_to_original[processed_entity] = original_entity
            original_to_processed[original_entity] = processed_entity
    
    return entities, processed_to_original, original_to_processed, duplicates

def analyze_entities(text_content, entities, processed_to_original, wav_paths):
    """分析文本中的實體出現次數和位置，同時生成 KWS 資料"""
    entity_counts = Counter({entity: 0 for entity in entities})
    entity_locations = defaultdict(set)
    processed_lines = []
    debug_info = []
    kws_data = []  # 用於存儲 KWS 訓練資料
    
    for line in text_content.split('\n'):
        if not line.strip():
            continue
        
        # 分割 ID 和文本
        parts = line.split(maxsplit=1)
        if len(parts) != 2:
            continue
        
        line_id, line_text = parts
        processed_text = process_text_for_matching(line_text)
        processed_lines.append((line_id, line_text, processed_text))
        
        # 檢查音檔路徑是否存在
        audio_path = wav_paths.get(line_id)
        if not audio_path:
            continue
        
        # 初始化標籤
        labels = [0] * len(entities)
        has_entity = False
        matches_in_line = []
        
        # 檢查每個處理後的實體
        for i, entity in enumerate(entities):
            processed_entity = process_text_for_matching(entity)
            if processed_entity in processed_text:
                entity_counts[entity] += 1
                entity_locations[entity].add(line_id)
                matches_in_line.append(entity)
                labels[i] = 1
                has_entity = True
        
        if matches_in_line:
            debug_info.append(f"{line_id}：找到實體 {', '.join(matches_in_line)}")
        
        # 如果有實體出現，加入 KWS 資料
        if has_entity:
            kws_data.append([audio_path] + labels)
    
    return dict(entity_counts), dict(entity_locations), processed_lines, debug_info, kws_data

def save_results(counts, locations, entities, processed_to_original, 
                counts_file, locations_file, processed_text_file, 
                processed_entity_file, debug_file, duplicates, debug_info, 
                processed_lines, kws_data, kws_csv_file):
    """儲存所有結果，計數結果按照出現次數排序"""
    # 儲存計數結果（按照計數從高到低排序）
    with open(counts_file, 'w', encoding='utf-8') as f:
        # 將所有實體和計數轉換為列表並排序
        sorted_counts = sorted([(entity, counts.get(entity, 0)) for entity in entities],
                             key=lambda x: (-x[1], x[0]))  # 先按計數降序，再按實體名稱升序
        
        # 寫入排序後的結果
        for entity, count in sorted_counts:
            f.write(f"{entity} {count}\n")
    
    # [其餘部分保持不變]
    # 儲存位置結果
    with open(locations_file, 'w', encoding='utf-8') as f:
        for entity in entities:
            ids = locations.get(entity, set())
            if ids:
                f.write(f"{entity}\t{','.join(sorted(ids))}\n")
    
    # 儲存處理過的文本
    with open(processed_text_file, 'w', encoding='utf-8') as f:
        for line_id, original_text, processed_text in processed_lines:
            f.write(f"{line_id}\tOriginal: {original_text}\tProcessed: {processed_text}\n")
    
    # 儲存處理過的實體列表和重複項
    with open(processed_entity_file, 'w', encoding='utf-8') as f:
        f.write("=== 處理後的實體列表 ===\n")
        for processed, original in processed_to_original.items():
            f.write(f"Original: {original}\tProcessed: {processed}\n")
        
        if duplicates:
            f.write("\n=== 發現的重複實體 ===\n")
            for duplicate, original in duplicates:
                f.write(f"重複項: {duplicate}\t對應到: {original}\n")
    
    # 儲存除錯資訊
    with open(debug_file, 'w', encoding='utf-8') as f:
        f.write("=== 匹配過程除錯資訊 ===\n")
        for info in debug_info:
            f.write(f"{info}\n")

    with open(kws_csv_file, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            # 寫入標頭
            header = ['audio_path'] + [e for e in entities]  # 使用原始實體列表保持順序
            writer.writerow(header)
            # 寫入資料
            writer.writerows(kws_data)

def print_summary(entities, counts, duplicates):
    """印出分析摘要"""
    print("\n=== 分析摘要 ===")
    print(f"實體總數: {len(entities)}")
    matched_entities = sum(1 for count in counts.values() if count > 0)
    print(f"匹配到的實體數: {matched_entities}")
    print(f"未匹配實體數: {len(entities) - matched_entities}")
    print(f"重複實體數: {len(duplicates)}")
    total_occurrences = sum(counts.values())
    print(f"實體出現總次數: {total_occurrences}")
    
    if duplicates:
        print("\n發現的重複實體:")
        for duplicate, original in duplicates:
            print(f"  - '{duplicate}' 與 '{original}' 在處理後形式相同")

def main():
    try:
        print("=== 實體分析程式開始執行 ===")
        print(f"\n使用以下檔案：")
        print(f"文本檔案：{TEXT_FILE}")
        print(f"Entity 列表：{ENTITY_FILE}")
        print(f"音檔列表：{WAV_SCP}")
        
        # 檢查文件格式
        print("\n檢查文本檔案格式...")
        with open(TEXT_FILE, 'r', encoding='utf-8') as f:
            print("前5行範例：")
            for i, line in enumerate(f):
                if i < 5:
                    print(f"行 {i+1}: {line.strip()}")
                else:
                    break
        # 讀取 wav.scp
        print("\n讀取 wav.scp 檔案...")
        wav_paths = read_wav_scp(WAV_SCP)
        print(f"載入音檔路徑數量：{len(wav_paths)}")
        # 讀取並處理實體
        print("\n讀取實體列表...")
        entities, processed_to_original, original_to_processed, duplicates = read_and_process_entities(ENTITY_FILE)
        print(f"載入實體數量：{len(entities)}")
        
        # 讀取文本
        print("\n讀取文本檔案...")
        text_content = read_text_from_file(TEXT_FILE)
        
        # 分析文本
        print("\n分析文本中...")
        counts, locations, processed_lines, debug_info, kws_data = analyze_entities(
            text_content, entities, processed_to_original, wav_paths)
        
        # 儲存結果
        print("\n儲存結果中...")
        save_results(
            counts, locations, entities, processed_to_original,
            COUNT_OUTPUT, LOCATION_OUTPUT, PROCESSED_TEXT_OUTPUT,
            PROCESSED_ENTITY_OUTPUT, DEBUG_OUTPUT,
            duplicates, debug_info, processed_lines,
            kws_data, KWS_CSV_OUTPUT
        )
        
        # 印出摘要
        print_summary(entities, counts, duplicates)
        
        print(f"\n=== 分析完成！===")
        print(f"已產生以下檔案：")
        print(f"- 計數結果：{COUNT_OUTPUT}")
        print(f"- 位置資訊：{LOCATION_OUTPUT}")
        print(f"- 處理後文本：{PROCESSED_TEXT_OUTPUT}")
        print(f"- 處理後實體：{PROCESSED_ENTITY_OUTPUT}")
        print(f"- 除錯資訊：{DEBUG_OUTPUT}")        
        print(f"- KWS 訓練資料：{KWS_CSV_OUTPUT}")
        
    except Exception as e:
        print(f"\n錯誤：{str(e)}")
        raise

if __name__ == "__main__":
    main()