import argparse
import os
import pandas as pd
from opencc import OpenCC
from fileio import read_file, write_file

def load_error_patterns(error_pattern_file):
    """讀取錯誤模式檔案並解析"""
    data = read_file(error_pattern_file, sp='\t')
    headers = data[0]
    patterns = data[1:]
    return patterns

def analyze_corrections(error_patterns):
    """分析並找出完全正確的 entity"""
    corrected_entities = []
    partially_corrected = []
    
    for pattern in error_patterns:
        entity = pattern[0]  # Entity 名稱
        error_count = int(pattern[2])  # 錯誤次數
        total_count = int(pattern[1])  # 總出現次數
        error_rate = float(pattern[3])  # 錯誤率
        error_patterns = pattern[4]  # 錯誤模式

        if error_count == 0:
            corrected_entities.append({
                'Entity': entity,
                'Total_Occurrences': total_count,
                'Error_Rate': error_rate,
                'Status': 'Fully Corrected'
            })
        elif error_count < total_count:
            partially_corrected.append({
                'Entity': entity,
                'Total_Occurrences': total_count,
                'Error_Count': error_count,
                'Error_Rate': error_rate,
                'Error_Patterns': error_patterns,
                'Status': 'Partially Corrected'
            })

    return corrected_entities, partially_corrected

def save_results(corrected_entities, partially_corrected, output_dir, model_name):
    """儲存分析結果"""
    # 轉換為 DataFrame
    df_corrected = pd.DataFrame(corrected_entities)
    df_partial = pd.DataFrame(partially_corrected)

    # 將簡體轉換為繁體
    cc = OpenCC('s2t')
    if not df_corrected.empty:
        df_corrected['Entity'] = df_corrected['Entity'].apply(lambda x: cc.convert(x))
    if not df_partial.empty:
        df_partial['Entity'] = df_partial['Entity'].apply(lambda x: cc.convert(x))

    # 儲存結果
    os.makedirs(output_dir, exist_ok=True)
    
    # 完全正確的 entities
    if not df_corrected.empty:
        df_corrected.to_csv(
            os.path.join(output_dir, f'{model_name}_fully_corrected.csv'),
            index=False, encoding='utf-8-sig'
        )
    
    # 部分正確的 entities
    if not df_partial.empty:
        df_partial.to_csv(
            os.path.join(output_dir, f'{model_name}_partially_corrected.csv'),
            index=False, encoding='utf-8-sig'
        )

    # 產生統計摘要
    with open(os.path.join(output_dir, f'{model_name}_summary.txt'), 'w', encoding='utf-8') as f:
        f.write(f"分析摘要\n")
        f.write(f"==========\n")
        f.write(f"完全正確的 Entity 數量: {len(corrected_entities)}\n")
        f.write(f"部分正確的 Entity 數量: {len(partially_corrected)}\n")
        
        if corrected_entities:
            avg_occurrences = sum(e['Total_Occurrences'] for e in corrected_entities) / len(corrected_entities)
            f.write(f"完全正確 Entity 的平均出現次數: {avg_occurrences:.2f}\n")

        if partially_corrected:
            avg_error_rate = sum(float(e['Error_Rate']) for e in partially_corrected) / len(partially_corrected)
            f.write(f"部分正確 Entity 的平均錯誤率: {avg_error_rate:.2f}%\n")

def main():
    parser = argparse.ArgumentParser(description='分析完全正確的 Entity')
    parser.add_argument('--error-pattern-file', required=True,
                        help='錯誤模式檔案路徑 (error_patterns.tsv)')
    parser.add_argument('--output-dir', required=True,
                        help='輸出目錄')
    parser.add_argument('--model-name', required=True,
                        help='模型名稱（用於輸出檔案名稱）')
    
    args = parser.parse_args()

    # 讀取錯誤模式
    error_patterns = load_error_patterns(args.error_pattern_file)
    
    # 分析修正情況
    corrected_entities, partially_corrected = analyze_corrections(error_patterns)
    
    # 儲存結果
    save_results(corrected_entities, partially_corrected, args.output_dir, args.model_name)

if __name__ == '__main__':
    main()

"""
使用範例：
python analyze_corrected_entities.py \
    --error-pattern-file /path/to/error_patterns.tsv \
    --output-dir /path/to/output \
    --model-name whisper_baseline
""" 