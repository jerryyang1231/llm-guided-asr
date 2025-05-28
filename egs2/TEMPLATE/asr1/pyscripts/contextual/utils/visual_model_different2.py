import argparse
import os
import seaborn as sns
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import gridspec
from matplotlib import font_manager
from opencc import OpenCC

from fileio import read_file

# 初始化簡轉繁轉換器
cc = OpenCC('s2t')

font_path = '/share/nas169/yuchunliu/miniconda3/envs/espnet_context0113/lib/python3.10/site-packages/matplotlib/mpl-data/fonts/ttf/MicrosoftJhengHei.ttf'

if os.path.isfile(font_path):
    print(f"Font file found at {font_path}")
else:
    print(f"Font file not found at {font_path}")

font_manager.fontManager.addfont(font_path)
prop = font_manager.FontProperties(fname=font_path)
plt.rcParams['font.family'] = prop.get_name()
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['font.sans-serif'] = ['Microsoft JhengHei']
plt.rcParams['axes.unicode_minus'] = False

def convert_to_traditional(text):
    """將簡體中文轉換為繁體中文"""
    return cc.convert(text)

def is_chinese(text):
    """檢查文字是否包含中文字符"""
    for char in text:
        if '\u4e00' <= char <= '\u9fff':
            return True
    return False

def get_datas(paths):
    """Read and process data from the provided paths."""
    datas = {}
    for name in paths:
        data = read_file(paths[name], sp='\t')
        data = {d[0]: float(d[3]) for d in data[1:]}
        datas[name] = data
    return datas

def merge_datas(contexts, datas, ibrs, occ_test):
    """Merge Imbalance Rate and Error Rates into a structured table."""
    table = {ctx: [] for ctx in contexts}
    for ctx in table:
        table[ctx].append(ibrs[ctx])
        for name in datas:
            if ctx in datas[name]:
                table[ctx].append(datas[name][ctx])
            else:
                table[ctx].append(0.0)
    
    # Remove entities that do not appear in the test set
    _table = {}
    for ctx in table:
        if occ_test[ctx] != 0:
            _table[ctx] = table[ctx]
    table = _table

    title = ["Entity"] + ["ImbalanceRate"] + list(datas.keys())
    table = [[ctx] + table[ctx] for ctx in table]
    table.insert(0, title)
    return table

def create_heatmap(imbalance_rates, error_rates, title, output_path):
    """創建並保存熱力圖"""
    # 將索引轉換為繁體中文
    imbalance_rates.index = imbalance_rates.index.map(
        lambda x: convert_to_traditional(x) if is_chinese(x) else x
    )
    error_rates.index = error_rates.index.map(
        lambda x: convert_to_traditional(x) if is_chinese(x) else x
    )

    # Enhance Imbalance Rate heatmap by duplicating columns
    num_duplicates = 1
    imbalance_rates_expanded = pd.concat([imbalance_rates]*num_duplicates, axis=1)
    imbalance_rates_expanded.columns = [f'ImbalanceRate' for i in range(num_duplicates)]

    # Create figure
    fig = plt.figure(figsize=(10, 25))
    gs = gridspec.GridSpec(1, 2, width_ratios=[0.1, 4], wspace=0)

    # Imbalance Rate heatmap
    ax0 = plt.subplot(gs[0])
    sns.heatmap(imbalance_rates_expanded, ax=ax0, cmap="OrRd",
                cbar_kws={'label': 'Imbalance Rate (%)'},
                linewidths=.5, linecolor='gray', vmin=0, vmax=100,
                yticklabels=imbalance_rates_expanded.index, cbar=False)
    ax0.set_xlabel('', fontsize=16)
    ax0.set_title("", fontsize=18)
    ax0.set_xticklabels(ax0.get_xticklabels(), rotation=45, ha='right', fontsize=12)

    # Error Rates heatmap
    ax1 = plt.subplot(gs[1])
    sns.heatmap(error_rates, ax=ax1, annot=False, cmap="YlGnBu",
                linewidths=.5, linecolor='gray',
                cbar_kws={'label': 'Error Rate (%)'},
                vmin=error_rates.min().min(), vmax=error_rates.max().max(),
                yticklabels=False)
    ax1.set_xlabel("Models", fontsize=16)
    ax1.set_ylabel("", fontsize=16)
    ax1.set_title("Error Rate Comparison across Models by Entity", fontsize=18)
    ax1.set_xticklabels(ax1.get_xticklabels(), rotation=45, ha='right', fontsize=12)
    ax1.set_ylim(ax0.get_ylim())

    plt.suptitle(title, fontsize=20)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(output_path)
    plt.close()

if __name__ == '__main__':
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Process data and generate heatmaps.")
    parser.add_argument('--path', action='append', help='Data paths in the format label:path', required=True)
    parser.add_argument('--context-path', help='Path to context data file', required=True)
    parser.add_argument('--occ-train-path', help='Path to occurrence in train set data file', required=True)
    parser.add_argument('--occ-test-path', help='Path to occurrence in test set data file', required=True)
    parser.add_argument('--output', help='Output filename for the heatmap image', default='./sorted_combined_heatmaps.png')
    args = parser.parse_args()

    # Build the paths dictionary
    paths = {}
    for path_arg in args.path:
        try:
            label, path = path_arg.split(':', 1)
            paths[label.strip()] = path.strip()
        except ValueError:
            print(f"Invalid format for --path argument: {path_arg}")
            exit(1)

    # Retrieve and process data
    datas = get_datas(paths)
    context_path = args.context_path
    # 讀取時就轉換為繁體中文
    context_datas = [convert_to_traditional(d[0]) if is_chinese(d[0]) else d[0] 
                    for d in read_file(context_path, sp='\t')]

    occ_train_path = args.occ_train_path
    occ_test_path = args.occ_test_path

    occ_train_datas = [int(d[0]) for d in read_file(occ_train_path, sp='\t')]
    occ_test_datas = [int(d[0]) for d in read_file(occ_test_path, sp='\t')]

    # Calculate Imbalance Rate (ibrs)
    ibrs = np.log(np.array(occ_train_datas) * 100 + 1)
    ibrs = 100 - (ibrs / np.max(ibrs)) * 100
    ibrs = {ctx: ibr for ctx, ibr in zip(context_datas, ibrs)}

    # Occurrence in test set
    occ_test = {ctx: occ for ctx, occ in zip(context_datas, occ_test_datas)}

    # Merge and convert into DataFrame
    merged_data = merge_datas(context_datas, datas, ibrs, occ_test)
    error_comparison_df = pd.DataFrame(merged_data[1:], columns=merged_data[0])
    heatmap_data = error_comparison_df.set_index('Entity').fillna(0)

    # Sort the data by the first model in descending order
    first_model_name = list(paths.keys())[0]
    sorted_heatmap_data = heatmap_data.sort_values(by=first_model_name, ascending=False)

    # Extract Imbalance Rate and Error Rates
    imbalance_rates = sorted_heatmap_data[['ImbalanceRate']][:100]
    error_rates = sorted_heatmap_data.drop(columns=['ImbalanceRate'])[:100]

    # 分離中文和英文 entity
    chinese_entities = sorted_heatmap_data.index.map(is_chinese)
    chinese_data = sorted_heatmap_data[chinese_entities][:100]
    english_data = sorted_heatmap_data[~chinese_entities][:100]

    # 生成三個熱力圖
    # 1. 所有 entity
    output_base = os.path.splitext(args.output)[0]
    create_heatmap(
        sorted_heatmap_data[['ImbalanceRate']][:100],
        sorted_heatmap_data.drop(columns=['ImbalanceRate'])[:100],
        "Imbalance Rate and Error Rate by Entity (All)",
        f"{output_base}_all.png"
    )

    # 2. 只有中文 entity
    create_heatmap(
        chinese_data[['ImbalanceRate']],
        chinese_data.drop(columns=['ImbalanceRate']),
        "Imbalance Rate and Error Rate by Entity (Chinese)",
        f"{output_base}_chinese.png"
    )

    # 3. 只有英文 entity
    create_heatmap(
        english_data[['ImbalanceRate']],
        english_data.drop(columns=['ImbalanceRate']),
        "Imbalance Rate and Error Rate by Entity (English)",
        f"{output_base}_english.png"
    )

"""
python -m visual_model_different2 
  --path "Whisper (Baseline):/share/nas169/yuchunliu/espnet_contextual/espnet/egs2/esun/asr2_aishell/exp/asr_whisper_medium_finetune_lr1e-5_adamw_wd1e-2_3epochs/decode_asr_whisper_noctc_greedy_asr_model_valid.acc.ave_3best/test/analysis/error_patterns.tsv" 
  --path "Whisper (Prompt Tuning):/share/nas169/yuchunliu/espnet_contextual/espnet/egs2/esun/asr2_aishell/exp/asr_whisper_medium_prompt_finetune_entity_random10/decode_asr_whisper_noctc_greedy_asr_model_valid.acc.ave_3best_ESUN_fuzzy_tradprompt/test/analysis/error_patterns.tsv" 
  --context-path /share/nas169/yuchunliu/espnet_contextual/espnet/egs2/esun/asr1_contextual/local/contextual/rarewords/esun_earningcall.entity.txt 
  --occ-train-path /share/nas169/yuchunliu/espnet_contextual/espnet/egs2/esun/asr1_contextual/local/contextual/rarewords/esun_earningcall.entity_occurrence_train.txt 
  --occ-test-path /share/nas169/yuchunliu/espnet_contextual/espnet/egs2/esun/asr1_contextual/local/contextual/rarewords/esun_earningcall.entity_occurrence_test.txt 
  --output /share/nas169/yuchunliu/espnet_contextual/espnet/egs2/esun/asr2_aishell/exp/sorted_combined_heatmaps.png
"""