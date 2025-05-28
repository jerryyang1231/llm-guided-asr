import os
import matplotlib.pyplot as plt
from collections import Counter

# 設定檔案路徑
entity_list_path = "/share/nas169/yuchunliu/espnet_contextual/espnet/egs2/esun/asr2_aishell/local/contextual/rarewords/esun_earningcall.entity.txt"
reference_path = "/share/nas169/yuchunliu/espnet_contextual/espnet/egs2/esun/asr2_aishell/dump/raw/test/text"
baseline_path = "/share/nas169/yuchunliu/espnet_contextual/espnet/egs2/esun/asr2_aishell/exp/asr_whisper_medium_finetune_lr1e-5_adamw_wd1e-2_3epochs/decode_asr_whisper_noctc_greedy_asr_model_valid.acc.ave_3best/test/text_convert"
prompt_tuning_path = "/share/nas169/yuchunliu/espnet_contextual/espnet/egs2/esun/asr2_aishell/exp/asr_whisper_medium_prompt_finetune_entity_random10/decode_asr_whisper_noctc_greedy_asr_model_valid.acc.ave_3best_ESUN_fuzzy_tradprompt/test/new_text_convert"
output_dir = "output"
os.makedirs(output_dir, exist_ok=True)

def read_file(file_path):
    """讀取檔案，每行作為一個元素，回傳 list"""
    with open(file_path, "r", encoding="utf-8") as f:
        return [line.strip() for line in f.readlines()]

def analyze_corrections(reference, baseline, prompt_tuning, entity_list):
    """分析哪些 entity 被修正，哪些仍然錯誤"""
    corrected_entities = []
    still_wrong_entities = []

    for ref, base, prompt in zip(reference, baseline, prompt_tuning):
        ref_words = ref.split()
        base_words = base.split()
        prompt_words = prompt.split()

        for word in entity_list:
            if word in ref_words:
                base_wrong = word not in base_words
                prompt_wrong = word not in prompt_words

                if base_wrong and not prompt_wrong:
                    corrected_entities.append(word)  # 修正的 Entity
                elif base_wrong and prompt_wrong:
                    still_wrong_entities.append(word)  # 仍然錯誤的 Entity

    corrected_counter = Counter(corrected_entities)
    still_wrong_counter = Counter(still_wrong_entities)

    return corrected_counter, still_wrong_counter

def plot_results(corrected_counter, still_wrong_counter, output_dir, top_n=10):
    """繪製修正與錯誤的 Entity 長條圖"""
    corrected_labels, corrected_counts = zip(*corrected_counter.most_common(top_n)) if corrected_counter else ([], [])
    still_wrong_labels, still_wrong_counts = zip(*still_wrong_counter.most_common(top_n)) if still_wrong_counter else ([], [])

    fig, ax = plt.subplots(figsize=(10, 6))

    y_positions = range(len(corrected_labels) + len(still_wrong_labels))

    # 長條圖
    ax.barh(corrected_labels, corrected_counts, color='green', label="修正的 Entity")
    ax.barh(still_wrong_labels, still_wrong_counts, color='red', label="仍然錯誤的 Entity")

    ax.set_xlabel("出現次數")
    ax.set_ylabel("Entity")
    ax.set_title("Prompt Tuning 修正與錯誤的 Entity 統計")
    ax.legend()
    ax.invert_yaxis()

    plt.savefig(os.path.join(output_dir, "entity_comparison.png"))
    plt.show()

# 執行分析
entity_list = read_file(entity_list_path)
reference = read_file(reference_path)
baseline = read_file(baseline_path)
prompt_tuning = read_file(prompt_tuning_path)

corrected_counter, still_wrong_counter = analyze_corrections(reference, baseline, prompt_tuning, entity_list)
plot_results(corrected_counter, still_wrong_counter, output_dir)
