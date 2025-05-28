import os
import re
from jiwer import cer
from tqdm import tqdm
from fileio import read_file, write_file
from text_aligner import align_to_index

rareword_list = '/share/nas169/yuchunliu/espnet_contextual/espnet/egs2/aishell/asr1/id_ner_list/aishell_ner_test.txt'
ref_path = '/share/nas169/yuchunliu/espnet_contextual/espnet/egs2/aishell/asr1/data/test/text'
hyp_path = "/share/nas169/yuchunliu/espnet_contextual/espnet/egs2/aishell/asr1/exp/asr_whisper_medium_prompt_finetune_description_topic_random10/decode_asr_whisper_noctc_greedy_asr_model_valid.acc.ave_3best_random10/test/text"

def check_passed(indexis, memory):
    for index in indexis:
        if index in memory:
            return True
    return False

def preprocess_sent(words):
    new_words = []
    for word in words:
        new_words.extend(list(word))
    return new_words

if __name__ == '__main__':
    rareword = [d[0] for d in read_file(rareword_list, sp=' ')]
    hyps = [[d[0], [i for i in d[1:] if i != '']] for d in read_file(hyp_path, sp=' ')]
    refs = [[d[0], d[1:]] for d in read_file(ref_path, sp=' ')]

    ref_rareword_sents = []
    hyp_rareword_sents = []
    ref_common_sents = []
    hyp_common_sents = []
    ref_sents = []
    hyp_sents = []

    for ref, hyp in zip(refs, hyps):
        blist = [word for word in rareword if word in " ".join(ref[1]).replace(' ', '')]
        ref_words = preprocess_sent(ref[1])
        hyp_words = preprocess_sent(hyp[1])
        ref_sents.append(' '.join(ref_words))
        hyp_sents.append(' '.join(hyp_words))
        
        ref_rareword_sent = []
        hyp_rareword_sent = []
        ref_common_sent = []
        hyp_common_sent = []
        passed_index = []

        chunks = align_to_index(ref_words, hyp_words)
        for chunk in chunks:
            wref, whyps, _, hindexis = chunk
            wref = wref.replace('-', '')
            whyps = ' '.join([whyp.replace('-', '') for whyp in whyps])

            if wref in blist:
                ref_rareword_sent.append(wref)
                hyp_rareword_sent.append(whyps)
            elif not check_passed(hindexis, passed_index):
                ref_common_sent.append(wref)
                hyp_common_sent.append(whyps)
                passed_index.extend(hindexis)

        if len(ref_rareword_sent) > 0:
            ref_rareword_sents.append(' '.join(ref_rareword_sent))
            hyp_rareword_sents.append(' '.join(hyp_rareword_sent))
        else:
            ref_rareword_sents.append('correct')
            hyp_rareword_sents.append('correct')

        if len(ref_common_sent) > 0:
            ref_common_sents.append(' '.join(ref_common_sent))
            hyp_common_sents.append(' '.join(hyp_common_sent))
        else:
            ref_common_sents.append('correct')
            hyp_common_sents.append('correct')

    all_cer = cer(ref_sents, hyp_sents)
    rareword_cer = cer(ref_rareword_sents, hyp_rareword_sents)
    commonword_cer = cer(ref_common_sents, hyp_common_sents)

    print(f'Overall CER   : {all_cer * 100:.2f}%')
    print(f'Rareword CER : {rareword_cer * 100:.2f}%')
    print(f'Commonword CER: {commonword_cer * 100:.2f}%')

    output_path = '/share/nas169/yuchunliu/espnet_contextual/espnet/egs2/aishell/asr1/exp/asr_whisper_medium_prompt_finetune_description_topic_random10/decode_asr_whisper_noctc_greedy_asr_model_valid.acc.ave_3best_random10/'
    write_file(os.path.join(output_path, 'ref_sents'), [[d] for d in ref_sents], sp='')
    write_file(os.path.join(output_path, 'hyp_sents'), [[d] for d in hyp_sents], sp='')
    write_file(os.path.join(output_path, 'ref_common_sents'), [[d] for d in ref_common_sents], sp='')
    write_file(os.path.join(output_path, 'hyp_common_sents'), [[d] for d in hyp_common_sents], sp='')
    write_file(os.path.join(output_path, 'ref_rareword_sents'), [[d] for d in ref_rareword_sents], sp='')
    write_file(os.path.join(output_path, 'hyp_rareword_sents'), [[d] for d in hyp_rareword_sents], sp='')

   
# ALL cer ---------------------------------------------
with open('/share/nas169/yuchunliu/espnet_contextual/espnet/egs2/aishell/asr1/exp/asr_whisper_medium_prompt_finetune_description_topic_random10/decode_asr_whisper_noctc_greedy_asr_model_valid.acc.ave_3best_random10/hyp_sents', 'r', encoding='utf-8') as file:
    lines = file.readlines()


processed_lines = []
for line in lines:
    # 使用 rpartition() 從最後一個 "開 始 吧 . " 切分
    before, separator, after = line.strip().rpartition('開 始 吧 . ')
    if separator:  # 如果找到 "開 始 吧 . "
        processed_lines.append(after)
    else:
        processed_lines.append(line.strip())


with open('/share/nas169/yuchunliu/espnet_contextual/espnet/egs2/aishell/asr1/exp/asr_whisper_medium_prompt_finetune_description_topic_random10/decode_asr_whisper_noctc_greedy_asr_model_valid.acc.ave_3best_random10/processed_hyp.txt', 'w', encoding='utf-8') as file:
    for line in processed_lines:
        file.write(line + '\n')


with open('/share/nas169/yuchunliu/espnet_contextual/espnet/egs2/aishell/asr1/exp/asr_whisper_medium_prompt_finetune_description_topic_random10/decode_asr_whisper_noctc_greedy_asr_model_valid.acc.ave_3best_random10/processed_hyp.txt', 'r', encoding='utf-8') as hyp_file:
    hyp_lines = [line.strip() for line in hyp_file.readlines()]


with open('/share/nas169/yuchunliu/espnet_contextual/espnet/egs2/aishell/asr1/exp/asr_whisper_medium_prompt_finetune_description_topic_random10/decode_asr_whisper_noctc_greedy_asr_model_valid.acc.ave_3best_random10/ref_sents', 'r', encoding='utf-8') as ref_file:
    ref_lines = [line.strip() for line in ref_file.readlines()]


def remove_parentheses(text):
    return re.sub(r'\(.*?\)', '', text).strip()


hyp_lines = [remove_parentheses(line) for line in hyp_lines]
ref_lines = [remove_parentheses(line) for line in ref_lines]


if len(hyp_lines) != len(ref_lines):
    print("Error: hyp.txt 和 ref.txt 行數不同，無法計算 WER")
else:
    
    total_wer = 0
    for hyp, ref in zip(hyp_lines, ref_lines):
        pair_wer = cer(ref, hyp)
        total_wer += pair_wer
        # print(f"目前累加 cer: {total_wer:.4f}")
        # input()  

    
    average_wer = total_wer / len(ref_lines)
    average_cer = average_wer *100
    print(f"平均 ALL cer: {average_cer:.2f}")

# Common cer ---------------------------------------------
with open('/share/nas169/yuchunliu/espnet_contextual/espnet/egs2/aishell/asr1/exp/asr_whisper_medium_prompt_finetune_description_topic_random10/decode_asr_whisper_noctc_greedy_asr_model_valid.acc.ave_3best_random10/hyp_common_sents', 'r', encoding='utf-8') as file:
    lines = file.readlines()


processed_lines = []
for line in lines:
    # 使用 rpartition() 從最後一個 "開 始 吧 . " 切分
    before, separator, after = line.strip().rpartition('開 始 吧 . ')
    if separator:  # 如果找到 "開 始 吧 . "
        processed_lines.append(after)
    else:
        processed_lines.append(line.strip())


with open('/share/nas169/yuchunliu/espnet_contextual/espnet/egs2/aishell/asr1/exp/asr_whisper_medium_prompt_finetune_description_topic_random10/decode_asr_whisper_noctc_greedy_asr_model_valid.acc.ave_3best_random10/processed_hyp_common.txt', 'w', encoding='utf-8') as file:
    for line in processed_lines:
        file.write(line + '\n')


with open('/share/nas169/yuchunliu/espnet_contextual/espnet/egs2/aishell/asr1/exp/asr_whisper_medium_prompt_finetune_description_topic_random10/decode_asr_whisper_noctc_greedy_asr_model_valid.acc.ave_3best_random10/processed_hyp_common.txt', 'r', encoding='utf-8') as hyp_file:
    hyp_lines = [line.strip() for line in hyp_file.readlines()]


with open('/share/nas169/yuchunliu/espnet_contextual/espnet/egs2/aishell/asr1/exp/asr_whisper_medium_prompt_finetune_description_topic_random10/decode_asr_whisper_noctc_greedy_asr_model_valid.acc.ave_3best_random10/ref_common_sents', 'r', encoding='utf-8') as ref_file:
    ref_lines = [line.strip() for line in ref_file.readlines()]


def remove_parentheses(text):
    return re.sub(r'\(.*?\)', '', text).strip()


hyp_lines = [remove_parentheses(line) for line in hyp_lines]
ref_lines = [remove_parentheses(line) for line in ref_lines]


if len(hyp_lines) != len(ref_lines):
    print("Error: hyp.txt 和 ref.txt 行數不同，無法計算 WER")
else:
    
    total_wer = 0
    for hyp, ref in zip(hyp_lines, ref_lines):
        pair_wer = cer(ref, hyp)
        total_wer += pair_wer
    
    average_wer = total_wer / len(ref_lines)
    average_cer = average_wer *100
    print(f"平均 Common CER: {average_cer:.2f}")

# Rareword CER ---------------------------------------------
with open('/share/nas169/yuchunliu/espnet_contextual/espnet/egs2/aishell/asr1/exp/asr_whisper_medium_prompt_finetune_description_topic_random10/decode_asr_whisper_noctc_greedy_asr_model_valid.acc.ave_3best_random10/hyp_rareword_sents', 'r', encoding='utf-8') as file:
    lines = file.readlines()


processed_lines = []
for line in lines:
    
    parts = line.strip().split('開 始 吧 . ', 1)
    if len(parts) > 1:
        processed_lines.append(parts[1])
    else:
        processed_lines.append(parts[0])


with open('/share/nas169/yuchunliu/espnet_contextual/espnet/egs2/aishell/asr1/exp/asr_whisper_medium_prompt_finetune_description_topic_random10/decode_asr_whisper_noctc_greedy_asr_model_valid.acc.ave_3best_random10/processed_hyp_rareword.txt', 'w', encoding='utf-8') as file:
    for line in processed_lines:
        file.write(line + '\n')

with open('/share/nas169/yuchunliu/espnet_contextual/espnet/egs2/aishell/asr1/exp/asr_whisper_medium_prompt_finetune_description_topic_random10/decode_asr_whisper_noctc_greedy_asr_model_valid.acc.ave_3best_random10/processed_hyp_rareword.txt', 'r', encoding='utf-8') as hyp_file:
    hyp_lines = [line.strip() for line in hyp_file.readlines()]


with open('/share/nas169/yuchunliu/espnet_contextual/espnet/egs2/aishell/asr1/exp/asr_whisper_medium_prompt_finetune_description_topic_random10/decode_asr_whisper_noctc_greedy_asr_model_valid.acc.ave_3best_random10/ref_rareword_sents', 'r', encoding='utf-8') as ref_file:
    ref_lines = [line.strip() for line in ref_file.readlines()]


def remove_parentheses(text):
    return re.sub(r'\(.*?\)', '', text).strip()


hyp_lines = [remove_parentheses(line) for line in hyp_lines]
ref_lines = [remove_parentheses(line) for line in ref_lines]


if len(hyp_lines) != len(ref_lines):
    print("Error: hyp.txt 和 ref.txt 行數不同，無法計算 WER")
else:
    
    total_wer = 0
    for hyp, ref in zip(hyp_lines, ref_lines):
        pair_wer = cer(ref, hyp)
        total_wer += pair_wer

    average_wer = total_wer / len(ref_lines)
    average_cer = average_wer *100
    print(f"平均 Rareword cer: {average_cer:.2f}")
