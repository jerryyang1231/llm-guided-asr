#!/usr/bin/env python3
# -*- coding: utf-8 -*-

fillers = {"啊", "啦", "齁", "吼", "呃"}

# ref
# input_path  = "/share/nas169/jerryyang/espnet/egs2/esun/work/dump/raw/test/text"
# output_path = "/share/nas169/jerryyang/espnet/egs2/esun/work/dump/raw/test/text_filtered"

# hyp
input_path  = "/share/nas169/jerryyang/espnet/egs2/esun/work/exp/asr_train_asr+llama3_conformer_contextual_biasing_v2_raw_zh_hugging_face_meta-llama-Llama-3.2-1B_sp/decode_bs1_ctc0.3_asr_model_valid.acc.ave/test/text"
output_path = "/share/nas169/jerryyang/espnet/egs2/esun/work/exp/asr_train_asr+llama3_conformer_contextual_biasing_v2_raw_zh_hugging_face_meta-llama-Llama-3.2-1B_sp/decode_bs1_ctc0.3_asr_model_valid.acc.ave/test/text_filtered"

with open(input_path,  encoding="utf-8") as fin, \
     open(output_path, "w", encoding="utf-8") as fout:
    for line in fin:
        line = line.strip()
        if not line:
            continue
        parts = line.split()
        uttid = parts[0]
        words = parts[1:]
        # 過濾掉 fillers
        words_filtered = [w for w in words if w not in fillers]
        # 重組並寫出
        fout.write(f"{uttid} {' '.join(words_filtered)}\n")
