#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import re

def load_entities(entity_path):
    """
    讀取 esun_earningcall.txt，回傳一個包含所有 named entity 的 list。
    會 strip 掉每行首尾空白與換行符號，並過濾掉空行。
    """
    entities = []
    with open(entity_path, "r", encoding="utf-8") as f:
        for line in f:
            ent = line.strip()
            if ent:
                entities.append(ent)
    return entities

def is_english_like(ent: str) -> bool:
    """
    判斷這個 entity 是否含有英文字母或數字（即以英文單字匹配方式為主）。
    如果 ent 中任何一個字元屬於 ASCII a–z、A–Z、0–9，就回傳 True；否則視為純 CJK entity 回傳 False。
    """
    return bool(re.search(r"[A-Za-z0-9]", ent))

def extract_entities_from_text(text: str, entities: list) -> list:
    """
    給定一段文字 text，以及所有的 entity list，
    1. 如果 entity 包含英文（字母或數字），就用正則 "\b... \b" 不區分大小寫比對整個單字/片語。
    2. 否則（純中文、純符號等），就做 substring 比對（也就是直接用 'ent in text'）。

    回傳 text 中出現的 entity（不重複，且依照 entities list 的原始順序）。
    """
    found = []
    for ent in entities:
        if is_english_like(ent):
            # 建立一個忽略大小寫 (re.IGNORECASE) 的 pattern，並加入 \b 單字邊界
            # 如果 ent 本身有空格，例如 "NPL Ratio"，re.escape 會自動把空格保留
            pattern = r"\b" + re.escape(ent) + r"\b"
            if re.search(pattern, text, flags=re.IGNORECASE):
                found.append(ent)
        else:
            # 純 CJK 或其他非 ASCII 的詞彙，就直接做 substring
            if ent in text:
                found.append(ent)
    return found

def process_samples(sample_path: str, entities: list, output_path: str, sep: str = " "):
    """
    讀取 samples.txt，逐行解析 sample_id 與文字，然後用 extract_entities_from_text 找出該行文字裡的 entity，
    最後把 sample_id 與對應的 entity（以逗號串聯）寫到 output.txt。

    參數：
      - sample_path: 原始樣本檔 (每行：sample_id + sep + text)
      - entities: 已載入的 entity list
      - output_path: 輸出的結果檔
      - sep: sample_id 與文字之間的分隔符(預設空格)
    """
    with open(sample_path, "r", encoding="utf-8") as fin, \
         open(output_path, "w", encoding="utf-8") as fout:

        for raw_line in fin:
            line = raw_line.rstrip("\n")
            if not line:
                continue

            parts = line.split(sep, 1)
            if len(parts) != 2:
                sample_id = parts[0].strip()
                text = ""
            else:
                sample_id, text = parts[0].strip(), parts[1].strip()

            matched = extract_entities_from_text(text, entities)

            if matched:
                fout.write(f"{sample_id}\t{','.join(matched)}\n")
            else:
                fout.write(f"{sample_id}\t\n")

if __name__ == "__main__":
    # --- 1. 設定檔案路徑（請依照實際位置修改） ---
    sample_file   = "/share/nas169/jerryyang/espnet/egs2/esun/work/dump/raw/train_sp/text"          # 每行：sample_id + sep + ground-truth text
    entity_file   = "/share/nas169/jerryyang/espnet/egs2/esun/work/local/contextual/rarewords/esun_earningcall.txt" # 每行一個 named entity
    output_file   = "/share/nas169/jerryyang/espnet/egs2/esun/work/dump/raw/train_sp/output.txt"           # 最終輸出的檔案
    
    # --- 2. 載入所有 entity ---
    entities = load_entities(entity_file)
    
    # --- 3. 處理並輸出 ---
    # 如果 sample_id 與文字之間不是空格，而是 tab 或其他符號，
    # 可以把 sep=" " 改成 sep="\t" 或其他字元。
    process_samples(sample_file, entities, output_file, sep=" ")
    
    print(f"已完成：{sample_file} → {output_file}")
