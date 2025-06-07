# 需要先安裝 transformers：pip install transformers
from transformers import AutoTokenizer

# 載入 Llama 3.2-1B Instruct 的 tokenizer（需要信任 remote code）
# 如果你使用的 Llama3 版本不同，可將此處 “meta-llama/Llama-3-1B-Instruct” 換成對應模型名稱或路徑
tokenizer = AutoTokenizer.from_pretrained(
    "meta-llama/Llama-3.2-1B",
    # use_auth_token=True,
    # trust_remote_code=True
)

total_tokens = 0
token_counts = []
entity_list = "/share/nas169/jerryyang/espnet/egs2/esun/work/local/contextual/rarewords/esun_earningcall.txt"

# 假設 esun_earningcall.txt 和這支腳本放在同一個工作目錄
with open(entity_list, "r", encoding="utf-8") as f:
    for line in f:
        entity = line.strip()
        if not entity:
            continue

        # 將單一 entity 轉成 token id 列表
        encoded = tokenizer(entity)
        length = len(encoded["input_ids"])
        token_counts.append((entity, length))
        total_tokens += length

# 印出每個 entity 對應的 token 長度
for entity, count in token_counts:
    print(f"{entity} → {count} tokens")

# 印出整個列表的總 token 數
print("-----")
print("總共的 token 數：", total_tokens)
