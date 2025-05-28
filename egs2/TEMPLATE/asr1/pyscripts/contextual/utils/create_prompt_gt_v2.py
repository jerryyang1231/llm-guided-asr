import re
import random
from collections import defaultdict
from pathlib import Path
from tqdm import tqdm

class PromptGenerator:
    def __init__(self, entity_files, random_fill=False, target_entity_count=10):
        """
        初始化 Prompt 生成器
        
        :param entity_files: 字典，key為split名稱（train_sp/dev/test），value為對應的實體列表檔案路徑
        :param random_fill: 是否隨機補充實體至指定數量
        :param target_entity_count: 目標實體數量（用於隨機補充）
        """
        self.random_fill = random_fill
        self.target_entity_count = target_entity_count
        self.entities = {}
        for split, file_path in entity_files.items():
            self.entities[split] = self._load_entities(file_path)
        
    def _load_entities(self, entity_file):
        """載入實體列表"""
        with open(entity_file, 'r', encoding='utf-8') as f:
            # 確保每個實體都是單詞（不包含空格）
            entities = []
            for line in f:
                word = line.strip()
                if word and ' ' not in word:  # 只接受不包含空格的單詞
                    entities.append(word)
            return entities
    
    def _find_entities_in_text(self, text, split):
        """在文本中找出所有完全匹配的單詞實體"""
        if split not in self.entities:
            raise ValueError(f"找不到 {split} 的實體列表")
        
        # 將文本分割成單詞
        words = text.split()
        # 只保留完全匹配的單詞
        found_entities = set(word for word in words if word in self.entities[split])
        
        return list(found_entities)
    
    def _generate_prompt(self, line_id, text, split):
        found_entities = self._find_entities_in_text(text, split)
        
        if not found_entities:
            return f"{line_id} Start transcribe."
            
        if len(found_entities) > 10:
            # 過濾三個字以上的實體
            filtered_entities = [entity for entity in found_entities if len(entity) > 3]
            if len(filtered_entities) > 10:
                found_entities = random.sample(filtered_entities, 10)
            else:
                found_entities = filtered_entities
        elif self.random_fill and len(found_entities) < self.target_entity_count:
            # 如果實體數量少於目標數量，保留所有實體並隨機補充
            additional_count = self.target_entity_count - len(found_entities)
            available_entities = list(set(self.entities[split]) - set(found_entities))
            if available_entities:
                additional_entities = random.sample(
                    available_entities,
                    min(additional_count, len(available_entities))
                )
                found_entities.extend(additional_entities)
        
        entities_str = ', '.join(found_entities)
        return f"{line_id} Keywords:{entities_str}. Start transcribe."
    
    def process_file(self, input_file, output_file, split):
        """處理單個檔案"""
        # 先計算總行數
        with open(input_file, 'r', encoding='utf-8') as f:
            total_lines = sum(1 for _ in f)
            
        with open(input_file, 'r', encoding='utf-8') as f_in, \
             open(output_file, 'w', encoding='utf-8') as f_out:
            for line in tqdm(f_in, total=total_lines, desc=f"處理 {split}", ncols=100):
                line = line.strip()
                if not line:
                    continue
                
                # 分割 ID 和文本
                parts = line.split(maxsplit=1)
                if len(parts) != 2:
                    continue
                    
                line_id, text = parts
                prompt = self._generate_prompt(line_id, text, split)
                f_out.write(f"{prompt}\n")
    
    def process_all_files(self, base_path, splits=('train_sp', 'dev', 'test')):
        """處理所有指定的檔案"""
        base_path = Path(base_path)
        results = {}
        
        for split in splits:
            if split not in self.entities:
                print(f"警告：找不到 {split} 的實體列表，跳過處理")
                continue
                
            input_file = base_path / f"{split}/text"
            output_file = base_path / f"{split}/prompt_gt_context_random.txt"
            
            # 確保輸出目錄存在
            output_file.parent.mkdir(parents=True, exist_ok=True)
            
            try:
                self.process_file(input_file, output_file, split)
                results[split] = str(output_file)
            except Exception as e:
                print(f"處理 {split} 時發生錯誤: {str(e)}")
        
        return results

def main():
    # 設定基本路徑
    base_path = '/share/nas169/yuchunliu/espnet_contextual/espnet/egs2/slidespeech/asr1_small/dump/raw'
    entity_base_path = '/share/nas169/yuchunliu/espnet_contextual/espnet/egs2/slidespeech/asr1_small/biasing_list'
    
    # 設定各split的entity檔案路徑
    entity_files = {
        'train_sp': f"{entity_base_path}/train_entities.txt",
        'dev': f"{entity_base_path}/dev_entities.txt",
        'test': f"{entity_base_path}/test_entities.txt"
    }
    
    random_fill = True  # 設定是否要隨機補充實體
    target_count = 10   # 目標實體數量
    
    try:
        print("=== 開始生成 Prompt ===")
        print(f"隨機補充實體: {'開啟' if random_fill else '關閉'}")
        
        # 初始化生成器
        generator = PromptGenerator(
            entity_files=entity_files,
            random_fill=random_fill,
            target_entity_count=target_count
        )
        
        # 處理所有檔案
        results = generator.process_all_files(base_path)
        
        print("\n處理完成！")
        for split, output_path in results.items():
            print(f"{split} 的 prompt 已儲存至: {output_path}")
            
    except Exception as e:
        print(f"\n錯誤：{str(e)}")
        raise

if __name__ == "__main__":
    main()