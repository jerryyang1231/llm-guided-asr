import re
import json
import random
from collections import defaultdict
from pathlib import Path
from tqdm import tqdm

class PromptGenerator:
    def __init__(self, entity_files, pos_files, random_fill=False, target_entity_count=10):
        """
        初始化 Prompt 生成器
        
        :param entity_files: 字典，key為split名稱，value為對應的實體列表檔案路徑
        :param pos_files: 字典，key為split名稱，value為對應的 POS 檔案路徑
        :param random_fill: 是否隨機補充實體至指定數量
        :param target_entity_count: 目標實體數量（用於隨機補充）
        """
        self.random_fill = random_fill
        self.target_entity_count = target_entity_count
        self.keyword_pos = {}
        self.entities = {}
        
        # 載入每個 split 的資料
        for split in entity_files.keys():
            if split in pos_files:
                self.keyword_pos[split] = self._load_keyword_pos(pos_files[split])
            else:
                print(f"警告：找不到 {split} 的 POS 文件")
                
            if split in entity_files:
                self.entities[split] = self._load_entities(entity_files[split])
            else:
                print(f"警告：找不到 {split} 的實體文件")
        
    def _load_keyword_pos(self, pos_file):
        """載入關鍵字和對應的 POS 標籤"""
        with open(pos_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
            return {item['keyword']: item['category'] for item in data}
    
    def _load_entities(self, entity_file):
        """載入實體列表"""
        with open(entity_file, 'r', encoding='utf-8') as f:
            entities = []
            for line in f:
                word = line.strip()
                if word and ' ' not in word:  # 只接受不包含空格的單詞
                    entities.append(word)
            return entities
    
    def _add_pos_tags(self, entity, split):
        """為實體添加 POS 標籤"""
        if split not in self.keyword_pos:
            return entity
            
        pos = self.keyword_pos[split].get(entity)
        if pos:
            return f"<{pos}>{entity}</{pos}>"
        return entity
    
    def _find_entities_in_text(self, text, split):
        """在文本中找出所有完全匹配的單詞實體，並加上 POS 標籤"""
        if split not in self.entities:
            raise ValueError(f"找不到 {split} 的實體列表")
            
        # 將文本分割成單詞
        words = text.split()
        # 只保留完全匹配的單詞，並加上 POS 標籤
        found_entities = set()
        for word in words:
            if word in self.entities[split]:
                found_entities.add(self._add_pos_tags(word, split))
        
        return list(found_entities)
    
    def _generate_prompt(self, line_id, text, split):
        found_entities = self._find_entities_in_text(text, split)
        
        if not found_entities:
            return f"{line_id} Start transcribe."
            
        if len(found_entities) > 10:
            # 過濾已標記的實體（需要考慮標籤）
            filtered_entities = [
                entity for entity in found_entities 
                if len(re.sub(r'<[^>]+>', '', entity)) > 3  # 移除標籤後長度大於3
            ]
            if len(filtered_entities) > 10:
                found_entities = random.sample(filtered_entities, 10)
            else:
                found_entities = filtered_entities
        elif self.random_fill and len(found_entities) < self.target_entity_count:
            # 隨機補充實體（需要加上 POS 標籤）
            current_entities = set(re.sub(r'<[^>]+>', '', entity) for entity in found_entities)
            available_entities = [
                self._add_pos_tags(entity, split)
                for entity in self.entities[split] 
                if entity not in current_entities
            ]
            if available_entities:
                additional_count = self.target_entity_count - len(found_entities)
                additional_entities = random.sample(
                    available_entities,
                    min(additional_count, len(available_entities))
                )
                found_entities.extend(additional_entities)
        
        entities_str = ', '.join(found_entities)
        return f"{line_id} {entities_str}. "
    
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
            output_file = base_path / f"{split}/prompt_gt_context_pos_random.txt"
            
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
    pos_base_path = '/share/nas169/yuchunliu/espnet_contextual/espnet/egs2/slidespeech/asr1_small/biasing_list'  # 設定 POS 文件的基本路徑
    
    # 設定各split的entity檔案路徑
    entity_files = {
        'train_sp': f"{entity_base_path}/train_entities.txt",
        'dev': f"{entity_base_path}/dev_entities.txt",
        'test': f"{entity_base_path}/test_entities.txt"
    }
    
    # 設定各split的POS檔案路徑
    pos_files = {
        'train_sp': f"{pos_base_path}/keyword_classifications_train.json",
        'dev': f"{pos_base_path}/keyword_classifications_dev.json",
        'test': f"{pos_base_path}/keyword_classifications_test.json"
    }
    
    random_fill = True  # 設定是否要隨機補充實體
    target_count = 8   # 目標實體數量
    
    try:
        print("=== 開始生成 Prompt ===")
        print(f"隨機補充實體: {'開啟' if random_fill else '關閉'}")
        
        # 初始化生成器
        generator = PromptGenerator(
            entity_files=entity_files,
            pos_files=pos_files,
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