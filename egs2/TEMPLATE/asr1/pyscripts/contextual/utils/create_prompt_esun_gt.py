import re
import random
from collections import defaultdict
from pathlib import Path

class PromptGenerator:
    def __init__(self, entity_file, random_fill=False, target_entity_count=10):
        """
        初始化 Prompt 生成器
        
        :param entity_file: 實體列表檔案路徑
        :param random_fill: 是否隨機補充實體至指定數量
        :param target_entity_count: 目標實體數量（用於隨機補充）
        """
        self.random_fill = random_fill
        self.target_entity_count = target_entity_count
        self.entities = self._load_entities(entity_file)
        
    def _load_entities(self, entity_file):
        """載入實體列表"""
        with open(entity_file, 'r', encoding='utf-8') as f:
            return [line.strip() for line in f if line.strip()]
    
    def _process_text_for_matching(self, text):
        """處理文本以準備進行匹配"""
        return text.replace(' ', '').lower()
    
    def _find_entities_in_text(self, text):
        """在文本中找出所有出現的實體"""
        processed_text = self._process_text_for_matching(text)
        found_entities = set()
        
        for entity in self.entities:
            processed_entity = self._process_text_for_matching(entity)
            if processed_entity in processed_text:
                found_entities.add(entity)
                
        return list(found_entities)
    
    def _generate_prompt(self, line_id, text):
        """為單行文本生成提示詞"""
        found_entities = self._find_entities_in_text(text)
        
        if not found_entities:
            return f"{line_id} 開始吧."
        
        # 如果需要隨機補充實體
        if self.random_fill and len(found_entities) < self.target_entity_count:
            # 計算需要補充的數量
            additional_count = self.target_entity_count - len(found_entities)
            # 從未使用的實體中隨機選擇
            available_entities = list(set(self.entities) - set(found_entities))
            if available_entities:
                additional_entities = random.sample(
                    available_entities,
                    min(additional_count, len(available_entities))
                )
                found_entities.extend(additional_entities)
        
        entities_str = ', '.join(found_entities)
        return f"{line_id} 主題為:{entities_str}. 開始吧."
    
    def process_file(self, input_file, output_file):
        """處理單個檔案"""
        with open(input_file, 'r', encoding='utf-8') as f_in, \
             open(output_file, 'w', encoding='utf-8') as f_out:
            for line in f_in:
                line = line.strip()
                if not line:
                    continue
                
                # 分割 ID 和文本
                parts = line.split(maxsplit=1)
                if len(parts) != 2:
                    continue
                    
                line_id, text = parts
                prompt = self._generate_prompt(line_id, text)
                f_out.write(f"{prompt}\n")
    
    def process_all_files(self, base_path, splits=('train_sp', 'dev', 'test')):
        """處理所有指定的檔案"""
        base_path = Path(base_path)
        results = {}
        
        for split in splits:
            input_file = base_path / f"{split}/text"
            output_file = base_path / f"{split}/prompt_new_.txt"
            
            # 確保輸出目錄存在
            output_file.parent.mkdir(parents=True, exist_ok=True)
            
            try:
                self.process_file(input_file, output_file)
                results[split] = str(output_file)
            except Exception as e:
                print(f"處理 {split} 時發生錯誤: {str(e)}")
        
        return results

def main():
    # 設定參數
    base_path = '/share/nas169/yuchunliu/espnet_contextual/espnet/egs2/esun/asr2/dump/raw'
    entity_file = '/share/nas169/yuchunliu/espnet_contextual/espnet/egs2/esun/asr2/local/contextual/rarewords/esun_earningcall.entity.txt'
    random_fill = False  # 設定是否要隨機補充實體
    target_count = 10   # 目標實體數量
    
    try:
        print("=== 開始生成 Prompt ===")
        print(f"隨機補充實體: {'開啟' if random_fill else '關閉'}")
        
        # 初始化生成器
        generator = PromptGenerator(
            entity_file=entity_file,
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