# 檔名: convert_duc2002.py
import os
import pandas as pd
import re

# --- 請根據您的環境修改以下路徑 ---
# 原始文件根目錄
DOCS_ROOT = r'C:\Users\guans\Desktop\metaheuristic-summarization\data\duc2002testdocswithsentences\docs.with.sentence.breaks'

# 標準摘要的根目錄 (*** 這是您需要找到並更新的路徑 ***)
SUMMARIES_ROOT = r'C:\Users\guans\Desktop\metaheuristic-summarization\data\duc2002testdocswithsentences\docs.with.sentence.breaks'

# 輸出 CSV 的路徑
OUTPUT_CSV = r'C:\Users\guans\Desktop\metaheuristic-summarization\data\raw\duc2002_test.csv'
# ------------------------------------

def main():
    """主執行函數"""
    output_dir = os.path.dirname(OUTPUT_CSV)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    tasks = []
    
    try:
        topic_ids = [d for d in os.listdir(DOCS_ROOT) if os.path.isdir(os.path.join(DOCS_ROOT, d))]
    except FileNotFoundError:
        print(f"錯誤：找不到原始文件目錄 '{DOCS_ROOT}'。請檢查 DOCS_ROOT 路徑是否正確。")
        return

    print(f"找到 {len(topic_ids)} 個主題，開始處理...")

    for topic_id in topic_ids:
        article_content_parts = []
        topic_path = os.path.join(DOCS_ROOT, topic_id)
        
        for doc_file in sorted(os.listdir(topic_path)):
            if doc_file.endswith('.S'):
                file_path = os.path.join(topic_path, doc_file)
                with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                    raw_text = f.read()
                    
                    # *** 修改開始: 先找到 <TEXT> 區塊 ***
                    text_block_match = re.search(r'<TEXT>(.*?)</TEXT>', raw_text, re.DOTALL)
                    
                    if text_block_match:
                        text_content = text_block_match.group(1)
                        # *** 再從區塊中提取句子 ***
                        sentences = re.findall(r'<s.*?>(.*?)</s>', text_content, re.DOTALL)
                        clean_text = ' '.join(s.strip() for s in sentences)
                        article_content_parts.append(clean_text)
                    # *** 修改結束 ***
        
        full_article = "\n".join(article_content_parts)

        # 讀取標準摘要 (邏輯不變)
        summary_file_path = os.path.join(SUMMARIES_ROOT, f'{topic_id}.A')
        highlights_content = ""
        if os.path.exists(summary_file_path):
            with open(summary_file_path, 'r', encoding='utf-8', errors='ignore') as f:
                highlights_content = f.read().strip()
        else:
            print(f"警告：在 '{SUMMARIES_ROOT}' 中找不到主題 '{topic_id}' 的摘要檔案，將留空。")

        tasks.append({
            'id': topic_id,
            'article': full_article,
            'highlights': highlights_content
        })

    if not tasks:
        print("錯誤：沒有處理任何主題，不產生 CSV 檔案。")
        return
        
    df = pd.DataFrame(tasks)
    df.to_csv(OUTPUT_CSV, index=False)
    print(f"\n處理完成！")
    print(f"成功轉換 {len(tasks)} 個主題，並儲存至 '{OUTPUT_CSV}'")

if __name__ == '__main__':
    main()