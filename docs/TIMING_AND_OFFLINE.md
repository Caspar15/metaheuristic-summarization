# 計時與離線使用說明

本文件彙整整個兩階段流程的時間紀錄檔案與一鍵彙總欄位，並說明如何以本機快取離線使用 Hugging Face 模型，避免 429。

## 計時與彙總

- Stage1/Stage2 選句時間
  - 檔案: `runs/<stamp>/time_select_seconds.txt`
  - 產生者: `python -m src.pipeline.select_sentences`

- Union 建構時間
  - 檔案: `<union_out>.time_union_seconds.txt`
  - 產生者: `scripts/build_union_stage2.py`

- 評估時間
  - 檔案: `runs/<stage2-stamp>/metrics.csv` 內的 `time_eval_seconds`
  - 產生者: `python -m src.pipeline.evaluate`

- 一鍵彙總（含各階段時間與 ROUGE）
  - 檔案: `runs/timed_summary_<timestamp>.csv`
  - 腳本: `scripts/run_two_stage_timed.py`
  - 欄位包含:
    - `time_stage1_base`, `time_stage1_llm`, `time_union`, `time_stage2`, `time_eval`
    - `rouge1`, `rouge2`, `rougeL`

## 離線與本機快照（避免 429）

- 建議在離線或有限網路下設定環境變數（PowerShell 範例）:
  - `$env:TRANSFORMERS_OFFLINE = "1"; $env:HF_HUB_OFFLINE = "1"`
  - 可選: `$env:HF_HOME = "C:\\Users\\<user>\\.cache\\huggingface"`

- 也可將 LLM 設定中的 `bert.model_name` 指向本機快照路徑，完全不連線:
  - 位置範例:
    - `C:\\Users\\<user>\\.cache\\huggingface\\hub\\models--bert-base-uncased\\snapshots\\<SHA>`
    - `C:\\Users\\<user>\\.cache\\huggingface\\hub\\models--roberta-base\\snapshots\\<SHA>`
    - `C:\\Users\\<user>\\.cache\\huggingface\\hub\\models--xlnet-base-cased\\snapshots\\<SHA>`
  - 對應設定檔: `configs/stage1/llm/{bert|roberta|xlnet}/k{K}.yaml`

- 其他注意:
  - 初次下載請在有網路時以模型名（如 `bert-base-uncased`）載入，讓快照寫入本機快取。
  - 對 RoBERTa：我們會自動關閉 pooling layer 的載入，避免未初始化權重訊息；不影響結果。
  - 對 XLNet：程式會自動使用慢速 tokenizer（減少額外依賴），仍建議安裝 `sentencepiece`、`protobuf`。

