# 抽取式摘要系統（Extractive Summarization）

本專案是一個模組化抽取式摘要系統，預設以 CNN/DailyMail 資料集為測試場景。提供基線規則特徵（TF‑ISF、長度、句位）與多種選句優化（Greedy/GRASP/NSGA‑II），並支援監督式打分與可選的生成式摘要（BART/Pegasus）。

本版已清理舊流程與重複檔案，統一以「前處理 → 選句 → 評估」三段式命令使用。

## 功能總覽
- 抽取式基線：規則特徵加權，Greedy + 相似度冗餘控制。
- 進階優化：GRASP 隨機化貪婪 + 局部搜尋；NSGA‑II 多目標（重要性/覆蓋/冗餘）。
- 監督式打分：以簡單特徵訓練分類器（LogReg），替代基線分數。
- 句向量與相似度：TF‑IDF 或 SBERT 表示，cosine 相似度矩陣。
- 生成式（可選）：BART CNN、Pegasus CNN/DM（需 `transformers`/`torch`）。

## 專案結構
```
configs/
data/
  raw/
  processed/
runs/
scripts/
  benchmark_small.py
  split_dataset.py
  jsonl_to_csv.py
src/
  data/
  features/
  models/
    extractive/
    abstractive/
  pipeline/
  representations/
  selection/
  eval/
```

## 安裝與環境
- Python 3.9+（建議 3.10）
- 建立虛擬環境：
  - Windows: `python -m venv .venv && .\.venv\Scripts\activate`
  - Unix: `python -m venv .venv && source .venv/bin/activate`
- 安裝依賴：`pip install -r requirements.txt`
- 可選依賴（視功能）：
  - 監督式/評估：`rouge-score`, `scikit-learn`
  - SBERT/生成式：`torch`, `transformers`, `sentence-transformers`

## 數據準備
- 預期 CSV 欄位：`id, article, highlights`
- 若你有一個完整 CSV，可用腳本切分：
```
python scripts/split_dataset.py --input path/to/all.csv --out_dir data/raw
```
- 也可用 `scripts/jsonl_to_csv.py` 將 JSONL 轉成 CSV（欄位自動對齊）。

## 快速開始
1) 前處理（分句、過濾短句）
```
python -m src.data.preprocess \
  --input data/raw/validation.csv \
  --split validation \
  --out data/processed/validation.jsonl \
  --max_sentences 25
```

2) 選句與產生摘要（Greedy 基線）
```
python -m src.pipeline.select_sentences \
  --config configs/features_basic.yaml \
  --split validation \
  --input data/processed/validation.jsonl \
  --run_dir runs \
  --optimizer greedy
```

3) 評估 ROUGE
```
python -m src.pipeline.evaluate \
  --pred runs/<stamp>/predictions.jsonl \
  --out  runs/<stamp>/metrics.csv
```

## 設定說明（configs/features_basic.yaml）
```yaml
objectives:
  lambda_importance: 1.0    # 重要性權重
  lambda_coverage: 0.8      # 覆蓋度權重（NSGA-II 標量化）
  lambda_redundancy: 0.7    # 冗餘懲罰權重（NSGA-II 標量化）
  length_penalty: 2.0       # 目前未直接使用，預留
representations:
  use: true
  method: "tfidf"           # tfidf | sbert
  cache: true
candidates:
  use: true
  k: 15                     # 候選池大小（依基線分數）
  by: "score"
redundancy:
  method: "mmr"
  lambda: 0.7               # MMR 調和係數（alpha）
  sim_metric: "cosine"
length_control:
  unit: "tokens"            # tokens | sentences（目前以 tokens 為主）
  max_tokens: 100
optimizer:
  method: "greedy"          # greedy | grasp | nsga2 | supervised | bart | pegasus
seed: 2024
```

## 選句方法
- Greedy（`greedy`）：以 `alpha*score - (1-alpha)*max_sim` 的 MMR 型式逐步選句，受長度上限約束。
- GRASP（`grasp`）：隨機化貪婪建構（RCL）+ swap/add/drop 局部搜尋，取最佳解。
- NSGA‑II（`nsga2`）：多目標（重要性最大、覆蓋最大、冗餘最小），最後以 `lambda_*` 標量化挑代表解。
- Supervised（`supervised`）：先以 `train_supervised.py` 訓練 LogReg，再在 `select_sentences` 指定 `--model`。

## 監督式打分（可選）
1) 用訓練集 JSONL 訓練（內部以 ROUGE‑1 增益的貪婪 oracle 打標）
```
python -m src.pipeline.train_supervised \
  --input data/processed/train.jsonl \
  --out_model runs/supervised/model.joblib \
  --max_tokens 100
```
2) 推論時帶入模型
```
python -m src.pipeline.select_sentences \
  --config configs/features_basic.yaml \
  --split validation \
  --input data/processed/validation.jsonl \
  --run_dir runs \
  --optimizer supervised \
  --model runs/supervised/model.joblib
```

## 生成式摘要（可選）
- 介面提供 `src/models/abstractive/{bart_cnn.py, pegasus_cnn_dm.py}`，需安裝 `transformers`/`torch`。
- 目前主管線未直接串接，可自行呼叫相應函式做對照實驗。

## 腳本工具
- 小樣本基準整套流程（抽樣 train/valid/test + 多種 optimizer + 評估 +彙總）：
```
python scripts/benchmark_small.py \
  --config configs/features_basic.yaml \
  --raw_dir data/raw \
  --train_n 200 --valid_n 50 --test_n 50 \
  --max_sentences 25
```

## 常見問題（FAQ）
- 未安裝 `rouge-score` 導致評估報錯：請先 `pip install -r requirements.txt`。
- 使用 SBERT：請安裝 `torch`, `sentence-transformers`，並在設定檔 `representations.method: sbert`。
- 長度單位：目前以 token 計數（以空白切分），中文文本建議先做斷詞再導入，或降低 `max_tokens`。
- Windows JSON 編碼：程式已以 `utf-8` 讀寫；若路徑含非 ASCII 字元，建議在 PowerShell 下執行並確保編碼為 UTF‑8。

## 變更記錄（重點）
- 2025‑08：清理舊流程與重複檔案（舊 `cli`、`evaluate_run`、部分 `utils`/`eval`/`features` 與示例腳本），統一至三段式命令。

## 授權
- 本專案未附帶明確授權條款。如需發布或再利用，請先於內部確認。
