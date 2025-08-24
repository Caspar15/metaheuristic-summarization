# 抽取式摘要專案交接文檔

本專案是一個模組化抽取式摘要系統，以 CNN/DailyMail 資料集為主要測試場景，支援規則特徵 Baseline 與可插拔的進階方法（SBERT、GRASP/NSGA-II、BART/Pegasus 等）。

## 專案目標
- Baseline 抽取式摘要：使用規則特徵 (TF-ISF、長度、句位) + Greedy 選句。
- 進階優化：句子表示、相似度矩陣、冗餘控制與超啟發式演算法 (GRASP / NSGA-II)。
- AI 整合：支援以預訓練模型 (BERT, SBERT, Pegasus, BART 等) 作為句子評分器或生成式摘要模型，並可做微調（可選）。

## 資料來源
- 使用 CNN/DailyMail（train.csv, validation.csv, test.csv）。
- 欄位：
  - `id`: 文章編號
  - `article`: 完整新聞文章
  - `highlights`: 人工編輯摘要（reference）

## 系統流程
1. 前處理 (`src/data/preprocess.py`)：讀取 CSV、清理、分句、可選 ADM#1 過濾；輸出 JSONL 至 `data/processed/`。
2. 特徵 (`src/features/`)：TF-ISF、句長度、句位，並在 `compose.py` 合成句子分數。
3. 表示與相似度 (`src/representations/`)：`sent_vectors.py`（TF-IDF 或 SBERT）與 `similarity.py`（cosine）。
4. 候選與約束 (`src/selection/`)：`candidate_pool.py`、`redundancy.py`（MMR/Jaccard）、`length_controller.py`。
5. 選句演算法 (`src/models/extractive/`)：`greedy.py`（基線），另含 `grasp.py`、`nsga2.py`（簡化實作／佔位）。
6. 二層打分 (`src/scoring/re_rank.py`)：coverage/coherence rerank（簡化實作）。
7. 生成式（可選） (`src/models/abstractive/`)：`bart_cnn.py`、`pegasus_cnn_dm.py`（需安裝 transformers）。
8. 評估 (`src/pipeline/evaluate.py`)：`rouge-score` 計算 ROUGE-1/2/L，輸出 `metrics.csv` 與 `predictions.jsonl`。

## 專案結構
```
configs/
data/
  raw/
  processed/
runs/
scripts/
src/
  data/
  features/
  representations/
  selection/
  scoring/
  models/
    extractive/
    abstractive/
  pipeline/
  eval/
tests/
```

## 安裝與環境
- Python 3.9+（建議 3.10）
- 建立虛擬環境：
  - Windows: `python -m venv .venv && .\.venv\Scripts\activate`
  - Unix: `python -m venv .venv && source .venv/bin/activate`
- 安裝依賴：`pip install -r requirements.txt`

## 快速開始
1) 前處理
```
python -m src.data.preprocess --input data/raw/train.csv --split train --out data/processed/train.jsonl
```

2) 選句與產生摘要（Greedy Baseline）
```
python -m src.pipeline.select_sentences --config configs/features_basic.yaml --split validation --input data/processed/validation.jsonl --run_dir runs
```

3) 評估 ROUGE
```
python -m src.pipeline.evaluate --pred runs/<timestamp>/predictions.jsonl --out runs/<timestamp>/metrics.csv
```

## Config 範例（configs/features_basic.yaml）
```yaml
objectives:
  lambda_importance: 1.0
  lambda_coverage: 0.8
  lambda_redundancy: 0.7
  length_penalty: 2.0
representations:
  use: true
  method: "tfidf"  # or "sbert"
  cache: true
candidates:
  use: true
  k: 15
  by: "score"      # or "title_sim"
redundancy:
  method: "mmr"    # or "jaccard"
  lambda: 0.7
  sim_metric: "cosine"
length_control:
  unit: "tokens"
  max_tokens: 100
optimizer:
  method: "greedy" # greedy | grasp | nsga2 | supervised | bart | pegasus
seed: 2024
```

## 常見問題
- 初次使用 `nltk` 分句需下載模型，可切換為內建 regex 分句器（預設已使用內建）。
- `rouge-score` 未安裝會導致評估失敗，請先 `pip install -r requirements.txt`。
- SBERT 與生成式模型需 `transformers`、`torch`；若未安裝，對應模組會給出明確錯誤訊息。

## Roadmap（簡）
- 監督式評分（SBERT fine-tune）、NSGA-II 強化、coherence rerank 強化、快取與並行優化。

