# 抽取式摘要（專案總覽）

文件快速索引（詳見 `docs/`）
- Pipeline（單段/二階段）：docs/PIPELINE.md
- 輸出與總表：docs/RUNS.md
- 設定檔指南：docs/CONFIGS.md
- Scripts 索引：docs/SCRIPTS_AND_CONFIGS.md
- 清理建議：docs/CLEANUP.md

## 概覽
- 模組化抽取式摘要，提供基線特徵（TF‑ISF/句長/句位）與多種選句器（Greedy/GRASP/NSGA‑II）。
- Stage1 提供「編碼器排序」LLM 選句（BERT/RoBERTa/XLNet，統一路徑 encoder_rank）；Stage2 僅使用 fast 系列（TF‑IDF 語義 + MMR/GRASP/NSGA2）。
- 支援以「句數」為單位的長度控制（常見三句摘要）。
- 二階段流程：Stage1 Base + LLM → 聯集 U → Stage2 在 U 上做最終決策（fast 系列）。

## 專案結構（摘錄）
```
configs/
  _generated/           # 腳本產生之設定（歷史上含 bert/fused Stage2；現行僅 fast）
  _archive/             # 歸檔的舊模板
scripts/
  build_union_stage2.py
  benchmark_small.py
  grid_stage2_fast.py
  run_two_stage_timed.py  # 新增：二階段整流程計時彙總
src/
  models/extractive/
    encoder_rank.py     # Stage1 LLM 統一路徑（bert|roberta|xlnet）
    greedy.py, grasp.py, nsga2.py, fast_fused.py
```

## 安裝環境
- Python 3.10+
- 建議使用虛擬環境
  - Windows: `python -m venv .venv && .\.venv\Scripts\activate`
  - Unix: `python -m venv .venv && source .venv/bin/activate`
- 安裝依賴：`pip install -r requirements.txt`
- 可選依賴：`rouge-score`（評估）、`torch`/`transformers`（編碼器）、`pymoo==0.6.1.1`（NSGA‑II）

## 資料格式
- CSV 欄位：`id, article, highlights`
- 分割 CSV：`python scripts/split_dataset.py --input path/to/all.csv --out_dir data/raw`
- JSONL/CSV 互轉：`python scripts/jsonl_to_csv.py`

## 快速開始（單段）
1) 前處理（切句/濾短/限制句數）
```
python -m src.data.preprocess \
  --input data/raw/validation.csv \
  --split validation \
  --out data/processed/validation.jsonl \
  --max_sentences 25
```
2) 單段選句（範例 greedy）
```
python -m src.pipeline.select_sentences \
  --config configs/stage1/base/k20.yaml \
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

## 三句摘要（常見）
- 設定：`configs/stage2/fast/3sent.yaml`
  - `length_control.unit: sentences`
  - `length_control.max_sentences: 3`
- 範例
```
python -m src.pipeline.select_sentences \
  --config configs/stage2/fast/3sent.yaml \
  --split validation \
  --input data/processed/validation.jsonl \
  --run_dir runs \
  --optimizer nsga2   # 或 greedy / grasp
```

## 二階段範例
- Stage1 Base：`configs/stage1/base/k20.yaml` + `--optimizer greedy|grasp|nsga2`（`max_sentences: K1`）
- Stage1 LLM：`configs/stage1/llm/{bert|roberta|xlnet}/k20.yaml` + `--optimizer bert|roberta|xlnet`（`max_sentences: K2`）
- 聯集 U：
```
python scripts/build_union_stage2.py \
  --input data/processed/validation.jsonl \
  --base_pred runs/<stage1-base>/predictions.jsonl \
  --bert_pred runs/<stage1-llm>/predictions.jsonl \
  --out data/processed/validation.stage2.union.jsonl \
  --cap 25
```
- Stage2 最終：在 U 上以 `--optimizer fast|fast_grasp|fast_nsga2`

## 計時腳本（二階段整體執行 + 時間彙總）
- 腳本：`scripts/run_two_stage_timed.py`
- 功能：一次執行 Stage1 Base、Stage1 LLM、Union、Stage2，並彙總各階段時間與 ROUGE 至 `runs/timed_summary_<timestamp>.csv`。
- 範例（100 筆、K=10、cap=15、Stage2=fast；RoBERTa）：
```
python scripts/run_two_stage_timed.py \
  --input data/processed/validation_100.jsonl \
  --run_dir runs \
  --base_cfg configs/stage1/base/k10.yaml --opt1 nsga2 \
  --llm_cfg  configs/stage1/llm/roberta/k10.yaml --opt2 roberta \
  --cap 15 \
  --stage2_cfg configs/stage2/fast/3sent.yaml --opt3 fast \
  --stamp_prefix k10-100
```

## LLM 選句（Stage1 編碼器排序）
- `optimizer.method = bert | roberta | xlnet`（統一路徑）
- AutoModel 最後層均值池化 → 與文件重心 cosine 排序 → 依 `length_control` 取前 N。
- 可用 `bert.model_name` 指定具體模型。

## 疑難排解
- 缺少 rouge-score：請安裝 requirements 或 `pip install rouge-score`
- NSGA‑II：需 `pymoo` 與相似度，否則會退回 Greedy
- 編碼器：需 `torch`/`transformers`；fast 系列不需
- 記憶體：降低 `--max_sentences` 或 `candidates.k`，或改用 fast 系列


