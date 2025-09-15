# 抽取式摘要系統（Extractive Summarization）

本專案的詳細文件集中於 `docs/`：
- Pipeline 說明（含二階段流程）：`docs/PIPELINE.md`
- 輸出與總表（runs 與 tune_summary）：`docs/RUNS.md`
- 設定檔總覽與欄位意義：`docs/CONFIGS.md`
- 清理與整理建議：`docs/CLEANUP.md`

簡介（保留原架構並修正內容）
- 模組化抽取式摘要系統，針對 CNN/DailyMail 類資料建立實驗環境。
- 內建基線特徵（TF‑ISF、句長、句位）與多種選句器（Greedy/GRASP/NSGA‑II）。
- 新增不需監督訓練的 BERT 排序與「融合 + MMR」（fused）；另提供不依賴 BERT 的 fast 系列（TF‑IDF semantic + MMR）。
- 支援以「句數」為單位的長度控制（例如嚴格輸出 3 句）。
- 已移除監督式（supervised）相關模組；若 `representations.use=false`，Greedy/GRASP 仍可運作，NSGA‑II 會自動回退 Greedy。
- 二階段流程：Stage1 兩路 Top‑K（Base 與 BERT）→ 聯集 U → Stage2 於 U 上做最終決策。

## 功能總覽
- 基線特徵融合與 Greedy 的冗餘抑制（MMR）。
- 進階選句：GRASP（隨機化貪婪 + 局部搜尋）、NSGA‑II（多目標覆蓋/冗餘）。
- 表示與相似度：TF‑IDF 或 SBERT 句向量，cosine 相似度矩陣。
- 候選池（Top‑K）支援 hard/soft 與來源聯集（score/position/centrality），可選 `recall_target` 動態擴 k。
- BERT 排序（`optimizer: bert`）與 融合 + MMR（`optimizer: fused`）。
- Fast 系列（`optimizer: fast|fast_grasp|fast_nsga2|fast_fused|tfidf_fused`）：TF‑IDF semantic + MMR（`fast_fused` 與 `tfidf_fused` 為等價別名）。

## 專案結構（以實際檔案為準）
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
  build_union_stage2.py
  tune_union_fusion.py
src/
  data/
  features/
  models/
    extractive/
      greedy.py, grasp.py, nsga2.py, bert_rank.py, fused.py, fast_fused.py
  pipeline/
  representations/
  selection/
  eval/
experimental/
  abstractive/
    bart_cnn.py, pegasus_cnn_dm.py
  rerank/
    cross_encoder.py
  pipeline/
    rerank.py
```

## 安裝與環境
- Python 3.10 以上。
- 建議使用虛擬環境：
  - Windows: `python -m venv .venv && .\\.venv\\Scripts\\activate`
  - Unix: `python -m venv .venv && source .venv/bin/activate`
- 安裝依賴：`pip install -r requirements.txt`
- 可選依賴：
  - 評估：`rouge-score`
  - SBERT/BERT：`torch`, `transformers`, `sentence-transformers`
  - NSGA‑II：`pymoo==0.6.1.1`

## 資料準備（資料欄位與轉換）
- CSV 欄位：`id, article, highlights`
- 切分資料集：`python scripts/split_dataset.py --input path/to/all.csv --out_dir data/raw`
- JSONL/CSV 互轉：`python scripts/jsonl_to_csv.py`

## 快速開始
1) 預處理（分句與過濾短句）
```
python -m src.data.preprocess \
  --input data/raw/validation.csv \
  --split validation \
  --out data/processed/validation.jsonl \
  --max_sentences 25
```

2) 產生摘要（單段）
```
python -m src.pipeline.select_sentences \
  --config configs/features_basic.yaml \
  --split validation \
  --input data/processed/validation.jsonl \
  --run_dir runs \
  --optimizer greedy
```

可選：以小樣本基準跑三個 split 與多種優化器
```
python scripts/benchmark_small.py \
  --config configs/features_basic.yaml \
  --raw_dir data/raw \
  --processed_dir data/processed \
  --run_dir runs \
  --optimizers greedy,grasp,nsga2 \
  --max_sentences 25
```

3) 評估 ROUGE
```
python -m src.pipeline.evaluate \
  --pred runs/<stamp>/predictions.jsonl \
  --out  runs/<stamp>/metrics.csv
```

## 三句摘要（常用）
- 設定：`configs/features_3sent.yaml`
  - `length_control.unit: sentences`
  - `length_control.max_sentences: 3`
- 執行（以 validation 為例）
```
python -m src.pipeline.select_sentences \
  --config configs/features_3sent.yaml \
  --split validation \
  --input data/processed/validation.jsonl \
  --run_dir runs \
  --optimizer nsga2   # 或 greedy / grasp
```
註：NSGA‑II 需 `pymoo`；若 `representations.use=false` 無相似度，會回退 Greedy。

## 二階段流程與範例
- Stage1 Base：`configs/features_20sent.yaml` + `--optimizer greedy|grasp|nsga2`，以 `unit: sentences, max_sentences: K1` 輸出 Top‑K1。
- Stage1 BERT：`configs/features_bert_20sent.yaml` + `--optimizer bert`，以 `max_sentences: K2` 輸出 Top‑K2。
- 聯集 U：
```
python scripts/build_union_stage2.py \
  --input data/processed/validation.jsonl \
  --base_pred runs/<stage1-base>/predictions.jsonl \
  --bert_pred runs/<stage1-bert>/predictions.jsonl \
  --out data/processed/validation.stage2.union.jsonl \
  --cap 25
```
- Stage2 最終決策：於 U 上以 `--optimizer bert|fused|fast|fast_grasp|fast_nsga2` 運行。
  - 注意：部份 Stage2 模板（如 `features_bert_3sent.yaml`, `features_fused_3sent.yaml`）由 `scripts/tune_union_fusion.py` 動態產生於 `configs/_generated/`，不一定存在於 `configs/` 根目錄。

補充：也可使用 `scripts/grid_stage2_fast.py` 或 `scripts/tune_union_fusion.py` 做網格與總表輸出（見 `docs/RUNS.md`）。

## 設定概覽（Pipeline Overview）
- Prepare（A）
  - P0: 準備資料或用 `scripts/split_dataset.py` 生成 CSV。
  - P1: 清理與分句 `src/data/preprocess.py`。
- Represent（B）
  - R1: 句向量 `src/representations/sent_vectors.py`。
  - R2: 類似度矩陣 `src/representations/similarity.py`。
- Select（C）
  - R4: 候選池 Top‑K（`src/pipeline/select_sentences.py`）。
  - R5: 選句器 `src/models/extractive/{greedy,grasp,nsga2}.py`。
  - R3: 冗餘/覆蓋（Greedy/GRASP 用 MMR；NSGA‑II 顧及覆蓋/冗餘）。
- Score & Output（D）
  - D2: 分數融合與長度控制（`features/compose.py`, `selection/length_controller.py`）。
  - D3: 輸出 `runs/<stamp>/predictions.jsonl`。
  - D4: 評估（`src/pipeline/evaluate.py`）。

附註 / 建議
- R4（Top‑K）可在較早階段對候選規模與相似度計算做節流；若後續考慮 LLM rerank 建議 `mode: soft` 並放大 k 以提高召回。
- `representations.use=false` 時，Greedy/GRASP 正常；NSGA‑II 會回退 Greedy（已在程式中處理）。

## 兩種「句數控制」差別
- 預處理 `--max_sentences`：控制每篇進入系統的候選數量（影響計算量與記憶體）。
- 輸出 `length_control.max_sentences`：控制最終摘要的句數（不影響候選規模）。

## 疑難排解（Troubleshooting）
- 缺少 rouge-score：`pip install -r requirements.txt` 或單獨安裝 `rouge-score`。
- NSGA‑II 使用錯誤：確認 `pymoo` 版本與相似度存在；否則將回退 Greedy。
- SBERT/transformers：若使用 `sbert/bert/fused`，請確認 `torch`, `transformers`, `sentence-transformers` 已安裝；否則改用 `tfidf` 或 fast 系列。
- 記憶體壓力：降低預處理的 `--max_sentences`、減小 `candidates.k`，或改用 fast 系列。
- Windows/編碼：本庫以 UTF‑8 編碼；建議在 PowerShell 下使用 UTF‑8（預設支援）。

## Pipeline 概覽（修正版，對齊目前實作）
- Datasets/Preprocess（A）
  - 輸入 CSV（`id, article, highlights`）→ `src.data.preprocess` 清理、分句、濾短句，輸出 JSONL（可 `--max_sentences`）。
- Represent（B，可選）
  - 句向量：TF‑IDF 或 SBERT；相似度：cosine。`representations.use=false` 時跳過，greedy/GRASP 正常，NSGA‑II 需相似度。
- Stage1（C，兩路並行）
  - Base 路徑：可用 `candidates` 建 shortlist（hard/soft + score/position/centrality + `recall_target`）；選句器 `greedy|grasp|nsga2`；輸出 Top‑K1（以 `unit: sentences, max_sentences: K1` 控制）。
  - BERT 路徑：BERT 句向量對文件重心排序；通常不啟用 `representations/candidates`；輸出 Top‑K2。
- Union（D，中間產物）
  - `scripts/build_union_stage2.py` 將 K1、K2 聯集為 U，支援 `--cap` 上限與 TF‑IDF 去重；產出新的 JSONL（只含 U 的句子）。
- Stage2（E，最終選擇 K=3）
  - `bert`：直接在 U 上以 BERT 排序取 3 句。
  - `fused`：base 分數與 BERT 分數加權，並以 BERT 相似度做 MMR 冗餘抑制。
  - `fast|fast_grasp|fast_nsga2`：不依賴 BERT，改用 TF‑IDF semantic 得分與相似度做加權 + MMR/GRASP/NSGA‑II。
- Evaluate（F）
  - `src.pipeline.evaluate` 計算 ROUGE，並輸出選句與評估耗時。

## 變更記錄（重點）
- 移除監督式訓練模組。
- 候選池支援 hard/soft、來源聯集與 `recall_target`。
- 支援 `length_control.unit: sentences` 的嚴格句數控制。
- 新增 `optimizer: bert` 與 `optimizer: fused`，以及 fast 系列。

## 目前可用的設定檔（configs）
- `configs/features_basic.yaml`：單段模板（tokens 控制）。
- `configs/features_3sent.yaml`：三句模板（sentences 控制）。
- `configs/features_20sent.yaml`：Stage1 Base Top‑K 模板。
- `configs/features_bert_20sent.yaml`：Stage1 BERT Top‑K 模板。
- `configs/features_fast_3sent.yaml`：Stage2 fast（三句；TF‑IDF semantic）。
- `configs/_generated/**`：由 `scripts/tune_union_fusion.py` 產生的 Stage1/Stage2 變體（包含 `stage2_bert.yaml`, `stage2_fused_*.yaml`, `stage2_fast_*.yaml`）。
