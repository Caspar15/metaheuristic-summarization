# metaheuristic-summarization 專案狀態

更新日期：2025-09-13

**近期重點更新**
- 已移除監督式（supervised）相關模組，以可組合的基線特徵與選句器為主。
- 候選池（Top‑K）支援 hard/soft、來源聯集（score/position/centrality），並可用 `recall_target` 動態擴 k。
- `representations.use=false` 時跳過相似度建立：Greedy/GRASP 正常、NSGA‑II 自動回退 Greedy。
- 新增「以句數」為單位的長度控制（`length_control.unit: sentences`, `max_sentences`），可嚴格輸出固定句數（如 3 句）。
- Rerank 與 Abstractive 移至 `experimental/`（保留範例，不影響主流程）。
- `optimizer: bert`（BERT 排序）與 `optimizer: fused`（base+BERT 融合後以 BERT 相似度做 MMR）。
- 二階段流程：Stage1 兩路 Top‑K（Base 與 BERT）→ 聯集 U → Stage2 最終決策。提供 `scripts/build_union_stage2.py` 與 `scripts/tune_union_fusion.py`。
  - Stage2 可選 fast 系列（TF‑IDF semantic + MMR）以免安裝/下載 BERT；支援 `fast|fast_grasp|fast_nsga2|fast_fused|tfidf_fused`。

## 已完成（Completed）
- 預處理（Prepare/P1）
  - `src/data/preprocess.py`：正規分句、過濾短句、限制每篇最大句數；輸入 CSV、輸出 JSONL。
- 特徵與基線（Features & Base Scoring）
  - `src/features/{tf_isf.py,length.py,position.py,compose.py}`：TF‑ISF、長度、句位特徵，線性加權並歸一化。
- 表示與相似度（Representations）
  - `src/representations/{sent_vectors.py, similarity.py}`：TF‑IDF 與 SBERT 句向量、cosine 相似度。
- 選句與長度控制（Selection & Length Control）
  - `src/models/extractive/{greedy.py,grasp.py,nsga2.py}` 與 `src/selection/length_controller.py`：Greedy（MMR）、GRASP（RCL+局搜）、NSGA‑II（多目標）。支援 tokens 或 sentences 長度控制。
- 候選池（Shortlist Top‑K）
  - `src/pipeline/select_sentences.py`：`mode: hard|soft`、`sources: [score|position|centrality]` 聯集；可設 `recall_target` 自動擴 k。
- Pipeline I/O
  - `src/pipeline/select_sentences.py`：輸出 `runs/<stamp>/predictions.jsonl` 與 `config_used.json`。
- 評估（Evaluation）
  - `src/pipeline/evaluate.py`, `src/eval/rouge.py`：計算 ROUGE，輸出 `metrics.csv`。
- 腳本（Scripts）
  - `scripts/{benchmark_small.py,split_dataset.py,jsonl_to_csv.py,build_union_stage2.py,tune_union_fusion.py}`：資料轉換、流程整合、二階段調參。
- 其他 Optimizers
  - `src/models/extractive/{bert_rank.py,fused.py,fast_fused.py}`：BERT 排序；base+BERT 融合+MMR；TF‑IDF semantic 快速版本。

## 進行中 / 待辦（TODO）
- 測試：`tests/` 目前為 TODO 占位，需補單元測試（候選池、相似度、長度控制、選句器介面）。
- Rerank：cross‑encoder（`experimental/`）尚未整合主流程，未來可做 Stage2 的 rerank。
- 設定模板：部份 Stage2 模板（如 `features_bert_3sent.yaml`, `features_fused_3sent.yaml`）由調參腳本動態產生於 `configs/_generated/`，如需固定模板可另行複制保存。

## 整體架構與完成度（What’s Done / Missing）
- 資料與前處理
  - [完成] CSV → JSONL：`src/data/preprocess.py`（分句、濾短、上限句數、抽樣）。
  - [完成] 範例與切分腳本：`scripts/split_dataset.py`, `scripts/jsonl_to_csv.py`。
- 表示與相似度（Represent）
  - [完成] TF‑IDF 句向量與 cosine 相似度：`src/representations/*`。
  - [可選/完成] SBERT 句向量（需 `sentence-transformers`/`torch`）。
- 候選池（Candidates）
  - [完成] hard/soft 模式、來源聯集（score/position/centrality）。
  - [完成] `recall_target` 動態擴 k（需 `rouge-score`）。
- Stage1（兩路）
  - [完成] Base：`greedy`, `grasp`, `nsga2`（`pymoo`）。
  - [完成] BERT：`optimizer: bert`（AutoModel 平均池化，對文件重心排序）。
- 聯集 U（Stage2 中間物）
  - [完成] 聯集與上限 `--cap`、TF‑IDF 去重：`scripts/build_union_stage2.py`。
- Stage2（最終 3 句）
  - [完成] `bert`：直接排序取句。
  - [完成] `fused`：base+BERT 加權 + BERT 相似度 MMR（需 `torch/transformers`）。
  - [完成] `fast|fast_grasp|fast_nsga2`：TF‑IDF semantic + MMR/GRASP/NSGA‑II（免 BERT）。
- 評估與產出
  - [完成] `predictions.jsonl`, `metrics.csv`（ROUGE），時間統計：`src/pipeline/evaluate.py`。
- 腳本與自動化
  - [完成] 小樣本基準：`scripts/benchmark_small.py`。
  - [完成] 二階段調參與彙總：`scripts/tune_union_fusion.py`（輸出 `_generated/` 與 `tune_summary_*.csv`）。
  - [完成] fast 系列網格：`scripts/grid_stage2_fast.py`；彙總/整理：`scripts/summarize_runs.py`, `scripts/organize_runs.py`, `scripts/cleanup_runs.ps1`。
- 設定檔
  - [完成] 基本模板：`configs/features_{basic,3sent,20sent,bert_20sent,fast_3sent}.yaml`。
  - [完成] 產生式模板：`configs/_generated/**`（由腳本輸出 stage1/stage2 變體）。
- 實驗性/未整合
  - [實驗] Rerank cross‑encoder：`experimental/rerank`, `experimental/pipeline/rerank.py`（未併入主流程）。
  - [實驗] Abstractive（BART/Pegasus）：`experimental/abstractive/*`。
- 尚未實作 / 待規劃
  - [缺] 自動化單元測試（`tests/`）。
  - [缺] 強化學習（RL）策略/訓練。
  - [缺] 更完整的多語處理與 tokenizer‑based 長度控制（目前以空白切詞）。

## 流程對照（Flow Mapping）
- A Prepare：資料就緒與清理（split_dataset.py → preprocess）
- B Represent：句向量與相似度（TF‑IDF/SBERT → cosine）
- C Select：候選池 Top‑K → 選句（Greedy/GRASP/NSGA‑II；冗餘/覆蓋）
- D Output：輸出 predictions.jsonl 與 ROUGE

## Runbook（快速檢查）
- 安裝環境
  - Python 3.10+；`pip install -r requirements.txt`

- 預處理（validation 範例）
  - `python -m src.data.preprocess --input data/raw/cnn_dailymail/validation.csv --split validation --out data/processed/validation.jsonl --max_sentences 25`

- 產生摘要（單段）
  - `python -m src.pipeline.select_sentences --config configs/features_basic.yaml --split validation --input data/processed/validation.jsonl --run_dir runs --optimizer greedy`
  - 三句摘要：`--config configs/features_3sent.yaml` 並設定 `length_control.unit: sentences, max_sentences: 3`
  - BERT 排序（單段）：使用 `--optimizer bert`（三句模板可由 `scripts/tune_union_fusion.py` 產生於 `_generated/`）
  - 可選快速基準：`python scripts/benchmark_small.py --config configs/features_basic.yaml --raw_dir data/raw --processed_dir data/processed --run_dir runs --optimizers greedy,grasp,nsga2 --max_sentences 25`

- 二階段（Stage1/2）
  - K1（Base, 20 句）：`python -m src.pipeline.select_sentences --config configs/features_20sent.yaml --split validation --input data/processed/validation.jsonl --run_dir runs --optimizer greedy --stamp stage1-greedy-top20`
  - K2（BERT, 20 句）：`python -m src.pipeline.select_sentences --config configs/features_bert_20sent.yaml --split validation --input data/processed/validation.jsonl --run_dir runs --optimizer bert --stamp stage1-bert-top20`
  - 聯集 U：`python scripts/build_union_stage2.py --input data/processed/validation.jsonl --base_pred runs/stage1-greedy-top20/predictions.jsonl --bert_pred runs/stage1-bert-top20/predictions.jsonl --out data/processed/validation.stage2.union.jsonl --cap 25`
  - 最終 3 句（於 U 上）：使用 `--optimizer bert|fused|fast|fast_grasp|fast_nsga2`。部分 Stage2 模板（`features_bert_3sent.yaml`, `features_fused_3sent.yaml`）由 `scripts/tune_union_fusion.py` 產生於 `configs/_generated/`。
  - 補充腳本：`scripts/grid_stage2_fast.py` 可對 fast 系列做網格搜尋與輸出彙總（詳見 `docs/RUNS.md`）。

- 評估 ROUGE
  - `python -m src.pipeline.evaluate --pred runs/<stamp>/predictions.jsonl --out runs/<stamp>/metrics.csv`

## 相容性與注意事項
- Python：建議 3.10+；Windows 預設使用 `orjson`，其他平台使用 `ujson`（在 `requirements.txt` 有條件安裝）。
- 相似度矩陣記憶體：若上限句數較大，可降低 `--max_sentences` 或使用 fast 系列；或縮小 `candidates.k`。
- `representations.use=false`：Greedy/GRASP 仍可運行；NSGA‑II 將自動回退 Greedy（在 pipeline 內已處理）。
- 編碼：本倉庫以 UTF‑8 儲存；若先前文件出現亂碼，請以 UTF‑8 檢視或參考本次更新版本。
 - 測試：`tests/` 為 TODO 占位，尚未納入自動化測試；請以 `scripts/benchmark_small.py` 或小集驗證功能。

## 參考成績（示意）

- CNN/DailyMail validation（全量，單段，候選上限 25）
  - Greedy：R1=0.3174, R2=0.1132, RL=0.1900
  - GRASP：R1=0.3175, R2=0.1131, RL=0.1900
  - NSGA‑II：R1=0.3209, R2=0.1162, RL=0.1928（需 `pymoo`）
- CNN/DailyMail validation（sample=100；二階段）
  - BERT（於 U，K1=20, K2=20, cap=20）：RL≈0.2096（示意）
  - fused（w_bert=0.5, α=0.7，cap=25）：RL≈0.2049（示意，待調整）
  註：候選池大小與 cap 會影響結果；數值僅供對照與迭代。
