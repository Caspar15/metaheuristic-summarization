# 專案狀態（metaheuristic-summarization）

更新日期：2025-09-16

**近期更新**
- 統一 Stage1 LLM 選句至單一路徑：`src/models/extractive/encoder_rank.py`（支援 `bert|roberta|xlnet`）。
- 移除三階段與個別 roberta/xlnet 的舊實作；相關 YAML 歸檔至 `configs/_archive/`。
- Stage2 僅保留 fast 系列（`fast|fast_grasp|fast_nsga2`），不再使用 bert/fused。
- 新增 `scripts/run_two_stage_timed.py`：一次執行二階段並彙總各階段時間與 ROUGE。
- README 與 docs 中文化與整理（移除亂碼/過時內容）。

**已完成（Completed）**
- 前處理：`src/data/preprocess.py`（切句/濾短/句數上限；CSV→JSONL）。
- 基線特徵：`src/features/{tf_isf,length,position,compose}.py`。
- 表示/相似度：`src/representations/{sent_vectors,similarity}.py`（TF‑IDF/SBERT + cosine）。
- 候選池：`src/pipeline/select_sentences.py`（hard/soft + 多來源 union + 可選 `recall_target`）。
- 選句器：Greedy（MMR）、GRASP、NSGA‑II、Encoder（BERT/RoBERTa/XLNet）、Fast 系列。
- I/O：`predictions.jsonl`, `config_used.json`, `time_select_seconds.txt`。
- 評估：`src/pipeline/evaluate.py`, `src/eval/rouge.py`（ROUGE）。
- 腳本：`benchmark_small.py`, `split_dataset.py`, `jsonl_to_csv.py`, `build_union_stage2.py`, `tune_union_fusion.py`, `grid_stage2_fast.py`, `run_two_stage_timed.py`。

**待辦（TODO）**
- 單元測試：補齊 `tests/`（候選池、相似度、長度控制、選句器介面）。
- 進一步清理歷史 `_generated/` 產物與歸檔策略。

**流程對照（Flow Mapping）**
- Prepare：`split_dataset.py` → `preprocess`。
- Represent：句向量與相似度（TF‑IDF/SBERT + cosine）。
- Select：候選池 Top‑K → 選句（Greedy/GRASP/NSGA‑II/Encoder/Fast）。
- Output：`predictions.jsonl` → `evaluate`（ROUGE）。

**Runbook（速查）**
- 前處理：
  - `python -m src.data.preprocess --input data/raw/validation.csv --split validation --out data/processed/validation.jsonl --max_sentences 25`
- 單段：
  - `python -m src.pipeline.select_sentences --config configs/stage1/base/k20.yaml --split validation --input data/processed/validation.jsonl --run_dir runs --optimizer greedy`
- 二階段：
  - Stage1 Base：`--config configs/stage1/base/k20.yaml --optimizer greedy|grasp|nsga2`
  - Stage1 LLM：`--config configs/stage1/llm/{bert|roberta|xlnet}/k20.yaml --optimizer bert|roberta|xlnet`
  - 聯集 U：`python scripts/build_union_stage2.py --input ... --base_pred ... --bert_pred ... --out ... --cap 25`
  - Stage2：`--optimizer fast|fast_grasp|fast_nsga2`
- 一鍵計時：
  - `python scripts/run_two_stage_timed.py --input data/processed/validation.jsonl --run_dir runs --base_cfg configs/stage1/base/k20.yaml --opt1 nsga2 --llm_cfg configs/stage1/llm/roberta/k20.yaml --opt2 roberta --cap 15 --stage2_cfg configs/stage2/fast/3sent.yaml --opt3 fast`


