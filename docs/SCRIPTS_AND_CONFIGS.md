Scripts 與 Configs 整理總表（維護指引）

目的
- 快速瞭解 `scripts/` 與 `configs/` 的用途、典型用法、相依與注意事項。
- 未來清理或擴充時作為對照表，避免路徑/模板不一致。

總覽
- Pipeline 相關腳本：
  - `scripts/benchmark_small.py`：小樣本基準（train/validation/test 各取少量），依多個 optimizer 跑選句與評估，整合結果到 `<stamp>-summary.csv`。
  - `scripts/run_7_2_1.py`：一鍵跑三個 split（可選抽樣/上限句數），為快速端到端驗證的替代方案。
  - `scripts/build_union_stage2.py`：將 Stage1 Base（K1）與 Stage1 BERT（K2）的 `predictions.jsonl` 聯集成 U，支援 `--cap` 與 TF‑IDF 去重；供 Stage2 使用。
  - `scripts/tune_union_fusion.py`：二階段（Stage1/2）調參與彙總：自動產生 `_generated/` YAML；跑 `bert|fused|fast*` 等 Stage2 方法；輸出 `runs/tune_summary_*.csv`。
  - `scripts/grid_stage2_fast.py`：僅針對 Stage2 fast 系列（不依賴 BERT）的網格化組合，含建 U、生成 stage2 YAML、執行與評估。

- 資料與維護腳本：
  - `scripts/split_dataset.py`：將單一 CSV 依比例切分成 `train/validation/test.csv`（欄位需含 `id, article, highlights`）。
  - `scripts/jsonl_to_csv.py`：將 JSONL（含 `id, article, reference/highlights` 欄）轉為 CSV。
  - `scripts/summarize_runs.py`：匯總多個 `tune_summary_*.csv` 到 `runs/summary_all.csv`（便於統計、對照）。
  - `scripts/organize_runs.py`：重組 `runs/` 目錄結構，將已命名規則的實驗搬移到結構化目錄（支援 `--apply` 實際搬移）。
  - `scripts/cleanup_runs.ps1`：以 PowerShell 清理/搬移 runs（Windows 友善版）。
  - `scripts/time_validation.ps1`：以 PowerShell 在 validation 上批量測時（產生 `time_select_seconds.txt`、`metrics.csv`）。

Configs（設定檔）
- 常用模板：
  - `configs/features_basic.yaml`：單段（以 tokens 控制），TF‑IDF 表示 + Greedy 預設，含候選池設定。
  - `configs/features_3sent.yaml`：三句摘要（以 sentences 嚴格控制），可搭配 `greedy|grasp|nsga2`。
  - `configs/features_20sent.yaml`：Stage1 Base Top‑K 的模板（以 sentences 控制 20 句），可用 `greedy|grasp|nsga2`。
  - `configs/features_bert_20sent.yaml`：Stage1 BERT Top‑K 模板（`representations.use=false`, `candidates.use=false`）。
  - `configs/features_fast_3sent.yaml`：Stage2 fast（三句；TF‑IDF semantic + MMR），對應 `optimizer: fast`。

- 產生式模板（自動輸出）：
  - `configs/_generated/**`：由 `scripts/tune_union_fusion.py` 或 `scripts/grid_stage2_fast.py` 產生，例如：
    - `base_k{K1}.yaml`, `bert_k{K2}.yaml`（Stage1）
    - `stage2_bert.yaml`, `stage2_fused_w{wb}_a{alpha}.yaml`, `stage2_fast*_w{w}_a{a}.yaml`（Stage2）
  - 使用時機：希望保留本次調參所用精確設定，並在 `runs/` 內形成可追溯紀錄。

- 歷史/歸檔模板：
  - `configs/_archive/**`：歷史模板版本；如需手動 Stage2（bert/fused）模板，可從此處複製精簡版到 `configs/`。

相依與注意事項
- 代表性相依：
  - `bert|fused` 與 SBERT 需 `torch`, `transformers`（SBERT 另需 `sentence-transformers`）。
  - `nsga2` 與 `fast_nsga2` 需 `pymoo==0.6.1.1`；若 `representations.use=false` 無相似度，NSGA‑II 會回退 greedy。
  - `recall_target` 需 `rouge-score` 才能計算 Oracle recall。
- 模板路徑一致性：
  - `scripts/tune_union_fusion.py` 預設引用 `configs/features_bert_3sent.yaml` 與 `configs/features_fused_3sent.yaml`。
  - 若這兩個檔案不在 `configs/` 根目錄（而位於 `_archive/`），請先複製到 `configs/` 或修改腳本的預設路徑；否則腳本會報找不到模板。

典型使用流程（示意）
- Stage1（K1/K2）：
  - Base：`python -m src.pipeline.select_sentences --config configs/features_20sent.yaml --split validation --input data/processed/validation.jsonl --run_dir runs --optimizer greedy --stamp stage1-greedy-k20`
  - BERT：`python -m src.pipeline.select_sentences --config configs/features_bert_20sent.yaml --split validation --input data/processed/validation.jsonl --run_dir runs --optimizer bert --stamp stage1-bert-k20`
- 聯集 U + Stage2：
  - `python scripts/build_union_stage2.py --input data/processed/validation.jsonl --base_pred runs/stage1-greedy-k20/predictions.jsonl --bert_pred runs/stage1-bert-k20/predictions.jsonl --out data/processed/validation.stage2.union.jsonl --cap 25`
  - `python -m src.pipeline.select_sentences --config configs/features_fast_3sent.yaml --split validation --input data/processed/validation.stage2.union.jsonl --run_dir runs --optimizer fast --stamp stage2-fast-top3`
- 一鍵調參：
  - `python scripts/tune_union_fusion.py --input data/processed/validation.jsonl --run_dir runs --k1 20 --k2 20 --cap 25 --methods fast fast_grasp fast_nsga2 bert fused --optimizer1 greedy`
  - 產生 `_generated/` YAML 與 `runs/tune_summary_*.csv`，便於對照與回溯。

清理與維護建議
- 保持 `configs/` 僅放「可直接引用」的模板；產生式 YAML 落在 `_generated/`，歷史版本放 `_archive/`。
- 若調參腳本預期的模板缺失（如 `features_bert_3sent.yaml`、`features_fused_3sent.yaml`），請先從 `_archive/` 複製精簡版到 `configs/`，或修改腳本預設路徑以提供回退。
- `scripts/organize_runs.py` 可將 `runs/` 目錄規整到 `runs/structured/`，便於查找；`scripts/summarize_runs.py` 可將 `tune_summary_*.csv` 匯總。
- Windows 使用者可優先使用 PowerShell 腳本（`*.ps1`）做批次清理與測試。

