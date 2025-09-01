# metaheuristic-summarization — 協作狀態與路線圖

更新時間：2025‑09‑01

重要更新
- 移除所有監督式（supervised）相關程式與腳本。
- 候選池（top‑k）已真正生效：僅在候選集合內選句，並正確映回原始索引。
- `representations.use` 已生效：為 false 時跳過向量/相似度；NSGA‑II 在無相似度時自動退回 Greedy；Greedy/GRASP 可在 `sim=None` 下運行。

## 一、完成項目（Completed)

- 前處理（Prepare/P1）
  - 檔案：`src/data/preprocess.py`
  - 功能：正則分句（中英標點）、空白正規化、最小詞數過濾、可限制句數；支援抽樣/限量導出。
  - I/O：CSV（`id, article, highlights`）→ JSONL（`id, sentences[], highlights`）。

- 特徵與基線打分（Features & Base Scoring）
  - 檔案：`src/features/{tf_isf.py,length.py,position.py,compose.py}`
  - 功能：TF‑ISF、長度、句位特徵；線性加權並 [0,1] normalize。

- 句向量與相似度（Representations）
  - 檔案：`src/representations/{sent_vectors.py, similarity.py}`
  - 功能：TF‑IDF 或 SBERT 產生句向量；cosine 相似度矩陣。

- 選句與長度約束（Selection & Length Control）
  - 檔案：`src/models/extractive/{greedy.py,grasp.py,nsga2.py}`、`src/selection/length_controller.py`
  - 功能：
    - Greedy：MMR 型效用 + tokens 上限。
    - GRASP：RCL 建構 + swap/add/drop 局部搜尋。
    - NSGA‑II：多目標（重要性/覆蓋/冗餘），以 `lambda_*` 標量化挑代表解。

- 候選池（Shortlist top‑k）
  - 檔案：`src/pipeline/select_sentences.py`
  - 功能：以 `top‑k` 分數子集化 `sentences/scores/sim`，選句後映回原索引並排序。

- 表示與相似度開關
  - 檔案：`src/pipeline/select_sentences.py`、`src/models/extractive/{greedy,grasp}.py`
  - 功能：`representations.use=false` 時不建向量/相似度；Greedy/GRASP 忽略冗餘項；NSGA‑II 無相似度時退回 Greedy。

- Pipeline 與輸出（Score & Output）
  - 檔案：`src/pipeline/select_sentences.py`
  - 功能：讀設定/資料，計分/相似度/選句，輸出 `runs/<stamp>/predictions.jsonl` 與 `config_used.json`。

- 評估
  - 檔案：`src/pipeline/evaluate.py`, `src/eval/rouge.py`
  - 功能：讀 `predictions.jsonl`，計 ROUGE，寫 `metrics.csv`。

- 腳本
  - 檔案：`scripts/{benchmark_small.py,split_dataset.py,jsonl_to_csv.py,run_7_2_1.py}`
  - 功能：資料切分/轉換，整合跑流程，小樣本基準。

## 一之二、流程藍圖對應（Flow Mapping）

- A Prepare
  - P0 Choose scope/datasets → `scripts/split_dataset.py`、原始 CSV 準備
  - P1 Clean & split text → `src/data/preprocess.py`（分句、正規化）
  - P2 ADM #1 drop low-value tokens → 句級過短過濾（已做）；token 級刪除待擴充（見 Gaps 1）

- B Represent
  - R1 Make sentence vectors → `src/representations/sent_vectors.py`
  - R2 Build similarity matrix → `src/representations/similarity.py`
  - R5 My selection algorithm → 建議移至 Select 區段（屬於選句決策）

- C Select
  - R3 Sentence attention / redundancy & coverage → 以相似度支援的冗餘抑制（MMR）與 NSGA‑II 覆蓋/冗餘
  - R4 Shortlist top‑k sentences → 已生效，於 `src/pipeline/select_sentences.py` 套用子集化
  - 建議：若追求效能，可將 R4 提前於 R1/R2，先以基線分數做 top‑k，再僅對候選做向量與相似度（目前為先建全量 sim 再切子矩陣，可調整為候選再向量化）

- D Score & Output
  - D1 2nd‑stage scoring (BERT/RoBERTa/XLNet) → 尚未實作（見 Gaps 3）
  - D2 Fuse scores & control length → `features/compose.py`、`selection/length_controller.py`
  - D3 Output extractive summary → `runs/<stamp>/predictions.jsonl`
  - D4 Evaluate (ROUGE; time; ablations) → `src/pipeline/evaluate.py`（ROUGE 已有；時間/ablation 待補，見 Gaps 4）

## 二、已知缺口與改進方案（Gaps & Proposals）

優先級標註：Critical > High > Medium > Low

1) 中文長度與分詞策略【High】
- 問題：`tokens` 以空白切分，中文句子常被視為 1 token。
- 建議：`length_control.unit: tokens|sentences|chars`；`tokenizer: whitespace|jieba|charbigram`（可選 `jieba`）。
- 驗收：在 `chars/jieba` 下長度限制準確；行為一致可測。

2) `representations.cache` 快取【Medium】
- 建議：於 `runs/<stamp>/cache/` 實作 TF‑IDF 與 sim 的文件級快取。
- 驗收：重跑同一 split 時耗時明顯下降且命中快取。

3) 二階段打分（候選摘要 rerank）【High】
- 建議：產生多候選摘要（多次 GRASP/不同超參）；以 cross‑encoder（如 `cross-encoder/ms-marco-MiniLM-L-6-v2`）對摘要級打分，選最優。
- 驗收：validation ROUGE 穩定提升，可開關還原。

4) 效能與可重現性【Medium】
- 建議：記錄 wall‑time（select/evaluate）；輸出環境版本（Python/關鍵套件）。
- 驗收：`metrics.csv` 增列 `time_select_seconds,time_eval_seconds`；`runs/<stamp>/env.txt` 存版本。

5) 特徵加權外部化與正規化策略【Low】
- 建議：將加權全搬至 config；支援 `normalize: max|zscore|rank`。
- 驗收：不改演算法的情況下測試通過；切換配置可觀察行為變化。

6) 測試覆蓋與消融腳本【Low】
- 建議：新增單測（候選池生效、`representations.use=false` 路徑、中文長度控制）；擴展基準輸出時間與彙總。

7) 強化學習（RL）做前置搜尋【High】
- 目標：在進入 D1（二階段 rerank）前，以 RL 產生高質量候選摘要集合，提高 rerank 上限。
- 設計：
  - 狀態：已選句索引/特徵、剩餘長度、摘要向量（可由已選句平均）、與候選的相似度摘要；
  - 動作：選擇一個候選句或 STOP；非法動作（長度超限/重複）遮罩；
  - 回饋：步進 ROUGE 增益 + 終局 ROUGE；可加長度正則；
  - 演算法：REINFORCE+baseline（首版）、後續可試 PPO；
  - 輸出：多條策略 rollouts 所得候選摘要，用於 D1 cross‑encoder rerank。
- 驗收：小樣本上 RL 產生之候選集合在經 rerank 後的 ROUGE 顯著優於無 RL 的基線；提供固定隨機種子與可重現腳本。

## 三、里程碑與優先序（Milestones)

- M0（基礎落地，完成）
  - [x] 候選池真正生效（含索引映射）
  - [x] 尊重 `representations.use`；允許 `sim=None`；NSGA‑II 缺相似度時退回 Greedy
  - [ ] `metrics.csv` 增列時間統計；輸出 `runs/<stamp>/env.txt`

- M1（二階段 reranker，~1–2 天）
  - [ ] 產生多候選摘要（多次 GRASP/不同超參）
  - [ ] 整合 cross‑encoder rerank；可開關

- M2（中文長度/分詞，~0.5–1 天）
  - [ ] `length_control.unit` 擴展；`tokenizer` 選項

- M3（RL 前置搜尋，~2–4 天）
  - [ ] 建立環境/狀態/動作與 REINFORCE baseline；
  - [ ] 產生多候選摘要並與 D1 串接；
  - [ ] 小樣本實驗與消融（無 RL vs RL+rerank）。

- M4（快取與設定外部化，~0.5–1 天）
  - [ ] 表示/相似度快取；特徵權重/正規化由 config 控制

- M5（測試與基準，~0.5–1 天）
  - [ ] 單元測試補強；基準腳本輸出時間/表格化彙總

## 四、快速檢核（Runbook)

- 安裝與環境
  - Python 3.10 以上
  - `pip install -r requirements.txt`

- 前處理（例：validation）
  - `python -m src.data.preprocess --input data/raw/cnn_dailymail/validation.csv --split validation --out data/processed/validation.jsonl --max_sentences 25`

- 選句與產生摘要
  - `python -m src.pipeline.select_sentences --config configs/features_basic.yaml --split validation --input data/processed/validation.jsonl --run_dir runs --optimizer greedy`

- 評估 ROUGE
  - `python -m src.pipeline.evaluate --pred runs/<stamp>/predictions.jsonl --out runs/<stamp>/metrics.csv`
