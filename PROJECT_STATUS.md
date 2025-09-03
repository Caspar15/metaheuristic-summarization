# metaheuristic-summarization — 協作狀態與路線圖

更新時間：2025‑09‑03

**重要更新**
- 已移除所有監督式（supervised）相關程式與腳本。
- 候選池（top‑k）已真正生效：支援 hard/soft 模式與多來源（score/position/centrality）聯合集合；可選 `recall_target` 自適應擴大 k。
- `representations.use` 已生效：為 false 時跳過向量/相似度；NSGA‑II 在無相似度時自動退回 Greedy；Greedy/GRASP 可在 `sim=None` 下運行。
- 新增以「句數」為單位的長度控制：`length_control.unit: sentences` 與 `max_sentences`（例如只輸出 3 句）。
- 新增二階段 rerank 的檔案預留與設定區塊（見下）。

---

## 一、完成項目（Completed）

- 前處理（Prepare/P1）
  - 檔案：`src/data/preprocess.py`
  - 功能：中英標點分句、空白正規化、最小詞數過濾、可限制每篇句數；支援抽樣/限量導出。
  - I/O：CSV（`id, article, highlights`）→ JSONL（`id, sentences[], highlights`）。

- 特徵與基線打分（Features & Base Scoring）
  - 檔案：`src/features/{tf_isf.py,length.py,position.py,compose.py}`
  - 功能：TF‑ISF、長度、句位特徵；線性加權並 [0,1] normalize。

- 句向量與相似度（Representations）
  - 檔案：`src/representations/{sent_vectors.py, similarity.py}`
  - 功能：TF‑IDF 或 SBERT 產生句向量；cosine 相似度矩陣。

- 選句與長度約束（Selection & Length Control）
  - 檔案：`src/models/extractive/{greedy.py,grasp.py,nsga2.py}`、`src/selection/length_controller.py`
  - 功能：Greedy（MMR）/GRASP（RCL+局部搜尋）/NSGA‑II（多目標）；長度上限可用 tokens 或 sentences 控制（已支援嚴格三句）。

- 候選池（Shortlist top‑k）
  - 檔案：`src/pipeline/select_sentences.py`
  - 功能：`mode: hard|soft`；`sources: [score|position|centrality]` 聯合集合；可選 `recall_target` 以 oracle 召回自適應擴大 k。
  - hard：僅在候選內搜索（含索引映射）；soft：不封鎖候選外句子，僅對候選分數乘法增益（`soft_boost`）。

- 表示與相似度開關
  - 檔案：`src/pipeline/select_sentences.py`、`src/models/extractive/{greedy,grasp}.py`
  - 功能：`representations.use=false` 不建立向量/相似度；Greedy/GRASP 忽略冗餘項；NSGA‑II 無相似度時退回 Greedy。

- Pipeline 與輸出（Score & Output）
  - 檔案：`src/pipeline/select_sentences.py`
  - 功能：讀設定與資料，計分/相似度/選句，輸出 `runs/<stamp>/predictions.jsonl` 與 `config_used.json`。

- 評估（Evaluation）
  - 檔案：`src/pipeline/evaluate.py`, `src/eval/rouge.py`
  - 功能：讀取 `predictions.jsonl`，計算 ROUGE，輸出 `metrics.csv`。

- 腳本與工具（Scripts）
  - 檔案：`scripts/{benchmark_small.py,split_dataset.py,jsonl_to_csv.py,run_7_2_1.py}`
  - 功能：資料切分/轉換，整合跑流程，小樣本基準。

---

## 一之二、流程藍圖對應（Flow Mapping）

- A Prepare
  - P0 Choose scope/datasets → `scripts/split_dataset.py`、原始 CSV 準備
  - P1 Clean & split text → `src/data/preprocess.py`
  - P2 ADM #1 drop low‑value tokens → 目前以句級最小長度替代；token 級刪除待擴展（見 Gaps 1）

- B Represent
  - R1 Make sentence vectors → `src/representations/sent_vectors.py`
  - R2 Build similarity matrix → `src/representations/similarity.py`
  - R5 My selection algorithm → 建議歸入 Select（屬選句決策）

- C Select
  - R3 Sentence attention / redundancy & coverage → 以相似度支援的冗餘抑制（MMR）與 NSGA‑II 覆蓋/冗餘
  - R4 Shortlist top‑k sentences → 已生效（子集化與/或軟性增益）
  - R6 Length control by sentences → `length_control.unit: sentences` 可直接以句數上限控制輸出
  - 建議：若重視效能，可將 R4 提前在向量化之前先做（僅對候選建向量/相似度）；若要接 LLM rerank，建議 soft 模式與較大 k，以確保候選召回。

- D Score & Output
  - D1 2nd‑stage scoring（BERT/RoBERTa/XLNet cross‑encoder）→ 檔案預留（見下 Gaps 3）
  - D2 Fuse scores & control length → `features/compose.py`、`selection/length_controller.py`
  - D3 Output extractive summary → `runs/<stamp>/predictions.jsonl`
  - D4 Evaluate（ROUGE；time；ablations）→ ROUGE 已有；時間/ablation 待補（見 Gaps 4）

---

## 二、已知缺口與改進方案（Gaps & Proposals）

1) 中文長度與分詞策略【High】（部分完成）
- 問題：`tokens` 以空白切分，中文句子常被視為 1 token。
- 進度：已支援 `length_control.unit: sentences` 與 `max_sentences`；`chars` 與 `tokenizer` 仍待擴展。
- 建議：新增 `length_control.unit: tokens|sentences|chars`；`tokenizer: whitespace|jieba|charbigram`（可選安裝 `jieba`）。
- 驗收：在 `chars/jieba` 下，中文長度限制能準確生效；行為一致可測。

2) `representations.cache` 快取【Medium】（規劃中）
- 建議：於 `runs/<stamp>/cache/` 實作 TF‑IDF 與 sim 的文件級快取，避免重複計算。
- 驗收：重跑同一 split 耗時下降且命中快取。

3) 二階段打分（候選摘要 rerank）【High】（規劃中）
- 建議：產出多個候選摘要（多次 GRASP/不同超參數）；以 cross‑encoder（例：`cross-encoder/ms-marco-MiniLM-L-6-v2`）對摘要級打分，選最優。
- 驗收：在 validation 上 ROUGE 有穩定提升，可開關還原。
- 檔案預留：`src/models/rerank/cross_encoder.py`、`src/pipeline/rerank.py`、`configs/features_basic.yaml` 的 `rerank` 區塊。

4) 效能與可重現性【Medium】（規劃中）
- 建議：記錄 wall‑time（select/evaluate）；輸出環境版本（Python、關鍵套件）；在 hard 模式下僅對候選建 sim；實作表示/相似度快取（見 2）。
- 驗收：`metrics.csv` 增列 `time_select_seconds,time_eval_seconds`；`runs/<stamp>/env.txt` 記錄版本；hard 模式大幅降低計算成本。

5) 特徵加權外部化與正規化策略【Low】（規劃中）
- 建議：將加權全部外部化到 config；支援 `normalize: max|zscore|rank`。
- 驗收：不改演算法的情況下測試通過；切換配置可觀察行為變化。

6) 測試覆蓋與消融腳本【Low】（規劃中）
- 建議：新增單測（候選池生效、`representations.use=false` 路徑、中文長度控制）；擴展基準輸出時間/彙總。

7) 強化學習（RL）前置搜尋【Low｜Optional】（規劃中）
- 目標：在進入 D1 之前，以 RL 產生高質量候選摘要集合，提高 rerank 上限。
- 設計：狀態（已選集合、剩餘長度、摘要向量、候選相似度摘要）、動作（選句/STOP；非法動作遮罩）、回饋（步進 ROUGE 增益 + 終局 ROUGE）。
- 驗收：小樣本上 RL 產生之候選集合在經 rerank 後的 ROUGE 有穩定增益；提供固定隨機種子與可重現腳本。此項為可選，僅在現有方法達到效能瓶頸時再投入。

---

## 三、里程碑與優先序（Milestones）

- M0（基礎落地，完成）
  - [x] 候選池真正生效（hard/soft、union、recall_target）
  - [x] 尊重 `representations.use`；允許 `sim=None`；NSGA‑II 缺相似度時退回 Greedy
  - [ ] `metrics.csv` 增列時間統計；輸出 `runs/<stamp>/env.txt`
  - [x] 以句數為單位的長度控制（`unit: sentences`, `max_sentences`）

- M1（二階段 reranker，~1–2 天）
  - [ ] 產生多候選摘要（多次 GRASP/不同超參數）
  - [ ] 整合 cross‑encoder rerank；可開關
  - [ ] 在 validation 上搜尋融合權重（或用小型 meta‑learner）

- M2（中文長度/分詞，~0.5–1 天）
  - [x] `length_control.unit: sentences`（已完成）
  - [ ] `length_control.unit: chars` 與 `tokenizer` 選項

- M3（效能與可重現性，~0.5–1 天）
  - [ ] 計時與環境紀錄；hard 模式下僅對候選建 sim；表示/相似度快取

- M4（RL 前置搜尋，~2–4 天，可選）
  - [ ] 建立環境/狀態/動作與 REINFORCE baseline；
  - [ ] 產生多候選摘要並與 D1 串接；
  - [ ] 小樣本實驗與消融（無 RL vs RL+rerank）。

- M5（測試與基準，~0.5–1 天）
  - [ ] 單元測試補強；基準腳本輸出時間/表格化彙總

---

## 四、快速檢核（Runbook）

- 安裝與環境
  - Python 3.10 以上
  - `pip install -r requirements.txt`

- 前處理（例：validation）
  - `python -m src.data.preprocess --input data/raw/cnn_dailymail/validation.csv --split validation --out data/processed/validation.jsonl --max_sentences 25`

- 選句與產生摘要
  - `python -m src.pipeline.select_sentences --config configs/features_basic.yaml --split validation --input data/processed/validation.jsonl --run_dir runs --optimizer greedy`
  - 三句摘要：將 `--config` 換為 `configs/features_3sent.yaml`，或在 config 中設定 `length_control.unit: sentences` 與 `max_sentences: 3`。

- 評估 ROUGE
  - `python -m src.pipeline.evaluate --pred runs/<stamp>/predictions.jsonl --out runs/<stamp>/metrics.csv`

---

## 五、近期結果（參考）

- CNN/DailyMail validation（全量，前處理保留每篇最多 25 句候選；三句摘要）：
  - Greedy：R1=0.3174, R2=0.1132, RL=0.1900
  - GRASP：R1=0.3175, R2=0.1131, RL=0.1900
  - NSGA‑II：R1=0.3209, R2=0.1162, RL=0.1928（需 `pymoo`）

註：不同抽樣、候選池與超參數會影響結果；上列數值僅供行為驗證與相對比較。
