# metaheuristic-summarization — 協作狀態與路線圖

本文檔彙整目前專案的完成進度、缺口與改進方案、優先順序與驗收標準，作為協作依據。

---

## 一、完成項目（Completed)

- 前處理（Prepare/P1）
  - 檔案：`src/data/preprocess.py`
  - 功能：正則分句（中英標點）、空白正規化、最小詞數過濾（句級）、可限制句數；支援隨機抽樣/限量導出。
  - 輸入/輸出：CSV（欄位 `id, article, highlights`）→ JSONL（欄位 `id, sentences[], highlights`）。

- 特徵與基線打分（Features & Base Scoring）
  - 檔案：`src/features/{tf_isf.py,length.py,position.py,compose.py}`
  - 功能：TF‑ISF、長度、句位特徵；`combine_scores` 做線性加權並 max-normalize。
  - 狀態：可用；目前特徵權重多處於程式內硬編（如 `select_sentences.build_base_scores`）。

- 句向量與相似度（Represent/R1,R2）
  - 檔案：`src/representations/{sent_vectors.py, similarity.py}`
  - 功能：TF‑IDF（預設）或 SBERT 產生句向量；計算 cosine 相似度矩陣（支援稀疏/稠密）。
  - 用途：冗餘抑制（MMR 類型）與 NSGA‑II 目標中的覆蓋/冗餘。

- 選句與長度約束（Select/R5、Length Control）
  - 檔案：`src/models/extractive/{greedy.py,grasp.py,nsga2.py}`、`src/selection/length_controller.py`
  - 功能：
    - Greedy：MMR 型效用，逐步加入，遵循 tokens 上限。
    - GRASP：RCL 隨機化建構 + swap/add/drop 局部搜尋。
    - NSGA‑II：重要性/覆蓋/冗餘多目標，最後以 lambda 標量化選代表解。
    - 長度控制：以 tokens 粗粒度計數。

- Pipeline 與輸出（Score & Output/D2,D3）
  - 檔案：`src/pipeline/select_sentences.py`
  - 功能：讀取設定/資料，計算分數與相似度，呼叫選句器，輸出 `runs/<stamp>/predictions.jsonl` 與 `config_used.json`。

- 監督式打分（Supervised，替代基線分數）
  - 檔案：`src/models/extractive/supervised.py`, `src/pipeline/train_supervised.py`
  - 功能：以輕量監督模型產生句級分數，推論時可作為 `base_scores` 來源。

- 評估（Score & Output/D4）
  - 檔案：`src/pipeline/evaluate.py`, `src/eval/rouge.py`
  - 功能：讀取 `predictions.jsonl`，計算 ROUGE，輸出 `metrics.csv`。

- 腳本與測試
  - 檔案：`scripts/{benchmark_small.py,split_dataset.py,jsonl_to_csv.py}`；`tests/{test_representations.py,test_selection.py}`（煙霧測試）。

---

## 二、已知缺口與改進方案（Gaps & Proposals）

優先級標註：Critical > High > Medium > Low

1) 候選池（shortlist top‑k）未真正生效【Critical】
- 現象：`src/pipeline/select_sentences.py` 中計算了 `cand_idx = topk_by_score(...)`，但後續只建立 `_ = [...]` 未限制選句器的可選集合。
- 影響：`candidates.k` 與 `candidates.use` 設定無效，導致搜尋空間與效率、效果不一致。
- 建議修法（外層子集法，最小侵入）：
  - 以 `cand_idx` 對 `sentences`/`base_scores`/`sim` 取子集；呼叫選句器得到子集索引；最後映回原始索引並排序。
- 驗收標準：
  - 設定 `candidates.use: true, k: 5` 時，實際只從 top‑5 中選句；`k` 改變會反映在輸出。
  - 新增單測驗證：當 `k=1` 時，若長度可容納，唯一候選必被選中。

2) `representations.use`/`cache` 未被尊重【High】
- 現象：即使 `use: false` 或 `redundancy.lambda=1.0`（忽略冗餘），依然建立向量與相似度。
- 建議：
  - 若 `representations.use == false` 或選句器不需相似度，跳過 `SentenceVectors` 與相似度計算；演算法側容忍 `sim=None`（或零矩陣）。
  - 可在 `runs/<stamp>/cache/` 實作按文件層級的 `TF‑IDF`/`sim.npy` 緩存（`cache: true`）。
- 驗收標準：
  - `use: false` 時執行時間降低且無向量/相似度 I/O；流程仍可跑完（greedy 可退化為僅重要性）。

3) 中文長度與分詞策略【High】
- 問題：目前 tokens 以空白切分，中文句子常被視為 1 token，長度控制失真。
- 建議：
  - 增加 `length_control.unit: tokens|sentences|chars`；
  - 提供 `tokenizer: whitespace|jieba|charbigram` 選項（可選安裝 `jieba`）。
- 驗收標準：
  - 在 `chars` 或 `jieba` 下，中文長度限制能準確生效；行為一致可測。

4) 覆蓋度（coverage）在非 NSGA‑II 中的支援【Medium】
- 問題：greedy/grasp 只做冗餘抑制，未考量「覆蓋未選句」效益。
- 建議：
  - 引入 coverage 近似項（例如對未選句的 max‑sim 平均提升）融合到效用函數，或提供 `optimizer.method: greedy_coverage`。
- 驗收標準：
  - 在高冗餘文檔上，`greedy_coverage` 較純 MMR 有穩定增益（以小樣本基準驗證）。

5) 二階段打分（句級平均融合 → 候選摘要 rerank）【High】
- 問題：僅做句級分數平均融合難以全局最佳化。
- 建議：
  - 產出多個候選摘要（多次 GRASP/不同超參），以 cross‑encoder（例：`cross-encoder/ms-marco-MiniLM-L-6-v2`）對「摘要級」打分，選最優。
  - 依賴：`torch`, `transformers`, `sentence-transformers`（cross‑encoder 模式）。
- 驗收標準：
  - 在 validation 上 ROUGE 有顯著提升，且可關閉還原。

6) 強化學習（RL）原型【Medium】
- 目標：替代監督式模型，對「逐步選句」學習動態策略。
- 設計要點：狀態（已選集合特徵、剩餘長度、相似度摘要）、動作（候選索引+STOP）、回饋（步進 ROUGE 增益 + 終局 ROUGE）、非法動作屏蔽。
- 首版：REINFORCE+baseline；訓練期用 reference 計 reward；推論期僅策略。
- 驗收標準：
  - 在小樣本上可訓練收斂且不劣於 greedy；提供可重現訓練腳本與固定隨機種子。

7) 效能與可重現性【Medium】
- 建議：
  - 記錄 wall‑time（pipeline `select_sentences.py`、`evaluate.py`）；
  - `config_used.json` 已輸出，補上環境版本（Python、關鍵套件）；
  - 實作簡單快取（見 2）。
- 驗收標準：
  - `metrics.csv` 增列 `time_select_seconds,time_eval_seconds`；`runs/<stamp>/env.txt` 記錄版本。

8) 特徵加權外部化與正規化策略【Low】
- 問題：部分權重硬編；normalize 方式單一（max‑normalize）。
- 建議：
  - 權重移至 config；增加 `normalize: max|zscore|rank` 選項。
- 驗收標準：
  - 不改動演算法的情況下能通過既有測試；配置切換可觀察到行為變化。

9) 測試覆蓋與消融腳本【Low】
- 建議：
  - 新增單測：候選池生效、`representations.use=false` 路徑、中文長度控制；
  - 擴展 `scripts/benchmark_small.py` 支援多配置對比與時間彙總。

---

## 三、里程碑與優先序（Milestones)

- M0（Critical 修補，~0.5–1 天）
  - [ ] 候選池真正生效（含單測）
  - [ ] 尊重 `representations.use`；允許 `sim=None` 路徑
  - [ ] `metrics.csv` 增列時間統計；輸出 `runs/<stamp>/env.txt`

- M1（二階段 reranker，~1–2 天）
  - [ ] 產生多候選摘要（多次 GRASP/不同超參）
  - [ ] 整合 cross‑encoder rerank；可開關

- M2（中文長度/分詞，~0.5–1 天）
  - [ ] `length_control.unit` 擴展；`tokenizer` 選項

- M3（Bandit 自調參，~1–2 天）
  - [ ] 以 LinUCB/Thompson 在文件級自調 `alpha/k` 等；紀錄探索/利用

- M4（RL 原型，~3–5 天）
  - [ ] 建立環境與 REINFORCE baseline；實驗腳本

- M5（快取與設定外部化，~0.5–1 天）
  - [ ] 表示/相似度快取；特徵權重/正規化由 config 控制

- M6（測試與基準，~0.5–1 天）
  - [ ] 單元測試補強；基準腳本輸出時間/表格化彙總

---

## 四、設計與實作備忘（Implementation Notes)

- 候選池子集法（範例流程）
  1. `cand_idx = topk_by_score(base_scores, k)`；
  2. `S_sent, S_score, S_sim = by_index(sentences, base_scores, sim, cand_idx)`；
  3. `picked_sub = greedy/grasp/nsga2(S_*, ...)`；
  4. `picked = sorted(cand_idx[i] for i in picked_sub)`。

- `representations.use` gating
  - 若 `use: false` 或 `alpha==1.0` 且所選演算法不使用 sim：跳過 `SentenceVectors/Similarity`；傳 `sim=None`。
  - 選句器需容忍 `sim is None` 時，把冗餘項視為 0。

- 時間與環境紀錄
  - `time.perf_counter()` 包住主流程；
  - 輸出 `env.txt`：Python 版本、作業系統、`pip freeze | rg 'scikit|sklearn|torch|transformers|sentence'`（可選）。

- 中文長度控制
  - `length_control.unit: chars`：以 `len(s)` 或 `len(s.encode('utf-8'))`；
  - `tokenizer: jieba`（可選安裝）：僅在 tokens 模式下作用。

- 二階段 reranker I/O
  - 輸入：候選摘要文本與（可選）特徵；輸出：單一分數；
  - 落地：`src/models/extractive/rerank.py` + `src/pipeline/rerank.py`（或整合於 `select_sentences.py`）。

- RL 原型接口
  - `env.reset(doc)` → 狀態；`env.step(action)` → (狀態, reward, done, info)；策略網路在 `torch`。

---

## 五、映射到原流程圖（Flowchart Mapping)

- Prepare
  - P0: 範圍/資料集選擇 → `configs/dataset_*.yaml`, `scripts/split_dataset.py`
  - P1: 清理與分句 → `src/data/preprocess.py`
  - P2: ADM #1（丟棄低價值 tokens）→ 句級過濾已做；token 級待做（本文件 2-3）

- Represent
  - R1: 句向量 → `src/representations/sent_vectors.py`
  - R2: 相似度矩陣 → `src/representations/similarity.py`
  - R5: 選句演算法 → `src/models/extractive/{greedy,grasp,nsga2}.py`

- Select
  - R3: 冗餘/覆蓋 → MMR/冗餘已做；覆蓋在 NSGA‑II；attention 無
  - R4: shortlist top‑k → 設計存在，需修正使之生效（本文件 2-1）

- Score & Output
  - D1: 二階段打分（BERT/RoBERTa/XLNet；ADM #2）→ 待做（本文件 2-5, 2-6）
  - D2: 分數融合與長度控制 → 已做（`features/compose.py`, `length_controller.py`）
  - D3: 匯出摘要 → 已做（`predictions.jsonl`）
  - D4: 評估（ROUGE/時間/ablation）→ ROUGE 已做；時間/ablation 待補（本文件 2-7, 2-9）

---

## 六、快速檢核（Runbook)

- 安裝與環境
  - Python 3.9+（建議 3.10）；建立 `.venv` 並安裝 `requirements.txt`。
  - 如需 SBERT/生成式或 reranker：安裝 `torch`, `transformers`, `sentence-transformers`。

- 基本三段式
  1) 前處理：`python -m src.data.preprocess --input data/raw/validation.csv --split validation --out data/processed/validation.jsonl --max_sentences 25`
  2) 選句：`python -m src.pipeline.select_sentences --config configs/features_basic.yaml --split validation --input data/processed/validation.jsonl --run_dir runs --optimizer greedy`
  3) 評估：`python -m src.pipeline.evaluate --pred runs/<stamp>/predictions.jsonl --out runs/<stamp>/metrics.csv`

- 常見參數
  - `optimizer.method: greedy|grasp|nsga2|supervised`
  - `redundancy.lambda`（MMR 調和）；`candidates.use/k`；`length_control.max_tokens`
  - `representations.method: tfidf|sbert`；`representations.use: true|false`

---

## 七、風險與開放問題（Risks & Open Questions)

- RL 訓練期依賴 reference 作為 reward，推論期缺 reference：是否可泛化？需要多樣資料與良好 reward shaping。
- Cross‑encoder reranker 計算成本：在長文/多候選下費時，是否需要先壓縮候選數？
- 中文分詞依賴：是否允許新增 `jieba` 作為可選依賴？在生產環境的安裝政策如何？
- Cache 與路徑：快取要不要共用於不同 run？是否需要加上雜湊鍵（config + split + 檔案大小/mtime）？

---

最後更新：由協作助手自動建立（請於提交 PR 前視情況更新）。

