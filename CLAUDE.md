# CLAUDE.md — AI 協作規則

> 這個檔案會被 Claude Code 自動讀取。其他 agent（GPT 等）請在開始工作前先讀這份。
> 目的：多個 AI 同時工作時不要互相破壞、不要重複推導、不要重犯已知錯誤。

---

## 0. 這是什麼專案

抽取式摘要（extractive summarization）研究。
論文《Combining Meta-Heuristic Optimization, Graph Centrality, and PLM Semantics for High-Quality Extractive Summarization》。

- **ICACT**：已投稿，獲 outstanding paper award
- **ICT Express**：已被拒（ICTE-D-26-00238），四位審稿人
- **現在目標**：修正後改投 **IEEE Access**

這個檔案位於研究 repo 根目錄；程式碼在 `src/`。

---

## 1. 文件編輯規則

**所有 .md 檔案都可以編輯。** 這些文件最終會濃縮成單一版本，使用者會持續對齊。

編輯時請遵守：

- **刪除既有內容前先確認它是否已被取代** —— 不同文件的結論可能來自不同的分析，
  不要因為與自己的判斷不同就移除；先在原處標註分歧與理由
- **修改結論性數字時，附上你是怎麼得到的**（跑了什麼、資料範圍多大）
- **保留 checkbox 的勾選狀態**，那是實際進度

---

## 2. 證據登錄（引用前核對適用範圍）

以下是目前證據快照。若 evaluator、資料、artifact 或程式版本改變，必須重新驗證；不得用「不要重新推導」阻止更正。證據與方法見 `docs/research/CODE_AUDIT_IEEE_Access.md`。

### ⛔ 全域警告：所有既有的 ROUGE-Lsum 數字都已過期（2026-07-30，PR #9）

`src/eval/rouge.py` 的分句器已從手寫 regex `(?<=[.!?。！？])\s+` 換成
`src/data/sentence_split.py` 的共用 Punkt tokenizer。舊 regex 在每個縮寫句點後都切一刀
（`"Mr. Smith met U.S. officials on Tuesday."` 會被切成三句），是真的 bug。

**這份文件裡出現的每一個 ROUGE-Lsum 數字，都是在舊分句器下量到的，重算前不得再引用。**

| | 舊 regex | 新 Punkt | 差 |
|---|---|---|---|
| ROUGE-1 | 0.4316 | 0.4316 | +0.0000 |
| ROUGE-2 | 0.1444 | 0.1444 | +0.0000 |
| **ROUGE-Lsum** | 0.3893 | 0.3925 | **+0.0032** |

（800 篇真實 validation、Lead-style extract、245-word budget 實測。）

適用範圍：
- **R-1 與 R-2 完全不受影響**——它們不看句界，數值仍然有效。
- **只有 R-Lsum 位移**，量級約 +0.003。
- 資料側沒有變：`preprocess_multinews.py` 本來就用 Punkt、縮寫清單相同，
  **canonical fingerprint 與凍結的 data policy 不受影響**。

### 🔴 兩個致命問題（會決定論文能不能投）

1. **legacy Multi-News 當家配置沒有贏過本地 Lead**
   全 5622 篇、ID 對齊、同一 Google `rouge_score` evaluator：
   系統 `0.4352 / 0.1405 / 0.3880` vs Lead（245 whitespace words）`0.4331 / 0.1453 / 0.3901`
   → R-2 和 R-Lsum 輸。這證明舊稿的全面勝出主張不成立；因系統 config 是在 test 上選出，數字只作 legacy diagnostic。CNN/DM 的 0.351 與文獻 Lead-3 0.4042 來自不同 split/evaluator，不得稱為公平勝負。

   ⚠️ **分句器更換後的適用範圍**：R-1 的 `+0.0021` 與 **R-2 的 `−0.0048` 仍然成立**（不受句界影響）。
   R-Lsum 的兩個值（`0.3880` / `0.3901`）都是舊分句器下的量測，**其 `−0.0021` 的差距尚未在新分句器下重新驗證**。
   兩邊會一起往上位移，方向大機率不變，但「未變」目前是推測不是實測。
   **F-0 的核心結論不受影響**：R-2 落後這一項本身就足以否定「across every metric」。

2. **11 個 Multi-News tuning/ablation runs 使用 test set 選設定**
   `runs/tuning_experiments/` 底下 11 個 run **全部是 5622 篇 = test set**，
   論文再從中挑 best config → test-set overfitting。
   **受此選模流程影響的 legacy 數字不可用於新論文。**

### 病因診斷（為什麼輸給 Lead）

系統選句位置中位數 0.143、前 25% 佔 67.6%；legacy greedy reference 是 0.462 / 31.3%。
系統選句 **61.7% 與 Lead 重疊、只有 22.8% 命中** 當時建立的 greedy reference。
根因：`position` 特徵是內建 lead prior，且候選池（`sources: [score]`）由此單獨決定，
三軌都只在這個已偏前的池子裡運作。

**重現方式**（腳本已版本化）：

```bash
python -m scripts.audit.selection_diagnostics \
  --data data/processed/multi_news_test.jsonl \
  --pred runs/tuning_experiments/ExpB_K20_Max_Coverage/predictions.jsonl \
  --budget 245 --limit 200
```

⚠️ 仍是 diagnostic：跑在 test-tuned legacy artifact 上、200 篇抽樣、
greedy reference 不是 official oracle、未做 paired test。新 validation pipeline 必須重做。

### 其他已驗證事實

| 事實 | 數值 |
|---|---|
| 多句資料的新內部協定採 **ROUGE-Lsum** | Full benchmark 0.2014 → **0.3857**；ExpB 0.2019 → **0.3880**（皆為 legacy diagnostic）。⛔ **兩個 Lsum 值都是舊 regex 分句器下量的，見本節開頭的全域警告，重算前不得引用** |
| 論文的 SciTLDR "oracle" 0.136 | 是資料集 `rouge_scores` 欄位的全句平均，**不是 oracle** |
| legacy greedy references（非官方、非 exact upper bound） | SciTLDR 3句 0.5136；Multi-News 245 words 約 0.59；不得直接引用 |
| SciTLDR **官方**協定 | 單句、files2rouge、以最大 R1 選定同一 reference；oracle 52.4 / PACSUM 28.7 / BERTSumExt 36.2 |
| PLM 計時 diagnostic | 載入時間遠大於推論（兩次量到 78% 與 93%，**佔比不穩定、不可引用特定數字**）；穩定的是純推論 BERT/RoBERTa 比值 **≈1.0**（量到 1.04× 與 1.02×）。腳本 `scripts/audit/plm_timing.py`，數字須依鎖定 runtime protocol 重測 |
| Stage 2 的 `w_bert` | 加權的是 **TF-IDF**，不是 BERT。Stage 2 完全沒有 PLM |
| CNN/DM 既有 run | 13368 筆 = **validation**，不是官方 test（11490） |
| pymoo `BitflipMutation()` | `prob=1.0` 是 per-individual；per-gene 為 `1/n_var` |
| Headroom exploratory diagnostic | Multi-News 0.152 / CNN-DM 0.171 / SciTLDR 0.190；需以正式 protocol 重做 |

---

## 3. 程式碼硬規則

### 評測

- ⚠️ 多句資料暫用 `src/eval/rouge.py` 的 ROUGE-Lsum；內部 hand-calculated golden tests 已完成，正式 protocol 仍需與文獻 evaluator 對齊
- 多句摘要的新內部協定使用 `rougeLsum`；`rouge_scores_legacy()` 只保留重現舊 artifact。SciTLDR 必須另用其官方 files2rouge／ROUGE-L 協定，不能套用這條多句規則。
- ⚠️ pred 與 ref **必須用同一個函式分句**。原始換行是雜訊不是句界
  （370/500 predictions 含雜訊換行，references 一個都沒有；不一致會低估 ~0.023）
- 🚫 **不要再寫任何新的分句 regex。** 句界的唯一權威是 `src/data/sentence_split.py`
  的共用 Punkt tokenizer，資料側（`preprocess_multinews.py`）與評測側（`eval/rouge.py`）
  都必須走它。「一句」在兩邊必須是同一個意思。
  ⚠️ `src/data/preprocess.py:13` 還留著第三個 regex（`\s*`，比舊的更兇），
  目前只餵 `backend/`（展示用）與 CNN/DM legacy 前處理。**P0-02 若要復活 CNN/DM，
  必須先把它換成共用 tokenizer**，否則會重蹈 F-2。
- Punkt 不處理 CJK `。！？`（舊 regex 有）。三個資料集都是英文，目前無影響；
  若日後加入 CJK 資料集，必須先擴充 `sentence_split.py`。

### 已套用並通過 regression tests

以下 patch 已接線。`pytest` 已加入依賴；2026-08-02 的 master（PR #10 合併後）
目前 **217 tests 全過**。這是 correctness checkpoint，不是方法效果證據；
SciTLDR 官方 conformance 尚未通過；它只在決定保留 optional stress test 時才是必要驗收，不阻塞 GovReport + Multi-News 主線：

| 檔案 | 修正 |
|---|---|
| `src/eval/rouge.py` | ROUGE-Lsum、同一 reference 由最高 R1 選定、長度 mismatch fail；仍需 official files2rouge conformance |
| `src/features/graph.py` | dense thresholding 不再竄改呼叫端；candidate route 預設改為有界 sparse TF-IDF kNN，整體 selector 仍待 sparse 化 |
| `src/models/extractive/encoder_rank.py` | 完整輸入 batch encode、模型快取、pinned revision、截斷率與 deterministic cost facts；CPU 與 3-row pipeline smoke 已過，正式 cold/warm/GPU cost pilot 待做 |
| `src/pipeline/optimizer_dispatch.py` | `pop_size`/`n_gen`/`seed` 接線；**移除靜默 fallback** |
| `src/objectives/factory.py`、`evaluator.py` | single-sentence 關閉 subset search；Greedy／GRASP／NSGA-II 共用同一 objective 與 feasibility contract，並保存 Pareto front。**raw-sum 禁令只涵蓋「有 `task_profile` 且 `output_mode=multi_sentence`」的路徑**（`factory.py:87-92`）；沒有 `task_profile` 的 legacy_unprofiled 路徑（`factory.py:44-57`）預設仍是 `sum`，見 `docs/research/CODE_AUDIT_IEEE_Access.md` F-14。group coverage 與最終 Pareto policy 尚待完成 |
| `src/pipeline/candidate_builder.py` | `route_top_k` 只定義 proposals、`min_per_route` 保留 route-specific evidence、RRF 只在 union/guard 內填 total cap；短文件保存 requested/effective reservation 與 shortfall，不再中止整批 run |
| `src/pipeline/select_sentences.py` | MVP selector 明確使用 normalized RRF salience；`validation_pilot_only` 在 runtime 禁止 test split；正式輸出前依 frozen data policy 核對 canonical fingerprint、file/manifest SHA、row/revision/U+FFFD counts 並保存 `dataset_preflight.json`；`min_words` 只在完整來源本身不可達時以 exact capacity 逐列調降，candidate-induced infeasibility 仍 fail loud，超過 active output budget 的句子不得消耗 candidate quota |
| `src/data/policy.py` | frozen data policy 的執行點：核對 policy SHA-256、row count、dataset revision、content fingerprint、file/manifest SHA 與 U+FFFD 計數；debug subset 一律拒絕。實測三種竄改（改 policy 檔、換資料檔、analysis 錯配）都會 fail loud |
| `src/eval/oracle.py` | 新增 greedy oracle reference；不是 exact upper bound。SciTLDR official oracle 僅在保留 optional stress test 時實作／重現 |
| `src/baselines/contract.py`、`lead.py`、`cli.py` | PR #10 已加入 governed Lead baseline 與三種 ordering，共用資料 preflight／長度上限並保存 output-budget provenance；兩個 primary 尚未跑完，TextRank／LexRank／PacSum／sentence-encoder／Random 仍未進 master，Gate 2 未通過 |

- 🚫 **絕對不要再加 `except Exception: fallback to greedy`**。研究模式必須 fail loud
- 🚫 不要在 config 加了鍵值卻沒接線（`pop_size` 就發生過，實際跑的一直是預設 100/100）

### 資料

- 🚫 **不要用 test set 調參**。validation 調參 → freeze config → test 只跑一次
- CNN/DM 官方 test 是 **11490**；13368 是 validation
- SciTLDR 的 `target` 是**多個替代 reference**，`preprocess_scitldr.py` 已改為 canonical `references: list[str]`，不可再串接
- Multi-News 正式資料只能由 `src.data.preprocess_multinews` 產生；固定 `alexfabbri/multi_news` revision，保留 `|||||` boundary，舊 flat JSONL 僅供 legacy reproduction

---

## 4. 工作規則

1. **驗證再宣稱**。這個專案已經因為「數字沒對過」被拒一次。
   要說某個數字，就跑出來；不能跑就標明是推測或引用。
2. **跨論文 baseline 數字不能當公平本地勝負**。論文 Table 7 的 Lead/LexRank/TextRank 是 "adopted from [16]"，
   與本文的預處理、分句、ROUGE 設定都不一致 —— 這正是問題的來源。
   所有主 baseline 必須在本地同一條 pipeline 上重跑；可重用官方或可信實作，不必為了「自己寫」再造輪子。
3. **`runs/` 底下的既有數字一律視為 invalid**（test-set 調參污染），
   不要拿來寫進論文，也不要當作 regression baseline。
4. 改完程式碼要說明**改了什麼、為什麼、怎麼驗證的**。
5. 不確定的地方**問，不要猜**。

---

## 5. 常用指令

按目前多句內部協定重算某個 run（不代表已對齊所有 published evaluator）：

```bash
python -m src.pipeline.evaluate --pred runs/<run>/predictions.jsonl --gold data/processed/<dataset>_<split>.jsonl --out runs/<run>/metrics_fixed.csv --protocol multisentence_lsum
```

算 greedy oracle reference：

```bash
python -m src.eval.oracle --input data/processed/multi_news_test.jsonl --max_words 245 --limit 300
```

跑選句：

```bash
python -m src.pipeline.select_sentences --config configs/<cfg>.yaml --split test --input data/processed/<data>.jsonl --run_dir runs
```

跑測試：

```bash
python -m pytest -q
```

協作時優先使用 feature branch + pull request；GitHub Actions 的 `Unit tests` 必須通過後再合併。若直接 push `master`，CI 只能在 push 後發現問題，無法在沒有 branch protection 的情況下事前阻擋。

跑稽核診斷（見 `scripts/audit/README.md`）：

```bash
python -m scripts.audit.lead_vs_system --data <data.jsonl> --pred <run>/predictions.jsonl --budget 245
```

---

## 6. 文件導覽

研究文件全部在 `docs/research/`：

| 檔案 | 內容 |
|---|---|
| `README.md`（repo 根目錄） | 程式碼總覽、安裝、狀態聲明 |
| `docs/research/INDEX.md` | 研究文件總索引 + 關鍵數字速查 ← 先看這個 |
| `docs/research/ACTION_PLAN.md` | **要做什麼、什麼順序** ← 日常執行看這份 |
| `docs/research/ARCHITECTURE.md` | Target Architecture v1、schema、objective 啟用矩陣、freeze gate；validation pilot 前尚未凍結 |
| `docs/research/paper_revision_plan_IEEE_Access.md` | 研究流程治理、投稿合規 |
| `docs/research/CODE_AUDIT_IEEE_Access.md` | 已驗證的程式缺陷 + 實測數字 |
| `docs/research/STRATEGY_ASSESSMENT.md` | 可行性評估、資料集選擇 |
| `docs/research/REPO_CLEANUP.md` | 專案整理計畫 |

> 重構前的 legacy 文件已移至 `docs/_legacy_docs/`（排除於版本庫外，只在原作者本機）。
> 它們引用的 `configs/stage1/`、`scripts/build_union_stage2.py` 等路徑多數已不存在，
> 且 `RUNS.md`／`TIMING_AND_OFFLINE.md` 仍把 metric 寫成 `rougeL`。**不要當現行規格。**

---

## 7. 不重要的資料夾（分析時可略過）

`frontend/`（展示用 web UI）、`backend/`（展示用 API）、`experimental/`（抽象式摘要與 rerank 探索）
—— 三者都與論文無關，分析時可完全略過。

`configs/_legacy_archive/`、`runs/_archive_2026_01_15/`、`scripts/_archive/`、
`data/processed/_archive_legacy/` 是歷史檔案，**已從版本庫排除，只存在原作者本機**。
協作者 clone 後不會看到它們，也不需要。

⚠️ `scripts/quick_tune*.ps1` 與 `run_missing_experiments.ps1` 是造成 P0-01 的腳本
（直接在 test set 上調參）。已加上 guard，不帶 `-IAcknowledgeTestSetTuning` 會拒絕執行。
**不要用它們產生任何新結果。**

> 投稿稿件、審稿意見（`Reviewer.docx`）與原始資料**不在這個 repo**，留在上一層本機資料夾。
