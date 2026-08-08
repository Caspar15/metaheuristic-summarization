# Metaheuristic Extractive Summarization

## 2026-08-08 development freeze checkpoint

- Multi-News canonical validation 已在任何新 optimization score 前，以固定 seed 3407
  reference-blind 凍結為 dev 3,935／dev-test 1,686。proposed-method 與 baseline runner
  會先驗完整 frozen data policy，再依相同 manifest 過濾，並保存 partition provenance。
- dev 可反覆搜尋；每個候選配置只能看一次 dev-test。GovReport 已由作者官方 archive
  重建為 973 筆可評估 canonical validation rows，並在任何方法分數前凍結為
  dev 681／dev-test 292；唯一排除的是官方 reference 為空的 CRS `98-228`。
- freeze 簽字前禁止 test split；目前只推進到「可以跑 test」的狀態，屆時停下等待批准。

## 2026-08-06 selector-comparison checkpoint

- 已新增同候選、同 SBERT salience/similarity/coverage、同 budget 的
  `Greedy / candidate-matched SBERT-MMR / NSGA-II` selector interface；每列保存
  matched-input SHA-256，詳見
  [`docs/research/SELECTOR_COMPARISON_PROTOCOL.md`](docs/research/SELECTOR_COMPARISON_PROTOCOL.md)。
- 已修正 pinned `all-MiniLM-L6-v2` 的 SentenceTransformer 契約：mean pooling 後
  逐句 L2 normalization；新結果不可與舊 raw-centroid artifact 混稱。
- 已接上 full-source `sbert_centroid`、`sbert_mmr` baseline，但尚未完成 5,621-row
  Multi-News validation 正式 run。
- 真實 canonical 3-row correctness/cost smoke 已通過 matched hashes；不含 ROUGE、
  不可作論文品質結論。200-row reference-blind pilot manifest 已在看分數前凍結。
  加上兩 primary partition、greedy-reference correctness、GovReport data layer、A1 runner、
  D1 inventory、document-aware position、可恢復 runner 與 diagnostics regression 後，
  現行完整測試為 **386 passed**。
- frozen 200-row pilot 已完成：candidate-matched MMR 對 Greedy 的 R-1／R-2
  分別 `+0.01488`／`+0.01472` 且 Holm 校正後顯著；NSGA-II 單 seed 無顯著改善，
  總時間約為 Greedy `4.6×`。五 seed extension 的 NSGA-II mean 三指標均低於
  Greedy、selection Jaccard 僅 `0.639`；因此 selector 已決定採 MMR，NSGA-II
  降為 comparator。這是 diagnostic，不是 full-validation 結論。
- A1 兩資料集的 reference-only 統計與候選協定已在分數前預註冊；study runner 已
  版本化，會為每個 run 寫 evidence／search log 並拒絕重看 dev-test。兩個 primary 的
  唯一一次 A1 dev-test 都已完成並凍結；兩 primary non-PLM matrix 已完成，PLM、greedy
  reference 與正式方法 paired comparison 仍缺；test split 仍鎖定。
- A1 Multi-News 已依預註冊規則選定 200–250 words：dev-test cross-method macro
  `0.310382`，相對 max-only250／median220／p75-cap260 的 paired-bootstrap 95% CI
  全為正，三個 Holm-adjusted `p=0.000600`。這個勝負主要來自 Greedy stopping，不能
  誤寫成「200–250 最貼近所有 reference」；Greedy 仍有 3/1,686 篇未達 floor（F-23）。
- A1 GovReport 已選定 500–650 words：dev-test cross-method macro `0.393415`，對三個
  no-floor 候選的 Holm-adjusted `p=0.000600`，292/292 可行。floor-bound lexical
  Greedy macro `0.375048` 仍低於 Lead `0.394815` 與 Random `0.410382`（F-24）；這只
  診斷 cheap method，semantic/graph 尚未評估。
- D1 dev-only Greedy 敏感度研究已在任何新分數前預註冊：完整盤點 19 組／90 個
  runtime config paths，分 lexical/objective、cheap multiroute、semantic 三個 family。
  document-aware position correctness 已修復（F-25），governed family runner 已版本化。
  兩個 primary 的 lexical/objective 各 12/12、Multi-News cheap-multiroute 12/12 已完成。
  Multi-News lexical+graph G02 相對純 lexical base `+0.013952`，平均 pool 48.03、最大
  60，但 macro `0.324305` 仍低於 Lead `0.326291`；GovReport 全文 lexical 對 base
  `+0.045200`，dev point estimate 首次高於同協定 Lead／Random。這尚未對強 baseline
  或做 paired significance。GovReport cheap-multiroute 已有 11 success + 1 uncapped
  section-guard structural failure；cap-aware follow-up 已另行預註冊並完成（681/681
  feasible、pool max 60）。Multi-News semantic 原 family 有 2 success + 1 three-route
  capacity failure：S00 不勝 graph 且成本約 10.37×；預註冊 S02b follow-up 已完成，
  macro `0.328077` 首次高於 Lead 點估計，但它同時改 capacity/guard 且尚無 paired
  significance，不能歸因 semantic 或晉級。GovReport semantic 原 family 亦完成：S00
  macro `0.407203`，高 graph G02 但仍低 Random／全文 lexical 且成本約 graph G00
  `15.15×`；S01 跨資料集失敗，該 selector 接法刪除。GovReport S02b 隨後完成，macro
  `0.417862`，高全文 lexical L10、graph G07 與 Random，但成本約 graph G00 `18.28×`。
  兩 primary S02b 均為目前 proposed 最高點估計；仍缺 strong baseline／paired inference，
  本 screen 不看 dev-test。capacity-matched ablation 顯示 S02b 相對移除 semantic／graph：
  Multi-News `+0.003868/+0.005358`，GovReport `+0.014268/+0.011481`；兩路跨資料集
  point estimate 都正向。預註冊 paired analysis 的 12/12 route endpoints 亦全部通過
  Holm 與 186-opportunity correction，semantic/graph 暫留；但 Multi-News 對 Lead 的
  R-2 顯著低 `0.008336`，強 baseline 尚未齊，不能晉級。F-30 Greedy 等價效能修正與 F-31 的
  錯誤 runtime 外推更正見
  [`D1_SENSITIVITY_STATUS.md`](docs/research/D1_SENSITIVITY_STATUS.md)。

抽取式摘要研究程式碼。多目標最佳化（NSGA-II）、圖中心性與句向量語意訊號的組合，
目標是在 **zero-training（不做任務微調）** 的條件下研究 quality–cost trade-off。

---

## ⚠️ 專案狀態：重構中，既有結果不可引用

這個 repo 目前正在依 IEEE Access 投稿標準做**正確性重構**。開始使用前請先知道：

| 項目 | 狀態 |
|---|---|
| `runs/` 底下的既有結果 | 🔴 **無效** —— 超參數是在 test set 上選的（test-set overfitting） |
| Stage 2 的 `w_bert` 參數 | 🔴 **命名誤導** —— 它加權的是 TF-IDF 分數，不是 BERT。Stage 2 目前沒有 PLM |
| ROUGE-L | 🟠 舊碼用單序列 `rougeL`；已改為多句適用的 `rougeLsum` 並通過內部手算 golden，但與 published Perl ROUGE 的 parity 尚未驗證 |
| Baseline | 🟡 **Phase 2 進行中** —— 兩 primary 的 non-PLM 各 23/23 已完成：Multi-News P08 比 S02b 高 `0.003663`；GovReport LexRank 比 S02b 高 `0.033758`、TextRank高 `0.013104`。PLM、greedy reference、paired matrix 尚未完成，**Gate 2 未過**；見 `docs/research/GATE2_BASELINE_STATUS.md` |
| Gate 2 搜尋 | 🟡 `gate2-baseline-matrix-v1` 已在正式分數前預註冊：每資料集 non-PLM 23、PLM 27。兩 primary non-PLM 完成；Multi-News PLM 2/27。F-51 cache 已通過 3,935-row cold/warm exact audit，可續跑既有網格；runner 只讀 frozen dev，dev-test/test 皆未讀 |
| 三軌候選生成 | 🟠 correctness contract 已完成：完整輸入排名、route proposals/reservations、RRF selector salience、total cap 與 coverage guard；實際效益仍待 validation pilot |
| 測試 | ✅ **386 local tests passed**（2026-08-09）；PR #15 Linux CI 綠燈（2026-08-05），CI 維持 push／PR 自動執行 |

**簡言之：程式可以跑，但目前的輸出不能當研究結論。**

---

## 📚 研究文件（協作者從這裡開始）

全部在 [`docs/research/`](docs/research/)：

| 文件 | 用途 |
|---|---|
| [`INDEX.md`](docs/research/INDEX.md) | **總索引 + 關鍵數字速查** ← 先看這個 |
| [`ACTION_PLAN.md`](docs/research/ACTION_PLAN.md) | 要做什麼、什麼順序、完成定義 ← 日常執行看這份 |
| [`ARCHITECTURE.md`](docs/research/ARCHITECTURE.md) | Target Architecture v1、schema、模組介面、freeze gate |
| [`paper_revision_plan_IEEE_Access.md`](docs/research/paper_revision_plan_IEEE_Access.md) | 研究流程治理、10 個 P0、投稿合規 |
| [`CODE_AUDIT_IEEE_Access.md`](docs/research/CODE_AUDIT_IEEE_Access.md) | 已驗證的程式缺陷 + 實測數字 |
| [`GATE2_BASELINE_STATUS.md`](docs/research/GATE2_BASELINE_STATUS.md) | 最新 baseline family 分數、證據與未完成項目 |
| [`STRATEGY_ASSESSMENT.md`](docs/research/STRATEGY_ASSESSMENT.md) | 可行性評估、病因診斷、資料集選擇 |
| [`REPO_CLEANUP.md`](docs/research/REPO_CLEANUP.md) | 專案整理計畫 |

AI 協作規則見 repo 根目錄的 [`CLAUDE.md`](CLAUDE.md)。

> 重構前的 legacy 文件（`PIPELINE.md`、`RUNS.md`、`CONFIGS.md`、`PROJECT_STATUS.md` 等）
> 已移至 `docs/_legacy_docs/` 並排除於版本庫外 —— 它們引用的路徑多數已不存在。

---

## 先看哪裡 / 可以先略過哪裡

```
src/          ← 研究主線，看這裡
scripts/audit/← 稽核診斷腳本（Lead 比較、選句位置分析、headroom、PLM 計時）
tests/        ← 單元測試
configs/      ← 實驗設定（歷史檔 _legacy_archive/ 已排除於版本庫外）

frontend/     ← 🚫 展示用 web UI，與論文無關，可以完全略過
backend/      ← 🚫 展示用 API server，與論文無關，可以完全略過
experimental/ ← 🚫 抽象式摘要與 rerank 的探索，與本論文主線無關
notebooks/    ← 🚫 空的
```

> **給協作者**：只需要看 `src/`、`scripts/audit/`、`tests/`、`configs/`。
> `frontend/`、`backend/`、`experimental/` 不用讀，它們不影響任何研究結果。

---

## 安裝

Python 3.10+（開發環境為 3.12）

```bash
python -m venv .venv
# Windows
.\.venv\Scripts\activate
# Unix
source .venv/bin/activate

pip install -r requirements.txt
```

只想跑展示用的 web app 才需要：

```bash
pip install -r requirements-demo.txt
```

跑測試：

```bash
python -m pytest -q
```

GitHub Actions 會在每次 push 到 `master` 或針對 `master` 的 pull request 自動執行相同測試。這是 unit-test CI，不會下載完整資料、模型或執行正式 benchmark；研究結果仍須依 `ACTION_PLAN.md` 的 gate 另外驗收。

---

## 模組總覽

| 模組 | 說明 |
|---|---|
| `src/data/` | 前處理（分句、濾短句、CSV/HF → JSONL） |
| `src/features/` | TF-ISF、句長、句位置、TextRank 中心性 |
| `src/representations/` | TF-IDF 向量與相似度矩陣 |
| `src/models/extractive/` | Greedy(MMR)、GRASP、NSGA-II、encoder 排序 |
| `src/pipeline/` | 特徵組合、候選池、optimizer dispatch、選句、評估 |
| `src/eval/` | ROUGE（Lsum + multi-reference）、metric-specific greedy reference |
| `src/selection/` | 長度控制與候選池工具 |

---

## 快速開始

Multi-News canonical 前處理（固定作者資料集 revision、保留 `|||||` 多文件邊界）：

```bash
python -m src.data.preprocess_multinews --split validation \
  --out data/processed/multi_news_validation_canonical.jsonl
python -m src.data.validate_dataset \
  --input data/processed/multi_news_validation_canonical.jsonl \
  --split validation --expected_rows 5621 \
  --expected_dataset_revision 1f20a01dbf6463236108a8d7fd39f3ae9750dcc3 \
  --report_out data/processed/multi_news_validation_health_strict.json
```

Pinned validation 原始 5,622 列中有 1 列是空來源，故 canonical 輸出固定為
5,621 列並另寫 exclusion manifest。strict validator 會因其中 72 列含 U+FFFD
而失敗；正式政策已凍結為「主分析保留 5,621 列且不修字，另報排除固定
72 列的 5,549-row clean sensitivity」。不得任意加
`--allow_replacement_character` 或看完分數再換 subset。另有 72 列全文不足 `min_words=200` —— 那是**不同的 72 列**，與 U+FFFD 批次交集為 0、聯集 144 列，兩者機制獨立。完整契約與統計見
`configs/data_policies/multinews_validation_v1.json`、
`docs/research/evidence/multinews_validation_health_summary.json`。

clone 後依 tracked policy/manifest 生成並驗證 ignored clean sensitivity：

```bash
python -m src.data.freeze_multinews_policy
```

預設命令不會改寫 tracked policy 或 72-row manifest；
`--initialize_policy` 只供「尚未看任何結果的新 policy version」使用。

Phase 1 的 requested `min_words=200` 會逐列依句子不可切割及 250-word
上限計算 exact source capacity；只有全文本身不可達時才產生較小的
`effective_min_words`，並把 requested/effective/capacity/reason 寫入 prediction。
完整 validation 有 72 列適用此規則。若全文可達但 hard candidate pool
不可達，run 會直接失敗；單句超過 active output budget 者亦不會占用
route top-K 或 candidate quota，而會留下 exclusion evidence。

Phase 1 Multi-News validation MVP（第一次會下載 pinned sentence encoder；不可先跑 test）：

```bash
python -m src.pipeline.select_sentences --config configs/phase1_mvp_multinews.yaml \
  --split validation \
  --input data/processed/multi_news_validation_canonical.jsonl \
  --run_dir runs --stamp phase1-mvp-multinews-validation
```

Phase 2 Lead baseline 的程式已進 master，但還沒有形成正式研究結果。正式 validation
輸出必須明確寫到 `runs_v2/`；`src.baselines.cli` 的預設 `--run_dir runs` 只保留相容性，
不要依賴預設值：

```bash
python -m src.baselines.cli --baseline lead \
  --config configs/phase1_mvp_multinews.yaml --split validation \
  --input data/processed/multi_news_validation_canonical.jsonl \
  --ordering document_order --run_dir runs_v2 \
  --stamp lead-multinews-validation
```

Lead 會共用資料 preflight 與長度上限，但不套用為 mean-salience 防退化而設的
`min_words=200` 下限；每列仍會記錄 requested floor、source capacity、實際字數與
不套用原因。這是 PR #10 / F-16 已測試的 baseline-specific contract，不代表 Gate 2
已通過。

回到 proposed pipeline；相同 frozen method 的 clean sensitivity run：

```bash
python -m src.pipeline.select_sentences --config configs/phase1_mvp_multinews.yaml \
  --data_policy_analysis clean_sensitivity --split validation \
  --input data/processed/multi_news_validation_clean_sensitivity.jsonl \
  --run_dir runs --stamp phase1-mvp-multinews-validation-clean
```

兩種 run 都會在建立正式輸出前核對 policy、dataset、manifest 與 exclusion
manifest 的 SHA-256，並將通過的身份寫入 `dataset_preflight.json`。

評估（ROUGE-1/2/Lsum）：

```bash
python -m src.pipeline.evaluate --pred runs/<run>/predictions.jsonl --gold data/processed/<dataset>_<split>.jsonl --out runs/<run>/metrics.csv --protocol multisentence_lsum
```

F-17 後 primary evaluation 預設計分**全部輸入列**，並在 `metrics.csv` 另報
feasible／infeasible 數；不能讓不同方法各自排除失敗列後直接比較。
`--feasible-only` 僅是單一 run 的診斷。跨方法 common-feasible sensitivity
必須使用 `scripts/audit/paired_run_intersection.py`，並明確指定 protocol；legacy
缺列 artifact 只能在 `--assume-legacy-feasible` 下作 diagnostic，不能升格正式結果。

Metric-specific greedy reference smoke（**不是** exact upper bound；正式 Gate 2 必須再由
frozen dev/dev-test manifest 過濾）：

```bash
python -m src.eval.oracle --input tests/fixtures/multi_news_validation_diagnostic_sample.jsonl --max_words 220 --limit 3
```

---

## 稽核診斷腳本

`scripts/audit/` 底下的腳本用來檢查系統行為，不是產生論文結果：

| 腳本 | 用途 |
|---|---|
| `lead_vs_system.py` | 在同資料、同 evaluator 下比較某個 run 與本地 Lead baseline |
| `selection_diagnostics.py` | 選句位置分布、與 Lead 的重疊率、對 greedy reference 的命中率 |
| `dataset_headroom.py` | 各資料集在 Lead 之上還有多少空間、lead bias 強度 |
| `plm_timing.py` | 拆解 PLM 成本為模型載入 vs 推論 |

用法與已重現的輸出見 `scripts/audit/README.md`。

---

## 已知的實作限制

投稿前必須處理，詳見團隊內部重構計畫：

- 舊的 flat `multi_news_*.jsonl` 已遺失 source-document boundary，只能重現 legacy artifact；正式實驗必須使用 `*_canonical.jsonl`
- `src/data/preprocess_scitldr.py` 已保留 SciTLDR 多個替代 reference；`scitldr_official` 評估在官方 wrapper 通過一致性測試前會拒絕執行
- `src/features/semantic.py` 的 `centrality` 與 `novelty` 數學上完全反相關，同時加權是退化的
- graph candidate route 已使用有界 sparse kNN，但 selector／coverage objective 目前仍可能建立 dense `N×N` similarity；完成 sparse selector/objective 後才能宣稱整條長文件 pipeline 都是 sparse
- canonical task-profile objective 已禁止 raw-sum salience；Greedy／GRASP／NSGA-II 已共用 objective 與 feasibility contract。legacy config 仍保留歷史 sum 行為，因此舊 run 依然有被長度上限支配的問題
- NSGA-II 已保存完整可行 Pareto front，但目前 final selection 仍是 provisional weighted sum；knee／reference-point policy 必須只用 validation 凍結
- `src/features/graph.py` 的 `compute_textrank_scores`（自製 PageRank power iteration）缺少已知答案的單元測試（星形圖、路徑圖、完全圖、不連通分量）；開啟 graph 候選路線之前必做，目前尚未排程

---

## 授權

尚未指定。在加入 LICENSE 之前，預設保留所有權利。

### 第三方依賴授權

投稿 IEEE Access 公開程式碼時會被檢視，先列在這裡：

| 套件 | 授權 | 用途 |
|---|---|---|
| [sumy](https://github.com/miso-belica/sumy) `==0.12.0` | Apache-2.0 | TextRank／LexRank baseline（`src/baselines/centrality.py`），見 `docs/research/COMPUTE_ENVIRONMENT.md` |
| [NLTK](https://github.com/nltk/nltk) `==3.10.0` | Apache-2.0 | sumy runtime 與專案 tokenizer code；repository **不含** NLTK data package |
