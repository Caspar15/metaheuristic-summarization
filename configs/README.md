# configs — 實驗設定

> ⚠️ 除 `phase1_mvp_multinews.yaml` 外，**其餘都是 legacy 設定。** 它們產生的結果全部是在 test set 上調參得到的（見
> `docs/research/CODE_AUDIT_IEEE_Access.md` 的 P0-01），**不可用於新論文**。
> 保留它們只為了重現 `runs/` 底下的既有 artifact。
> 新架構以 `phase1_mvp_*.yaml` 命名，且只允許 validation pilot；schema 以 `docs/research/ARCHITECTURE.md` 為準。

歷史設定（`_legacy_archive/`）已排除於版本庫外，只存在原作者本機。

---

## Phase 1 validation MVP

| 檔案 | 用途 | 尚未完成 |
|---|---|---|
| `phase1_mvp_multinews.yaml` | canonical Multi-News 上隔離 lexical + pinned sentence encoder；top-K proposals、route reservations、RRF selector salience、total cap 與 document guard | provenance-aware 3-row smoke 已過；尚缺正式 cost、baseline reality check、route unique recall、budget freeze，未過 gate 前不可跑 test |
| `phase1_mvp_multinews_length_normalized.yaml` | F-18 validation-only diagnostic；與上一列只有 `importance_aggregation: mean → length_normalized` 一項實驗差異 | 不得當正式 frozen config 或 test config；結果須綁定 F-18 evidence manifest |

此設定刻意不含 graph 與 NSGA-II：先確認 lexical + semantic MVP 能否勝過同協定 Lead／PacSum，再決定是否擴張完整架構。

### Phase 2 baseline 共用設定

PR #10 的 `src.baselines.cli` 會讀同一份 `phase1_mvp_multinews.yaml`，因此 Lead
與 proposed run 共用 dataset identity、frozen policy、split guard 與輸出上限；它不是
另一套偷偷放寬的 config。Lead 因為是 prefix baseline、沒有 mean-salience objective，
不套用 provisional `min_words=200` 下限，但 artifact 必須逐列保存 requested floor、
source capacity、selected words 與 reason。正式 run 請明確指定 `--run_dir runs_v2`；
CLI 目前的 `runs` 預設值不可當作新舊結果分界。

截至 2026-08-02，master 只包含 Lead。Random 的 PR #11 尚未合併；其他 baseline
仍是 Phase 2 待辦。

---

## 兩階段流程對照

Stage 1 三軌各跑一次 `select_sentences.py`，輸出各自的 `predictions.jsonl`；
再用 `scripts/utils_fusion.py` 取聯集，Stage 2 在聯集上做最終選句。

| 檔案 | 階段 | optimizer | 說明 |
|---|---|---|---|
| `1_Base_NSGA2.yaml` | Stage 1a | `nsga2` | 統計特徵（TF-ISF 0.8 / 句長 0.2 / 句位 0.2），候選 K=40 |
| `1_LLM_BERT.yaml` | Stage 1b | `bert` | `bert-base-uncased` 重心相似度排序，候選 K=40 |
| `1_Graph_TextRank.yaml` | Stage 1c | `greedy` | 純圖中心性（graph 權重 1.0），候選來源 `[graph]` |
| `2_Fusion_Final.yaml` | Stage 2 | `fast_nsga2` | 當家融合設定 |

### Stage 2 的權重掃描

四個設定只差在 objective 權重與融合權重，其餘相同（`max_tokens: 245`、
`max_sentences: 30`、`pop_size: 100`、`n_gen: 100`、`seed: 2024`）：

| 檔案 | λ_coverage | λ_redundancy | w_base | w_bert |
|---|---|---|---|---|
| `2_Fusion_Final.yaml` | 2.5 | 1.2 | 0.5 | 0.5 |
| `2_Fusion_ExpA.yaml` | 2.5 | 1.2 | 0.3 | **0.7** |
| `2_Fusion_ExpB.yaml` | **4.0** | 1.2 | 0.5 | 0.5 |
| `2_Fusion_ExpC.yaml` | 2.5 | **2.0** | 0.5 | 0.5 |
| `2_Fusion_NoNsga2.yaml` | 2.5 | 1.2 | 0.5 | 0.5 | ← optimizer 改用 `fast_fused`（MMR），NSGA-II ablation |

> 🔴 **`w_bert` 的命名是錯的。** 在 `src/models/extractive/fast_fused.py` 裡，
> 它加權的是 **TF-IDF centroid 相似度**，不是 BERT 分數。Stage 2 整條路徑沒有任何 PLM。
> 這是「移除 BERT 只掉 0.0005 ROUGE-1」的真正原因（F-3）。
> 程式已接受 `w_plm` 作為別名，但語義問題要等重構才解決。

---

## 消融／測試設定

| 檔案 | 變動的單一因素 |
|---|---|
| `test_v2_features.yaml` | 啟用 v2 特徵（TF-ISF 加停用詞與 sublinear TF、position 改 inverse、fusion 加交互項），並開 centrality/novelty |
| `test_v2_tfisf_only.yaml` | 只把 TF-ISF 換成 v2，其餘維持 v1 |
| `test_coverage_set.yaml` | NSGA-II 的 coverage 改用 `set`（貪婪子模） |
| `test_coverage_diversity.yaml` | NSGA-II 的 coverage 改用 `diversity`（coverage 減冗餘懲罰） |

> ⚠️ `centrality` 與 `novelty` 在數學上完全反相關（`centrality_norm = 1 - novelty_norm`），
> 同時給獨立權重是退化的 —— `test_v2_features.yaml` 的那兩個權重實際只有一個自由度。

---

## 使用方式

```bash
python -m src.pipeline.select_sentences \
  --config configs/1_Base_NSGA2.yaml \
  --split validation \
  --input data/processed/<split>.jsonl \
  --run_dir runs \
  --stamp my-run
```

`--optimizer` 可覆寫 config 裡的 `optimizer.method`。

---

## 設定鍵值說明

| 鍵 | 說明 |
|---|---|
| `experiment.status / dataset` | `validation_pilot_only` 只允許 canonical validation rows，且 dataset identity 必須一致；目前其他 status 一律 fail closed |
| `data_policy.policy_path / policy_sha256 / analysis` | 指向並鎖定 versioned frozen policy，再選擇 `main` 或 `clean_sensitivity`；runner 會在建立 run 前核對 policy SHA、row count、revision、content fingerprint、file/manifest SHA 與 U+FFFD 數量 |
| `objectives.lambda_*` | 從 Pareto front 選解時的加權（importance / coverage / redundancy） |
| `objectives.coverage_method` | `max`（預設）/ `set` / `diversity` |
| `features.weights.*` | 各特徵在 base score 的權重 |
| `features.{tf_isf,position,fusion}.version` | `v1`（預設）或 `v2` |
| `graph_params.threshold` | 圖的邊剪枝閾值 τ |
| `candidates.k` / `sources` / `mode` | 候選池大小、來源（`score`/`position`/`centrality`/`graph`）、`hard`/`soft` |
| `candidate_budget.route_top_k` | 每 route 保存與分析的 proposal depth；不是最終池保證配額 |
| `candidate_budget.min_per_route` | total cap 前優先保留的 route evidence；全域值不得大於 configured `route_top_k`，短文件逐列 clamp 並在 artifact 保存 requested/effective/shortfall |
| `candidate_budget.total` | selector pool 的上限，不從 proposal union／coverage guard 外補句；union 小於上限時允許誠實 underfill |
| `selector.salience_source` | `rrf_fusion` 會把 normalized provenance fusion 實際送入 selector；`membership_only` 僅供消融 |
| `compute_budget.mode / enabled_routes` | 目前只實作 validation-frozen `fixed`；宣告 `adaptive` 會直接失敗 |
| `length_control.min_words` | requested lower bound；逐文件依完整來源在 `max_words/max_sentences` 下的精確可達上限產生 `effective_min_words`。只有 source-intrinsic shortfall 可調降；candidate pool 若達不到 effective bound 必須報錯。兩者與 relaxation reason 均寫入 artifact |
| `routes.semantic.*` | sentence encoder 名稱、固定 revision、batch size、`max_model_tokens`；輸出記錄實際 revision 與截斷率 |
| `routes.graph.*` | 新 graph route 預設 `sparse_knn`；`dense_legacy` 必須明確指定 |
| `coverage_guard.*` | document／section／position strata 保留規則，不屬於語意 route |
| `length_control.unit` | `words`、legacy `tokens`（實為**空白切詞**）或 `sentences`；word budget 不再繞過 selector |
| `optimizer.{method,pop_size,n_gen}` | 選句器與 NSGA-II 參數 |
| `seed` | 全域亂數種子 |

> ⚠️ **`unit: tokens` 其實數的是空白切詞，不是 model token。**
> 新架構會把 `max_words` / `max_sentences` / `max_model_tokens` 拆成不同欄位
> （見 `docs/research/ARCHITECTURE.md` §1.2）。
>
> ℹ️ `pop_size` / `n_gen` / `seed` 過去**未被程式讀取**（實際一律跑 100/100），
> 已在 `optimizer_dispatch.py` 修好。因此 legacy run 的 config_used.json
> 不代表當時真正生效的參數。
