# D1 Greedy 敏感度研究狀態

更新日期：2026-08-08（由系統時間取得）

## 邊界與目前進度

- 本研究只讀兩個 primary 的 frozen validation `dev` membership；沒有 CLI split
  參數，且尚未讀取 dev-test 或 test。
- 27 個配置與兩資料集的 delta 已在任何 D1 分數前凍結於
  `configs/preregistrations/d1_greedy_sensitivity_v1.json`。
- 目前完成兩個 primary 的 **lexical/objective family**、**cheap-multiroute family**，
  以及兩個 primary 的 semantic 原 family 與 capacity-correct follow-up；Multi-News
  每個完整 run 為 3,935 rows、GovReport 為 681 rows。capacity-matched route
  ablation 與事前註冊的 paired inference 已完成；強 baseline 尚未齊，因此沒有配置可晉級，
  也尚未做 D1 的一次性 dev-test。

| Primary | lexical/objective | cheap multiroute | semantic |
|---|---:|---:|---:|
| Multi-News | **12/12 complete** | **12/12 complete** | **2 success + 1 structural failure; capacity follow-up complete** |
| GovReport | **12/12 complete** | **11 success + 1 structural failure; cap-aware follow-up complete** | **2 success + 1 structural failure; capacity follow-up complete** |

## Multi-News lexical/objective 結果

同一 frozen 200–250-word contract、同一 Greedy selector、all-row macro
`mean(R-1,R-2,R-Lsum)`：

| ID | 單一變動 | Macro | Δ vs L00 | Feasible |
|---|---|---:|---:|---:|
| L10 | 不做 candidate prefilter（全文） | **0.320912** | **+0.010559** | 3,934/3,935 |
| L03 | coverage weight 0.8 → 1.6 | 0.316569 | +0.006216 | 3,931/3,935 |
| L09 | document-aware position weight 0 → 0.2 | 0.312921 | +0.002568 | 3,930/3,935 |
| L07 | TF-ISF unigram → unigram+bigram | 0.311928 | +0.001574 | 3,931/3,935 |
| L05 | 保留 stopwords | 0.311160 | +0.000807 | 3,931/3,935 |
| L02 | importance weight 1.0 → 0.5 | 0.311116 | +0.000763 | 3,931/3,935 |
| L11 | 開啟 position coverage guard | 0.311022 | +0.000669 | 3,931/3,935 |
| L00 | frozen base | 0.310353 | 0 | 3,931/3,935 |
| L08 | feature fusion v1 → v2 | 0.310353 | 0 | 3,931/3,935 |
| L06 | sublinear TF → linear TF | 0.310137 | −0.000216 | 3,931/3,935 |
| L04 | redundancy weight 0.7 → 1.4 | 0.305827 | −0.004526 | 3,931/3,935 |
| L01 | length-normalized importance → mean | 0.283376 | −0.026977 | 3,935/3,935 |

這只是 OFAT 優先序，不是 promotion 結論。後續已完成 31-config、兩資料集的
capacity-matched route ablation 與 paired bootstrap；完整結果與多重比較限制見本文件後段，
但 strong baseline matrix 尚未完成。

### 與便宜 baseline 的同協定脈絡

- A1 frozen-dev Lead macro：`0.326291`；Random：`0.307098`。
- L10 全文搜尋雖比 L00 與 Random 高，仍比 Lead 低 `0.005379`。它的 R-1/R-2/
  R-Lsum 是 `0.431657/0.135168/0.395911`；Lead 是
  `0.435033/0.148139/0.395701`。也就是 R-Lsum 僅微高，R-2 明顯不足。
- 因正式 greedy reference 尚未跑完，本階段不能計算 headroom capture；不得用
  「贏 Lead 幾分」取代 headroom 指標。

### 品質與成本一起解讀

- L00 selector pool 平均 `35.95`、p95 `40`、最大 `44`；run wall time 約
  `234.3 s`。
- L10 selector pool 平均 `81.85`、p95 `210.3`、最大 `3,318`；wall time 約
  `1,138.5 s`，是 L00 約 `4.86×`。時間包含 selection subprocess、評分與 evidence
  建立，僅能作同機器相對成本。
- 因此目前證據支持「top-40 lexical candidate prefilter 是品質瓶頸」，但不支持把
  全文 Greedy 當最終方法。下一輪應先看兩路 candidate budget、graph／semantic 是否
  能用遠小於全文的 pool 回收這個增益。

## Multi-News cheap-multiroute 結果

同一 frozen dev／length／evaluator／Greedy selector；G00 是 lexical+graph 兩路 base：

| ID | 單一變動 | Macro | Δ vs G00 | selector pool mean/max |
|---|---|---:|---:|---:|
| G02 | route top-K 40 → 80 | **0.324305** | **+0.000898** | 48.03 / 60 |
| G04 | total budget 60 → 80 | 0.324209 | +0.000802 | 48.09 / 80 |
| G09 | graph min similarity 0 → 0.10 | 0.323913 | +0.000506 | 46.24 / 60 |
| G03 | min per route 20 → 10 | 0.323728 | +0.000320 | 46.19 / 60 |
| G10 | graph alpha 0.85 → 0.90 | 0.323466 | +0.000059 | 46.17 / 60 |
| G11 | 關閉 Multi-News document guard | 0.323428 | +0.000021 | 46.18 / 60 |
| G00 | lexical + sparse graph | 0.323407 | 0 | 46.19 / 60 |
| G05 | RRF constant 60 → 30 | 0.323123 | −0.000284 | 46.19 / 60 |
| G08 | graph neighbors 8 → 16 | 0.322582 | −0.000825 | 45.78 / 60 |
| G01 | lexical + TF-IDF centroid | 0.322130 | −0.001278 | 45.38 / 60 |
| G07 | soft/full pool + membership-only | 0.320997 | −0.002410 | 81.85 / 3,318 |
| G06 | membership-only salience | 0.319384 | −0.004023 | 46.19 / 60 |

### 可支持與不可支持的結論

- G00 相對純 lexical L00 是 `+0.013054` macro；G02 相對 L00 是 `+0.013952`，且比
  全文 lexical L10 高 `+0.003393`。因此 Multi-News dev 已支持：**sparse graph route
  能用平均 48、最大 60 的受控 selector pool 回收並超過全文 lexical 的品質**。
- G02 三項是 `0.436696/0.137103/0.399115`；同協定 Lead 是
  `0.435033/0.148139/0.395701`。G02 的 R-1／R-Lsum 較高，但 R-2 低 `0.011036`，macro
  仍低 `0.001986`。沒有 paired inference，也未對 TextRank／LexRank／PacSum／
  SBERT+MMR／greedy reference，因此不能稱為勝出或進入 dev-test。
- G06 比 G00 低 `0.004023`，顯示 route-aware salience 不是可刪的包裝；只讓 graph
  決定 membership 會系統性低估它。G07 同時更慢、pool 最大 3,318 且品質更低，
  因此不保留 soft/full-pool + membership-only 組合。
- G02 與 G04 的差只有 `0.000096`；這個 OFAT screen 只能把 candidate budget 列為
  後續優先項，不能先把 80/60 或 40/80 組合成未預註冊的新配置後直接看 dev-test。

## Multi-News semantic 結果

原預註冊三案中 S00/S01 完成，S02 因三路 reservations 加 document guard 超過 cap
而 fail loud：

| ID | 設計 | Macro | R-1/R-2/R-Lsum | selection seconds |
|---|---|---:|---:|---:|
| S00 | lexical + semantic、RRF selector | **0.322404** | 0.435165 / 0.135871 / 0.396177 | 1,387.23 |
| S01 | 同候選，semantic raw + SBERT similarity selector | 0.299634 | 0.409085 / 0.117613 / 0.372204 | 1,394.26 |
| S02 | lexical + semantic + graph | **failed** | mandatory 61 > cap 60 | — |
| S02b | 三路、total 80、guard cap 20（獨立 follow-up） | **0.328077** | 0.440759 / 0.139803 / 0.403670 | 1,467.99 |

- S00 相對純 lexical L00 是 `+0.012051`，但低於 graph G00 `0.001003`、graph G02
  `0.001900`、Lead `0.003886`。semantic route 有平均 10.48 個 unique candidates、
  3.68 個 unique selected sentences，證明不是完全重複；但目前沒有 quality gain beyond
  graph，且 selection 約 graph G00 的 `10.37×`。S02b 已出現三路整體正訊號，但非
  semantic 純 ablation；仍須等 GovReport 與 paired evidence 才依 §5.3 決定保留、
  按需 routing 或刪除。
- S01 比 S00 低 `0.022771`，直接以 semantic raw salience/SBERT similarity 取代目前
  selector inputs 明顯失敗；這個 selector 接法不保留。
- S02 原 failure 保留。兩-primary S02b 已在 GovReport semantic 分數前獨立預註冊：
  total 80、guard max 20，由 `3 routes × 20 + 20 guards = 80` 推導，不用分數選值；
  Multi-News follow-up 已於 prereg commit `73769c5` 後完成。3,930/3,935 feasible、
  selector pool mean/max `52.09/80`，macro `0.328077`；比 graph G02 高 `0.003772`、
  比 Lead 高 `0.001786`，但 R-2 仍低 Lead `0.008336`。因 S02b 同時改 total 與 guard
  cap，這是三路**整體配置**的正訊號，不是 semantic 的純因果增益；沒有 paired
  significance、強 baseline 或 GovReport 複現前不得 promotion。

## 完整性與可重現性檢查

- L00 macro `0.3103531191842453` 與 selected-indices SHA-256
  `3b63acb201e9350227ad45b1a78bafbb655cb66744e45d780ea4d71642365d81`
  均與 A1 dev 完全一致。
- `runs_v2/search_log.jsonl` 保存 12 個 final success 加 1 個外層 60-minute job
  timeout failure；失敗的 L11 partial 已封存，重試成功沒有覆蓋失敗紀錄。
- diagnostics schema v2 分開保存 actual selector search-space 與 provenance pool。
  `candidates.use=false` 時 provenance size 是 0，但 actual selector size 是全文，不能
  再把兩者混稱。
- 主 evidence：`runs_v2/d1_greedy_sensitivity/multinews/dev/lexical_objective/`。
  所有 final evidence 均標示 `test_split_accessed=false`；study summary 另標示
  `dev_test_accessed=false`。
- cheap-multiroute 主 evidence：
  `runs_v2/d1_greedy_sensitivity/multinews/dev/cheap_multiroute/`。12 個 final runs
  均為 3,935 rows；search log 另保留 G00／G01 各一個 external interruption failure。
  family summary 為 `dev_test_accessed=false`、`test_split_accessed=false`。

## GovReport lexical/objective 結果

同一 frozen 500–650-word contract、同一 Greedy selector、all-row macro
`mean(R-1,R-2,R-Lsum)`：

| ID | 單一變動 | Macro | Δ vs L00 | Feasible |
|---|---|---:|---:|---:|
| L10 | 不做 candidate prefilter（全文） | **0.415585** | **+0.045200** | 681/681 |
| L07 | TF-ISF unigram → unigram+bigram | 0.383758 | +0.013373 | 681/681 |
| L03 | coverage weight 0.8 → 1.6 | 0.379900 | +0.009515 | 681/681 |
| L11 | 開啟 position coverage guard | 0.373002 | +0.002617 | 681/681 |
| L09 | document-aware position weight 0 → 0.2 | 0.373001 | +0.002616 | 681/681 |
| L06 | sublinear TF → linear TF | 0.372896 | +0.002510 | 681/681 |
| L05 | 保留 stopwords | 0.372819 | +0.002434 | 681/681 |
| L02 | importance weight 1.0 → 0.5 | 0.371373 | +0.000988 | 681/681 |
| L00 | frozen base | 0.370385 | 0 | 681/681 |
| L08 | feature fusion v1 → v2 | 0.370385 | 0 | 681/681 |
| L04 | redundancy weight 0.7 → 1.4 | 0.362779 | −0.007606 | 681/681 |
| L01 | length-normalized importance → mean | 0.332161 | −0.038224 | 681/681 |

### 與便宜 baseline 的同協定脈絡

- A1 frozen-dev Lead macro `0.399232`；Random `0.408844`。L10 的 macro 高
  `+0.016352`／`+0.006741`，三項為 `0.541583/0.190097/0.515075`。
- L10 對 Random 三項都較高；對 Lead 的 R-1／R-Lsum 較高，但 R-2 仍低
  `0.003935`。這只是 dev point estimate；12 個 lexical 比較尚未做 paired bootstrap。
  兩 primary non-PLM 與 Multi-News PLM 已完成，但 GovReport PLM 與 greedy reference 未完成，不能稱為
  significant win 或進入 dev-test。
- L10 selector pool mean/p95/max 是 `318.52/698/2,889`，summary 平均 `649.07`
  words。品質增益證實 top-40 candidate recall 是 GovReport 的主要瓶頸，但全文 dense
  搜尋的成本仍排除它作最終架構；graph／semantic route 必須用受控 pool 回收增益。

### F-30／F-31 執行與更正

- pre-F-30 L10 attempt 在 `639.47 CPU s` 後中止並完整封存；關閉 writer 後確認 partial
  已有 frozen ordered dev 的前 152 rows，不是先前依開啟中 size=0 誤判的兩列。
- 依這 152-row prefix 的句數平方占比，舊實作全量 proxy 約 `0.766 CPU h`；原
  `4.05 h` 推論已作廢。F-30 batched exact additions 完整 357 tests 通過。
- post-F-30 L10 selection 實測 `817.84 s`，不是舊版完整 run 的直接配對，因此只能說
  相對 prefix-calibrated projection 約 `3.37×`；不能寫成實測 speedup。
- 固定 post-F-30 reference-blind audit 已重跑全部 681 frozen-dev rows；逐篇 selected
  indices **0 差異**，pre/post digest 同為 `8273f162...d7982`。evidence 在
  `runs_v2/f30_greedy_equivalence/govreport_l00_post_f30/evidence.json`；沒有評估
  references，也沒有讀 dev-test/test。F-30 至此通過。
- GovReport search log 保存 12 final successes 加 1 個 L10 interruption failure；全部
  `dev_test_score=null`、`test_split_accessed=false`。

## GovReport cheap-multiroute 結果

原預註冊的 12 個配置已全部嘗試：11 success，G11 因 section guard mandatory
reservations 77 > total cap 60 正確 fail loud。

| ID | 單一變動 | Macro | Δ vs G00 | selector pool mean/max |
|---|---|---:|---:|---:|
| G07 | soft/full pool + membership-only | **0.414831** | **+0.011335** | 318.52 / 2,889 |
| G02 | route top-K 40 → 80 | 0.404282 | +0.000786 | 59.94 / 60 |
| G11b | cap-aware section guard（獨立 follow-up） | 0.404182 | +0.000685 | 59.82 / 60 |
| G03 | min per route 20 → 10 | 0.404078 | +0.000582 | 59.82 / 60 |
| G04 | total budget 60 → 80 | 0.403594 | +0.000098 | 73.17 / 80 |
| G00 | lexical + sparse graph | 0.403496 | 0 | 59.82 / 60 |
| G10 | graph alpha 0.85 → 0.90 | 0.403496 | −0.000001 | 59.82 / 60 |
| G09 | graph min similarity 0 → 0.10 | 0.403483 | −0.000013 | 59.81 / 60 |
| G05 | RRF constant 60 → 30 | 0.402661 | −0.000835 | 59.82 / 60 |
| G08 | graph neighbors 8 → 16 | 0.401121 | −0.002375 | 59.79 / 60 |
| G06 | membership-only salience | 0.400872 | −0.002624 | 59.82 / 60 |
| G01 | lexical + TF-IDF centroid | 0.400421 | −0.003075 | 59.78 / 60 |
| G11 | section guard（uncapped） | **failed** | — | first row mandatory 77 > cap 60 |

### 解讀與 follow-up 治理

- G00/G02 相對純 lexical L00 分別 `+0.033111/+0.033897`，graph 又比 TF-IDF
  centroid 第二路 G01 高 `0.003075`。graph 刪除條件尚未觸發，但 G02 hard pool
  macro 只高 Lead `0.005050`、仍低 Random `0.004562`，不能晉級。
- G07 高於 Lead／Random macro `+0.015599/+0.005987`，但比全文 lexical L10 仍低
  `0.000754`；selection `804.13 s`，是 G00 `60.53 s` 的約 `13.28×`。它沒有證明
  graph 能在全文池增量勝過 lexical，且成本不適合作 final architecture。
- G06 在兩 primary 都低於 G00（Multi-News `−0.004023`；GovReport `−0.002624`）。
  因此 membership-only 架構已得到跨資料集負證據，route-aware salience 必須保留。
- G11 failure 不得靠事後放大 total cap 覆寫。事前 inventory 已限定
  `coverage_guard.max_items` 只在 overflow 時啟用；因此另以系統時間預註冊
  `G11b_section_guard_cap20`。20 由 `60 − 2×20` 的最壞情況容量推導，不使用分數選值；
  原 G11 failure 永久保留。follow-up 已在 prereg commit `3b6813e` 後執行：681/681
  feasible、pool max 60、macro `0.404182`（對 G00 `+0.000685`）；只作 feasibility/
  diagnostic，不單獨 promotion。

## GovReport semantic 結果

原預註冊三案已全部嘗試：S00/S01 完成，S02 與 Multi-News 相同，因三路最低保留加
coverage guard 在第一列達 mandatory 61 > cap 60 而正確 fail loud。

| ID | 設計 | Macro | R-1/R-2/R-Lsum | selection seconds |
|---|---|---:|---:|---:|
| S00 | lexical + semantic、RRF selector | **0.407203** | 0.534512 / 0.187499 / 0.499599 | 917.18 |
| S01 | 同候選，semantic raw + SBERT similarity selector | 0.349832 | 0.474428 / 0.135957 / 0.439112 | 974.81 |
| S02 | lexical + semantic + graph | **failed** | first row mandatory 61 > cap 60 | — |
| S02b | 三路、total 80、guard cap 20（獨立 follow-up） | **0.417862** | 0.545091 / 0.196227 / 0.512267 | 1,106.27 |

- S00 相對 lexical L00 `+0.036818`、graph G00 `+0.003707`、graph G02 `+0.002921`，
  並高 Lead `0.007971`；但仍低 Random `0.001641`、全文 lexical L10 `0.008382`。
  semantic route 平均有 26.17 個 unique candidates、8.43 個 unique selected sentences，
  不是重複訊號；但 selection 約 graph G00 的 `15.15×`，目前 quality/cost trade-off
  不足以支持預設全開。
- S01 比 S00 低 `0.057371`；連同 Multi-News 的 `−0.022771`，semantic raw salience +
  SBERT similarity selector 已得到跨資料集一致負證據，依刪除條件不再保留。
- 原 S02 failure 永久保留。GovReport S02b 已在任何 GovReport semantic 分數前預註冊，
  並於 commit `a338737` 後按原值執行：681/681 feasible、pool mean/max `78.74/80`、
  macro `0.417862`。它高 S00 `0.010659`、全文 lexical L10 `0.002277`、graph G07
  `0.003031`、Random `0.009018`；但 selection 約 graph G00 的 `18.28×`。兩 primary
  的 S02b 都是目前已跑 proposed 配置最高點估計，構成跨資料集正訊號；後續
  capacity-matched route ablation 與 paired route inference 已完成，但尚未對齊 strong
  baseline matrix，故仍不能 promotion。

## Multi-News capacity-matched route ablation

在看過 S02b aggregate 後、任何 matched-ablation 分數前另行預註冊；固定 total 80、
guard cap 20、Greedy/RRF 與其餘設定，只移除一路：

| 配置 | Routes | Macro | Δ：S02b − 配置 | selection seconds | pool mean/max |
|---|---|---:|---:|---:|---:|
| S02b | lexical + semantic + graph | **0.328077** | — | 1,467.99 | 52.09 / 80 |
| A01 | lexical + graph（無 semantic） | 0.324209 | **+0.003868** | 330.05 | 48.09 / 80 |
| A02 | lexical + semantic（無 graph） | 0.322719 | **+0.005358** | 1,646.88 | 47.18 / 80 |

- A01 的 aggregate 與既有 G04 total-80 完全一致，表示 Multi-News 的 guard cap 20
  沒有改變該配置選句；S02b−A01 因而是比先前 G02 更乾淨的 semantic 邊際比較。
- semantic 與 graph 在 point estimate 都有正增量，graph 的增量較大；但尚未做 paired
  bootstrap／31-config multiplicity correction，不能宣稱任一路已通過保留 gate。
- A01 比 graph G00 更慢，反映 pool 60→80 的 selector 成本；A02 與 S02b 的 wall-time
  差不能只歸因 graph，因兩次獨立 process 仍受系統負載影響。正式 cost 結論需重複 pilot。

## GovReport capacity-matched route ablation

使用同一預註冊與固定 S02b contract：

| 配置 | Routes | Macro | Δ：S02b − 配置 | selection seconds | pool mean/max |
|---|---|---:|---:|---:|---:|
| S02b | lexical + semantic + graph | **0.417862** | — | 1,106.27 | 78.74 / 80 |
| A01 | lexical + graph（無 semantic） | 0.403594 | **+0.014268** | 163.77 | 73.17 / 80 |
| A02 | lexical + semantic（無 graph） | 0.406381 | **+0.011481** | 1,093.27 | 72.39 / 80 |

- A01 與既有 GovReport G04 total-80 aggregate 完全一致；guard cap 20 同樣沒有改變
  該對照。semantic 與 graph 的 point-estimate 邊際皆大，且與 Multi-News 同號。
- 目前證據排除「三路提升全是 total cap 變大」；依 route deletion rule，semantic 與
  graph 都暫留到 paired analysis。這仍不代表它們必須 always-on：A01 比完整 S02b
  快約 `6.75×`，semantic 的品質增益要與成本／adaptive routing 一起判斷。
- A02 與 S02b 耗時相近，顯示 sparse graph 的邊際計算成本小；但正式成本仍需重複測量。

## 預註冊 paired analysis

固定完整 frozen-dev 分母，逐篇重算同 evaluator ROUGE，10,000 bootstrap resamples。
route family 是兩資料集 × 兩個 route removals × 三 metrics 共 12 tests，作 Holm；另以
31 searched configs × 兩資料集 × 三 metrics = 186 opportunities 作 selection-aware
Bonferroni。

### Route 保留結果

| Dataset | 比較 | R-1 Δ [95% CI] | R-2 Δ [95% CI] | R-Lsum Δ [95% CI] |
|---|---|---|---|---|
| Multi-News | S02b − 無 semantic | +0.003994 [0.003126, 0.004854] | +0.002704 [0.001717, 0.003675] | +0.004906 [0.004029, 0.005765] |
| Multi-News | S02b − 無 graph | +0.005306 [0.004396, 0.006189] | +0.003687 [0.002616, 0.004773] | +0.007081 [0.006156, 0.008002] |
| GovReport | S02b − 無 semantic | +0.015568 [0.013985, 0.017163] | +0.011961 [0.010019, 0.013865] | +0.015274 [0.013684, 0.016915] |
| GovReport | S02b − 無 graph | +0.011538 [0.009863, 0.013193] | +0.010108 [0.007862, 0.012289] | +0.012796 [0.011054, 0.014556] |

12/12 endpoints 的 raw bootstrap p 均為 `0.000200`、Holm `0.002400`、186-opportunity
Bonferroni `0.037196`；全部通過預註冊 strong-endpoint rule。§5.3／§5.4 的直接刪除
條件目前未觸發：semantic 與 graph 暫留。這不等於 always-on；semantic 的成本仍需
adaptive quality-cost 判斷。

### Lead／Random 只是 interim，不構成 Gate 2

- Multi-News 對 Lead：R-1 `+0.005726`、R-Lsum `+0.007969` 的 CI 為正，但 R-2
  **`−0.008336`**，95% CI `[−0.010951, −0.005706]`，是顯著缺口。
- GovReport 對 Lead：R-1 `+0.030559`、R-Lsum `+0.023134` 為正；R-2 `+0.002196`
  的 CI `[−0.005531, 0.009608]` 跨 0。
- 對 Random，兩資料集三項 CI 都為正且 12-test Holm 通過；但預註冊 372-opportunity
  correction 在 10,000 resamples 下的最小可得 corrected p 是 `0.074393`，所以
  0/12 cheap-baseline endpoints 通過 strong rule。依「看到分數後不改顯著性方法」規則，
  不事後增加 resamples；只記錄解析度限制。
- PacSum／SBERT+MMR／TextRank／LexRank／greedy reference 尚未在兩 primary 同矩陣完成，
  因此即使 route paired evidence 很強，仍不得進 dev-test。

## 還沒做

1. Gate 2：兩 primary non-PLM 各 23/23、Multi-News PLM 27/27 已完成；Multi-News P08
   高 PLM winner `0.000282`、高 S02b `0.003663`，GovReport LexRank 高 `0.033758`。
   接著完成 GovReport PLM、Lead、Random 與
   metric-specific greedy reference 的兩-primary frozen-dev 矩陣。
2. candidate recall/headroom、adaptive cost rule 與 selector Greedy/MMR/NSGA-II full-dev。
3. 任何 dev-test promotion 或 test。test 在 freeze 簽字前仍禁止。
