# D1 Greedy 敏感度研究狀態

更新日期：2026-08-08（由系統時間取得）

## 邊界與目前進度

- 本研究只讀兩個 primary 的 frozen validation `dev` membership；沒有 CLI split
  參數，且尚未讀取 dev-test 或 test。
- 27 個配置與兩資料集的 delta 已在任何 D1 分數前凍結於
  `configs/preregistrations/d1_greedy_sensitivity_v1.json`。
- 目前完成兩個 primary 的 **lexical/objective family**：各 12/12 configs，
  Multi-News 每個 3,935 rows、GovReport 每個 681 rows。兩資料集的
  cheap-multiroute／semantic 都尚未完成，因此沒有配置可晉級，也尚未做一次性
  dev-test。

| Primary | lexical/objective | cheap multiroute | semantic |
|---|---:|---:|---:|
| Multi-News | **12/12 complete** | pending | pending |
| GovReport | **12/12 complete** | pending | pending |

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

這只是 OFAT 優先序，不是顯著性或 promotion 結論。27-config family 與兩資料集尚未
完成，多重比較的 paired bootstrap 也尚未執行。

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
  `0.003935`。這只是 dev point estimate；12 個 lexical 比較尚未做 paired bootstrap，
  PacSum、SBERT+MMR、TextRank／LexRank 與 greedy reference 也未完成，不能稱為
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
- L00 selected-indices digest `8273f162...d7982` 與 A1 完全一致，但兩者都是 pre-F-30
  artifacts。仍需另跑 reference-blind post-F-30 equivalence audit，才能勾完 F-30。
- GovReport search log 保存 12 final successes 加 1 個 L10 interruption failure；全部
  `dev_test_score=null`、`test_split_accessed=false`。

## 還沒做

1. Multi-News cheap-multiroute 與 semantic family。
2. 兩資料集的 cheap-multiroute／semantic family；不能把 lexical 排名外推。
3. 全 family／跨資料集優先序與 paired bootstrap、多重比較校正。
4. Gate 2 PacSum、SBERT-centroid+MMR、TextRank、LexRank、Lead、Random 與
   metric-specific greedy reference 的兩-primary frozen-dev 矩陣。
5. 任何 dev-test promotion 或 test。test 在 freeze 簽字前仍禁止。
