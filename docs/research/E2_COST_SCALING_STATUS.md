# E2 GovReport cost, memory, and scaling status

## 結論（2026-08-15）

E2 已完成。30 篇 reference-blind GovReport frozen-dev 文件按來源長度 q10/q50/q90
各取 10 篇；九個系統在無重疊 timed jobs 的 CPU-only 環境中，各跑 cold／warm-cache
一個 discarded smoke 與三個 measured fresh subprocess。正式 54/54 repetitions 完整，
跨 repetition 與 cold/warm 的 selected-index digests 全部一致。另保留一個 cache 分類錯誤
後主動中斷、且未納入分析的失敗 attempt。dev-test/test 均未存取。

## 30-document median results

| System | Cache applicable | Cold wall (s) | Warm/control wall (s) | Cold peak RSS (MB) | Warm/control peak RSS (MB) |
|---|:---:|---:|---:|---:|---:|
| Lead | No | 4.64 | 4.63 | 335.5 | 336.0 |
| PacSum-TFIDF P07 | No | 5.16 | 5.17 | 341.2 | 338.8 |
| LexRank | No | 30.50 | 30.42 | 344.4 | 344.4 |
| PacSum-SBERT β=0.5 | Yes | 173.12 | 7.12 | 770.8 | 410.8 |
| SBERT centroid | Yes | 173.31 | 7.11 | 764.4 | 412.9 |
| **Frozen C01 Proposed** | **Yes** | **176.03** | **10.52** | **774.1** | **422.9** |
| D2 matched Greedy-TFIDF | Yes | 176.26 | 10.35 | 773.0 | 419.6 |
| Full-source SBERT+MMR λ=0.9 | Yes | 177.81 | 12.45 | 764.9 | 413.2 |
| D2 matched NSGA-II-TFIDF | Yes | 222.61 | 56.74 | 777.8 | 421.9 |

MB 以 `1 MiB = 1,048,576 bytes` 換算。每格是三次 measured fresh subprocess 的
median；smoke 與 warm-prime 均不納入。`warm/control` 對 cache-inapplicable 系統只是
相同條件的控制重跑，不能解讀成享有 embedding cache。

## 可寫與不可寫的解釋

- Proposed 相對 primary quality comparator SBERT+MMR：cold wall 約快 1.0%，warm 約快
  15.5%；cold/warm peak RSS 則約高 1.2%／2.3%。E1 同時顯示 Proposed 官方 macro
  顯著較高，因此目前沒有「為品質付出更慢 wall-time」的證據，但有小幅 RAM trade-off。
- Proposed 相對 matched Greedy：cold 幾乎相同，warm median只多約 0.16 秒／30 篇；
  TF-IDF-MMR selector 的額外成本遠小於 semantic encoding。
- NSGA-II 相對 Proposed：cold 約 1.26×、warm 約 5.40×；這支持把 NSGA-II 留作
  ICACT 延伸中的 matched comparator/負結果，而不是 IEEE Access 主 selector。
- LexRank 的 cold/control 約 30.5 秒且 per-document sentence-count log-log slope 約
  1.8，顯示長文件全句 graph 計算具有明顯超線性成本。其餘 slopes 只作 30 篇描述性
  scaling diagnostic，不能當普遍複雜度定理。
- 不可把 Proposed warm 10.52 秒與其他方法 cold 數字比較；主表必須 cold 對 cold、
  warm 對 warm，並同時報 cache lifecycle。

## Cache-classification erratum

D2 Greedy／NSGA-II 名稱雖是 TF-IDF selector，但 frozen S02b candidate generator 的
`compute_budget.enabled_routes` 含 semantic，因此也會產生 embeddings。第一次 runner
把它們誤列為 non-cache systems；D2 Greedy cold smoke 在 170.41 秒時被主動中斷，
partial cache、RSS trace 與失敗 evidence 完整保留且排除。修正後兩者均先 prime 13.73 MB
cache，cold 使用獨立空 cache，warm 使用 byte-verified shared cache。詳見
`configs/preregistrations/govreport_e2_cache_classification_erratum_v1.json`。

## Evidence

- Summary：`runs_v2/govreport_cost_scaling_v1/analysis.json`
- Environment：`runs_v2/govreport_cost_scaling_v1/environment.json`
- Sample：`configs/pilot_manifests/govreport_cost_scaling_sample_v1.json`
- Cost addendum：`configs/preregistrations/govreport_centered_cost_addendum_v1.json`
- 78 completed attempts + 1 failed attempt 均已進 `runs_v2/search_log.jsonl`。

## 尚未完成

- ICACT camera-ready extension audit、test data policy、freeze audit 與完整簽字。

E3 已於同日完成；見 `E3_ROUTE_PROVENANCE_ABLATION_STATUS.md`。

E2 只補成本證據，不能依 timing 重新調參、promote 方法或解鎖 test。
