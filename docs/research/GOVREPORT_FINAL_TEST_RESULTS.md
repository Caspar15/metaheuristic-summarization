# GovReport final official-test results

## Final status

GovReport official test 已在 973/973 rows、事前凍結的九個 system families、
LongDocSum-compatible Stanza + Perl ROUGE-1.5.5 尺度下完成。Proposed 排名第一，
預註冊的 proposed 對 full-source SBERT+MMR 主要比較通過，最終決策為：

> **retain GovReport-scoped superiority claim**

此結論只限 GovReport 長篇單文件、no-task-training 設定；不宣稱跨資料集 SOTA。
Multi-News 其後以獨立 frozen secondary protocol 完成 official test：macro 排名第一但
與 PacSum 統計同級，並呈現 R-1/R-L 正向、R-2 負向 trade-off；見
`MULTINEWS_FINAL_TEST_RESULTS.md`。

## Official results

| Rank | System | R-1 | R-2 | R-L | Macro |
|---:|---|---:|---:|---:|---:|
| 1 | **Proposed** | **0.58374** | **0.24711** | **0.54898** | **0.459943** |
| 2 | SBERT+MMR, λ=0.9 | 0.58040 | 0.24092 | 0.54730 | 0.456207 |
| 3 | SBERT centroid | 0.57807 | 0.24106 | 0.54518 | 0.454770 |
| 4 | LexRank | 0.57622 | 0.23889 | 0.54510 | 0.453403 |
| 5 | PacSum-SBERT, β=0.5 | 0.57424 | 0.24199 | 0.54379 | 0.453340 |
| 6 | TextRank | 0.55700 | 0.22163 | 0.52211 | 0.433580 |
| 7 | PacSum TF-IDF P07 | 0.54562 | 0.21222 | 0.51549 | 0.424443 |
| 8 | Random, 10-seed mean | 0.544267 | 0.182574 | 0.512904 | 0.413248 |
| 9 | Lead | 0.52746 | 0.20332 | 0.50178 | 0.410853 |

## Confirmatory inference

Proposed − SBERT+MMR macro 差為 **+0.003700**，973 paired rows、100,000 bootstrap
resamples，95% CI **[+0.002008, +0.005420]**，two-sided `p=0.000040`。

- R-1：`+0.003315`，95% CI `[+0.001642,+0.004997]`，Holm-3 `p=0.000280`。
- R-2：`+0.006150`，95% CI `[+0.004116,+0.008202]`，Holm-3 `p=0.000060`。
- R-L：`+0.001636`，95% CI `[-0.000072,+0.003366]`，Holm-3 `p=0.060279`。

R-L 沒有單獨達顯著，但其 CI upper 不小於 0，因此未觸發事前設定的
component harm guard；`macro_pass=true`、`component_guard_pass=true`、
`final_confirmatory_pass=true`。論文可宣稱 macro 顯著領先，不可宣稱三個
ROUGE components 全部單獨顯著。

## Execution erratum

18 個 prediction runs 與 official/internal per-row scores 全部完成後，frozen runner 在
internal Random 聚合時因 per-row 未存 `macro_rouge` 而 `KeyError`。失敗 evidence
已保留；commit `540cf42` 的 recovery 只對已存 R1/R2/R-Lsum 取算術平均，
再完成原預註冊統計。沒有重跑 prediction、Stanza 或 Perl ROUGE，也沒有修改
任何 source score。

Canonical result artifacts：

- `runs_v2/govreport_final_test_v1/analysis.json`
- `runs_v2/govreport_final_test_v1/execution_evidence.json`
- `runs_v2/govreport_final_test_v1/internal_summary.json`

Analysis SHA-256：`a6a0bbec2393d0c050ab6fd2d8521915c4d66fe7caebbb23f5c65307567a976d`。
Post-score tuning 仍為禁止。
