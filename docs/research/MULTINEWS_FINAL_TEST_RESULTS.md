# Multi-News final official-test results

## Final status

Multi-News official test 已在 5,621/5,621 rows、事前凍結的九個 system
families、Stanza + Perl ROUGE-1.5.5 尺度下完成。Proposed 的三指標 macro
排名第一，但與事前指定的最強 frozen-dev baseline PacSum-TFIDF P08 幾乎同分，
主要比較不顯著。因此最終決策為：

> **report the frozen Multi-News secondary result without a superiority claim**

這不是負面失敗：Proposed 顯著提高 R-1 與 R-L，也顯著勝過 Lead、Random、
TextRank、LexRank、SBERT centroid 與 SBERT+MMR 的 macro；但它在 R-2 顯著低於
兩個 PacSum 變體。論文必須完整揭露這個 precision／coverage trade-off，不可宣稱
在 Multi-News 全面勝過最強 baseline。

## Official results

| Rank | System | R-1 | R-2 | R-L | Macro |
|---:|---|---:|---:|---:|---:|
| 1 | **Proposed** | **0.45011** | 0.14314 | **0.41351** | **0.335587** |
| 2 | PacSum TF-IDF P08 | 0.44773 | 0.14990 | 0.40888 | 0.335503 |
| 3 | PacSum SBERT P03 | 0.44590 | **0.15002** | 0.40696 | 0.334293 |
| 4 | Lead | 0.43901 | 0.14729 | 0.40077 | 0.329023 |
| 5 | SBERT+MMR, lambda=0.7 | 0.44147 | 0.13977 | 0.40078 | 0.327340 |
| 6 | LexRank | 0.43803 | 0.13884 | 0.39863 | 0.325167 |
| 7 | SBERT centroid | 0.43277 | 0.13614 | 0.39236 | 0.320423 |
| 8 | Random, 10-seed mean | 0.423227 | 0.123884 | 0.386661 | 0.311257 |
| 9 | TextRank | 0.41947 | 0.12991 | 0.37792 | 0.309100 |

## Primary confirmatory comparison

Proposed - PacSum-TFIDF P08 的 macro 差為 **+0.000091**，5,621 paired rows、
100,000 bootstrap resamples，95% CI **[-0.001542, +0.001730]**，two-sided
`p=0.907111`。`macro_superiority_supported=false`。

- R-1：`+0.002389`，95% CI `[+0.000776,+0.004000]`，Holm-3
  `p=0.003520`。
- R-2：`-0.006758`，95% CI `[-0.008667,-0.004863]`，Holm-3
  `p=0.000060`。
- R-L：`+0.004641`，95% CI `[+0.003030,+0.006233]`，Holm-3
  `p=0.000060`。

Proposed 對 PacSum-SBERT P03 的 macro 差為 `+0.001303`，95% CI
`[-0.000392,+0.002980]`，Holm-32 `p=0.259037`，同樣不顯著。對其餘六個
baseline 的 macro 優勢皆在 Holm-32 校正後顯著。

## Reporting boundary

- GovReport 是主要 confirmatory long-single-document benchmark；其顯著 macro
  優勢不因本結果改變。
- Multi-News 是獨立凍結的 secondary multi-document benchmark。可報告排名第一、
  R-1/R-L 優勢與跨資料型態穩健性；不可報告相對 PacSum 的 macro 顯著優勢。
- test 分數不得用來修改方法、選擇新配置或重跑另一套 protocol。
- 內部 `rouge-score` parity 分析結論一致：Proposed macro `0.331734`、P08
  `0.331412`，差 `+0.000322`，95% CI 跨 0，`p=0.694`。

Canonical result artifacts：

- `runs_v2/multinews_final_test_v1/analysis.json`
- `runs_v2/multinews_final_test_v1/internal_analysis.json`
- `runs_v2/multinews_final_test_v1/execution_evidence.json`

Official analysis SHA-256：
`fd2410f28c9c27dd8c7087fa7346fda20886b8e9e435b8558721a203219b45c7`。
Post-score tuning 仍為禁止。
