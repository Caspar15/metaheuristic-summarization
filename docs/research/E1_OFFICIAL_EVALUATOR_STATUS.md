# E1 GovReport official-evaluator status

## 結論（2026-08-15）

E1 已完成。九個 immutable systems 均在同一份 GovReport frozen dev（681 rows）上，
依作者公開 `LongDocSum` 的 Stanza `tokenize,mwt` + Perl ROUGE-1.5.5
`-c 95 -r 1000 -n 2 -m` protocol 重評。dev-test 與 test 均未存取。

Proposed 在官方尺度排名第一；相對預註冊 primary comparator、full-source
SBERT+MMR λ=0.9 的 macro 差為 `+0.004569`，100,000 次 paired bootstrap 的
95% CI 為 `[+0.002467,+0.006680]`、two-sided `p=0.000020`。因此 E1 的 frozen-dev
判準通過，可保留 GovReport superiority claim 到之後的 untouched-test gate；這仍不是
test 結果，也不解鎖 protected split。

## 官方尺度完整結果

| System | R-1 | R-2 | R-L | Macro |
|---|---:|---:|---:|---:|
| **Proposed** | **0.58050** | **0.24943** | **0.54484** | **0.458257** |
| Full-source SBERT+MMR λ=0.9 | 0.57653 | 0.24169 | 0.54285 | 0.453690 |
| LexRank | 0.57375 | 0.24099 | 0.54243 | 0.452390 |
| SBERT centroid | 0.57446 | 0.24185 | 0.54085 | 0.452387 |
| PacSum-SBERT β=0.5 | 0.57181 | 0.24264 | 0.54102 | 0.451823 |
| TextRank | 0.55383 | 0.22350 | 0.51840 | 0.431910 |
| PacSum-TFIDF P07 | 0.53841 | 0.20830 | 0.50759 | 0.418100 |
| Random seed 3407 | 0.53990 | 0.18191 | 0.50813 | 0.409980 |
| Lead | 0.51595 | 0.19414 | 0.49005 | 0.400047 |

所有列的分母都是 681，不能把本表和內部 Google `rouge_score` 或文獻中不同 split／
不同 evaluator 的數字直接混合比較。

## Primary inference 的正確寫法

- Macro：`+0.004569`，95% CI `[+0.002467,+0.006680]`，`p=0.000020`，通過。
- R-1：`+0.003980`，Holm-3 `p=0.000120`，通過。
- R-2：`+0.007730`，Holm-3 `p=0.000060`，通過。
- R-L：`+0.001995`，95% CI `[-0.000106,+0.004120]`，Holm-3 `p=0.063219`，
  **未證實顯著**。

因此可以寫「官方尺度的預註冊 macro、R-1 與 R-2 優於 primary comparator」，不能寫成
「三個 ROUGE 分項全部顯著優於」。

## Provenance 與失敗紀錄

- Frozen implementation commit：`96744a0b7e727fa1663810f9662d7ef25fb3f1a6`。
- 主要 evidence：`runs_v2/govreport_official_evaluator_v1/analysis.json`、
  `environment.json` 與每個 system 的 `evidence.json`。
- Lead 已機械驗證 Perl `-d` 開關不改 corpus score。
- 三個 score-blind Windows 啟動失敗均保留：Unicode repo path、alias guard typo、
  Unix Berkeley DB 無法由 Windows Perl 讀取。最終以 ASCII `R:` alias 與由同一組
  WordNet plaintext 在本機重建的 native DB 解決；runtime hashes 已另行凍結。
- 本階段完整回歸：`464 passed`；freeze verifier 回報
  `protected_splits_unlocked=false`、`test_split_accessed=false`。

## 尚未完成

- E2：30-document reference-blind sample 的 cold/warm runtime、peak RSS 與 scaling。
- E3：五個預註冊 route/provenance ablations 與 20-endpoint Holm inference。
- ICACT camera-ready extension audit、test data policy 與老師／完整作者群簽字。

上述項目完成並通過 freeze audit 前，仍不得跑 test。
