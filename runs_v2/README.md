# Governed experiment registry

`search_log.jsonl` 是 freeze 前所有候選配置的 append-only 索引，包括失敗 run。
每個 `run` record 至少要保存：config hash、dataset／partition manifest identity、dev
分數、若有則唯一一次 dev-test 分數、晉級決策與原因、evidence 路徑、實際量測時間。

大型 `predictions.jsonl`／`per_example.jsonl` 由 `.gitignore` 排除，但每個 run 的小型
evidence JSON 必須版本化，包含 artifact SHA-256、dependency 版本、command、commit、
runtime 與完整失敗原因。artifact SHA 不得被宣稱為跨機器可重現性證據；跨機器比較
逐篇 `selected_indices`。

`registry_initialized` 是治理事件，不是實驗分數。後續禁止覆寫或刪除舊 record；若要
更正，新增 superseding record 並指向被更正的 ID。

## D1 目前狀態（2026-08-08）

`d1_greedy_sensitivity/multinews/dev/lexical_objective/` 已完成 12 個 final runs。
`search_log.jsonl` 對本 family 保存 12 筆 `run_attempt=final` success，以及一筆
`attempt_01_interrupted` failure；後者是外層 job 60-minute timeout，partial artifact
與 interruption evidence 均保留，之後以 `--resume` 只補跑缺少的 L11。

candidate diagnostics schema v2 的 `selector_candidate_size_*` 才是 selector 實際
搜尋空間；`provenance_candidate_size_mean` 是 candidate builder 產生的 records。關閉
prefilter 時後者為 0、前者為全文，兩者不得混稱。研究解讀與未完成項目見
`docs/research/D1_SENSITIVITY_STATUS.md`。

`d1_greedy_sensitivity/govreport/dev/lexical_objective/` 亦已完成 12 個 final runs，
每個 681 frozen-dev rows；另保留一筆 pre-F-30 L10 external interruption failure。
L10 resume 使用 commit `4198025` 的 exact batched Greedy additions；舊 partial 的
Windows size=0 誤判與更正見 F-31。GovReport 本 family 沒有讀 dev-test/test，不能因
L10 dev point estimate 高於 Lead／Random 就提前晉級。

`f30_greedy_equivalence/govreport_l00_post_f30/` 是固定的 reference-blind 真實 pipeline
等價性稽核：681 frozen-dev rows 的逐篇 selected indices 與 pre-F-30 L00 artifact
**0 差異**，兩者 digest 同為 `8273f162...d7982`。`predictions.jsonl` 依慣例不進 Git；
可提交的 `evidence.json` 記錄 commit、輸入/config hash、列數與 split guards。

`d1_greedy_sensitivity/multinews/dev/cheap_multiroute/` 已完成 12 個 final runs，每個
3,935 frozen-dev rows；另保留 G00/G01 各一個 external interruption failure。G02
route-top-K 80 的 macro `0.324305` 居首，selector pool mean/max `48.03/60`；相對
純 lexical L00 是 `+0.013952`，但仍低於同協定 Lead macro `0.326291`。family summary
明示 `dev_test_accessed=false`、`test_split_accessed=false`，不得提前 promotion。

`d1_greedy_sensitivity/govreport/dev/cheap_multiroute/` 已嘗試全部 12 個原預註冊
configs：11 success，G11 因 77 mandatory section/route reservations 超過 total cap 60
而 fail loud。原 failure 永久保留；另在執行前預註冊
`d1_govreport_section_guard_followup_v1.json`，以容量推導的 `max_items=20` 作獨立
feasibility follow-up，不取代 G11。原 family 沒有讀 dev-test/test。

`d1_section_guard_followup/govreport/dev/G11b_section_guard_cap20/` 是上述獨立 follow-up：
prereg commit `3b6813e` 後才執行，681/681 feasible、pool max 60、macro `0.404182`；
evidence 與 search log 都記 `comparison_family_size=28`、dev-test/test 未讀。

`d1_greedy_sensitivity/multinews/dev/semantic_route/` 原三案為 2 success + S02 structural
failure。S00 macro `0.322404`、selection `1,387.23 s`；S01 `0.299634`。S02 的三路
reservations 加 document guard 為 mandatory 61 > cap 60，原 failure 保留；另在執行前
預註冊兩-primary S02b capacity follow-up（total 80、guard max 20）。

`d1_three_route_followup/multinews/dev/S02b_three_route_capacity_80/` 已在 prereg commit
`73769c5` 後完成：3,935 rows、3,930 feasible、pool mean/max `52.09/80`、macro
`0.328077`，dev-test/test 未讀。它高於 graph G02 與 Lead 的 dev 點估計，但同時變更
total/guard cap，不能當 semantic-only ablation；原 S02 structural failure 仍永久保留。

`d1_greedy_sensitivity/govreport/dev/semantic_route/` 已完成原三案：S00 macro
`0.407203`、selection `917.18 s`；S01 `0.349832`、`974.81 s`；S02 第一列 mandatory
61 > cap 60，原 structural failure 保留。成功 runs 各 681 rows、dev-test/test 未讀。
S01 接法在兩資料集均失敗，後續刪除；GovReport S02b 依既有 preregistration 另跑。

`d1_three_route_followup/govreport/dev/S02b_three_route_capacity_80/` 已在 commit
`a338737` 後完成：681/681 feasible、pool mean/max `78.74/80`、macro `0.417862`、
selection `1,106.27 s`，dev-test/test 未讀。它是目前 GovReport proposed 最高點估計，
但仍是 capacity follow-up 而非 route 純 ablation；原 S02 failure 不被取代。

下一步的 `d1_capacity_matched_route_ablation/` 已在任何該 family 分數前預註冊：從
S02b 只移除 semantic 或 graph，其他 capacity/guard/selector contract 固定。註冊時間
明確晚於 S02b aggregate，不冒充事前未知；runner 固定 validation-dev，無 split CLI。

`d1_capacity_matched_route_ablation/multinews/dev/` 已完成：A01 無 semantic macro
`0.324209`、A02 無 graph `0.322719`，S02b 分別高 `0.003868/0.005358`。兩個 runs
各 3,935 rows、pool max 80，dev-test/test 未讀；後續 paired route analysis 已完成，結果見
`d1_paired_analysis_v1/summary.json`。

`d1_capacity_matched_route_ablation/govreport/dev/` 已完成：A01 無 semantic macro
`0.403594`、A02 無 graph `0.406381`，S02b 分別高 `0.014268/0.011481`。兩個 runs
各 681/681 feasible、pool max 80，dev-test/test 未讀；跨資料集 point estimate 同號，
後續 paired route analysis 已完成，結果見 `d1_paired_analysis_v1/summary.json`。

`d1_paired_analysis_v1/` 已在逐篇 scoring 前預註冊並完成。它固定比較 S02b 對兩個 route
removals 與 Lead／Random；各 family 先作 12-test Holm，再報 31-config 搜尋對應的
186／372-opportunity selection-aware Bonferroni。route 12/12 endpoints 通過 strong rule；
cheap-baseline 0/12，且 Multi-News 對 Lead R-2 顯著為負。10k resamples 對 372 correction
的最小 corrected p=`0.074393`，不事後改協定。runner 無 split CLI，只讀 frozen dev。
summary SHA-256 `4b63b59a7f6ddf96fa9698522909ebe363fa96ae2ca5c937357577bbdf877937`。

## Gate 2 baseline matrix 狀態（2026-08-09）

Multi-News PLM family 已完成 27/27。winner `pacsum_sbert_P03_previous_-0.3` 的
R1/R2/Lsum 為 `0.442111/0.150742/0.401520`、macro `0.331458`；仍比 non-PLM
P08 低 `0.000282`。最佳 full-source SBERT-MMR 是 λ=0.7（macro `0.322581`），低
proposed S02b `0.005496`。第三候選的中斷已依 F-48 封存並登錄，final retry 成功。
F-51 execution-only cache 已先通過 3,935-row cold/warm exact audit；F-53 verifier 又
確認後續 25 runs 共 98,375 hits、同一 contract／ordered-key digest 與完整 PLM runtime
versions。證據在 `f51_embedding_cache_equivalence_v1/multinews/dev/equivalence_summary.json`
及 `gate2_baseline_matrix_v1/multinews/dev/plm/analysis_summary.json`。所有輸出只讀
frozen dev；dev-test/test 未讀。

`gate2_baseline_matrix_v1/multinews/dev/non_plm/` 已完成事前註冊的 23/23 candidates。
family winner 是非退化 `pacsum_tfidf_P08_previous_-0.8`，macro `0.331740`；beta=1.0
在 3,935/3,935 rows 全部 score-degenerate，只能視為 canonical-order skip-tolerant control。
P07 第一次外層中斷保留於 `attempt_01_interrupted/`，失敗與 final retry 都已寫入
`search_log.jsonl`。`analysis_summary.json` 固定驗證 partition guards、ranking、退化率與
相對 frozen Lead 的逐篇選句重疊。兩 primary PLM 現已完成；greedy reference 與
paired inference尚未完成；dev-test/test 未讀。

`gate2_baseline_matrix_v1/govreport/dev/plm/` 已完成 27/27。winner 是 full-source
SBERT-MMR λ=0.9，R1/R2/Lsum `0.575010/0.241576/0.541718`、macro `0.452768`；
只高 non-PLM LexRank `0.001148`、高 S02b `0.034906`。P06 的一次外層中斷保留於
`attempt_01_interrupted/`，failed attempt 與 final retry 都在 search log。F-53 驗得
27×681 row accesses、680 unique cold misses、17,707 hits；dev-test/test 未讀。

`gate2_baseline_matrix_v1/govreport/dev/non_plm/` 亦完成 23/23，無 failed attempt。
winner 是 LexRank，R1/R2/Lsum `0.572517/0.241004/0.541340`、macro `0.451620`，比
frozen Lead 高 `0.052388`、比 proposed S02b 高 `0.033758`；TextRank 亦比 S02b 高
`0.013104`。LexRank 與 Lead 的 exact selected-indices match 為 0/681、mean Jaccard
`0.059722`。GovReport beta=1 也在 681/681 rows 退化。PLM／greedy reference／paired
inference 尚未完成，dev-test/test 未讀。
