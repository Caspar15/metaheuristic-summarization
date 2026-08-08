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
