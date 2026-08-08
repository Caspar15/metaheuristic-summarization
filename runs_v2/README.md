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
