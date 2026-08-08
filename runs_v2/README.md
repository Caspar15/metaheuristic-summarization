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
