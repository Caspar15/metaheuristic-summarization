# Experiment preregistrations

每個 freeze 前實驗都必須先在這裡寫明：改什麼、預期什麼、成功／刪除條件、會看幾次
dev-test，以及多重比較 family。`status=frozen_before_candidate_system_scores` 只代表該份
protocol 在候選 system score 前凍結；若已看過歷史 diagnostic，必須在內容中誠實列出，
不得假裝整個研究從未看過 validation。

修改已凍結 preregistration 時不得覆寫原檔；新增帶版本號的 superseding 文件並說明原因。

`gate2_paired_finalists_v1.json` 在任何 paired resampling outcome 前凍結 S02b 對每個
primary 八個 baseline-family finalists 的 post-score dev diagnostic。它誠實標註 aggregate
與 per-example scores 已存在，固定 64-endpoint Holm family，另報 12,896 個搜尋機會的
selection-aware Bonferroni；結果不得授權 dev-test/test。
