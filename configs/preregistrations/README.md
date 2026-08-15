# Experiment preregistrations

**2026-08-15 execution status:** all method-search registrations through D3b are closed,
and the frozen-dev E1-E3 evidence-completion study is complete. Its post-evidence inventory
is `govreport_pretest_evidence_index_v1.json`. `govreport_centered_final_evaluation_v1.json`
remains a locked protocol, not permission to access test; policy ordering and human sign-off
were resolved through `../data_policies/govreport_test_authorization_v1.json`. Stage A
policy materialization is complete. The exact runner, scientific commit, environment,
commands, output paths, nine system families, and evaluator resources are pinned by
`govreport_final_execution_freeze_v1.json` before any test prediction or score. Test
scoring remains locked until the score-free dry run is committed and an activation file
pins that evidence.

每個 freeze 前實驗都必須先在這裡寫明：改什麼、預期什麼、成功／刪除條件、會看幾次
dev-test，以及多重比較 family。`status=frozen_before_candidate_system_scores` 只代表該份
protocol 在候選 system score 前凍結；若已看過歷史 diagnostic，必須在內容中誠實列出，
不得假裝整個研究從未看過 validation。

修改已凍結 preregistration 時不得覆寫原檔；新增帶版本號的 superseding 文件並說明原因。

`gate2_paired_finalists_v1.json` 在任何 paired resampling outcome 前凍結 S02b 對每個
primary 八個 baseline-family finalists 的 post-score dev diagnostic。它誠實標註 aggregate
與 per-example scores 已存在，固定 64-endpoint Holm family，另報 12,896 個搜尋機會的
selection-aware Bonferroni；結果不得授權 dev-test/test。

`govreport_centered_evidence_completion_v1.json` 在新 official-evaluator score、受控成本量測
或 final GovReport ablation 前，凍結 E1 evaluator parity、E2 cold/warm cost/scaling
與 E3 route/provenance ablation。這三組是 evidence completion，不是新的方法搜尋；
只能用 frozen GovReport dev，dev-test/test 仍禁止。

`govreport_centered_final_evaluation_v1.json` 預先凍結 one-shot GovReport final protocol，
但狀態明確為 execution locked。在 E1–E3、GovReport test policy、exact commit/environment
與老師／完整作者群簽字全部完成前，不得建立 test run。

`govreport_final_execution_freeze_v1.json` 是 Stage B 的機器可驗證凍結：它釘住
973-row canonical test、最終 proposed config、九個 system families、10 個 random seeds、
official evaluator 資源、套件版本、16 workers 與輸出路徑。建立當下為零
prediction、零 score；它本身不是執行許可，還需 score-free dry run 與另一份
activation 檔。
