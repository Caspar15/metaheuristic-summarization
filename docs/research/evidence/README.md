# Research evidence index

本目錄保存小型、版本化、可由人或程式核對的 audit evidence。它不是 final-result registry；
大型 predictions、metrics、preregistrations 與正式分析仍在 `runs_v2/`、`configs/` 和對應
status documents。若本目錄的早期 diagnostic 與 final frozen artifact 衝突，以 final artifact
與 `../FINAL_EXPERIMENT_STATUS_2026_08_20.md` 為準。

## 資料與 policy health

- `a1_multinews_dev_reference_lengths.json`
- `a1_govreport_dev_reference_lengths.json`
- `a3_govreport_validation_data_audit.json`
- `multinews_validation_health_summary.json`
- `multinews_test_canonical_health_v1.json`
- `govreport_validation_health_summary.json`
- `govreport_test_canonical_health_v1.json`
- `multinews_final_dry_run_v1.json`
- `govreport_final_dry_run_v1.json`

## Selector evidence

- `selector_comparison_smoke_3row.json`：接線 smoke，不是論文分數。
- `selector_comparison_pilot_v1_summary.json`：200-row pilot。
- `selector_comparison_nsga5_stability.json`：NSGA-II 五 seed stability。
- 完整 D2 frozen-dev evidence 位於 `runs_v2/`，摘要見
  `../SELECTOR_COMPARISON_PROTOCOL.md`。

## Baseline/tokenizer parity

- `nltk_391_310_sentence_split_parity.json`
- `f19_word_tokenizer_parity.json`
- `f19_textrank_lexrank_baselines.json`
- `f19_centrality_final_pipeline.json`
- `f19_textrank_run.txt`
- `f19_lexrank_run_crash.txt`
- `f19_lexrank_run2_isolated_success.txt`
- `f19_lexrank_idf_collapse_scan.json`
- `f45_pacsum_upstream_audit.json`

保留 crash／retry 檔是研究完整性要求；不得只留下成功 attempt。

## Correctness and development diagnostics

- `f17_pr12_validation_regression.json`
- `f18_length_normalized_final_pipeline.json`
- `f18_paired_intersection/`：legacy/common-row paired diagnostic；不是 final official test。
- `f21_greedy_reference_correctness.json`
- `f30_greedy_scaling_projection.json`
- `d1_effective_tunable_inventory.json`

## ICACT extension audit

- `icact_camera_ready_audit_v1.json`：六頁 camera-ready 內容與新 pipeline 的逐頁 mapping；
  bibliography、DOI、award proof 與 similarity report 仍須另外完成。

## 使用規則

1. 每個 claim 必須回到對應 status document 與 frozen run artifact，不直接從檔名推論。
2. `smoke`、`pilot`、`diagnostic`、`projection` 不得改稱 official result。
3. failed attempts 不刪除；排除時記原因。
4. raw SHA-256 不是跨平台 selected-output reproducibility 的唯一判準。
5. 不把 legacy `runs/` 的 test-tuned結果混入 IEEE Access 新稿。

