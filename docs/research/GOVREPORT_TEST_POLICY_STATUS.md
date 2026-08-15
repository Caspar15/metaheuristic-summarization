# GovReport official-test policy status

## Stage A result（2026-08-16 Asia/Taipei）

Stage A 已由提出請求的作者轉述老師與作者端核准，並在任何 test prediction 或 ROUGE
之前完成。核准來源誠實記為 project collaboration message；repo 沒有偽造手寫簽名或
宣稱獨立驗證外部訊息。

| 項目 | 結果 |
|---|---:|
| Official membership | CRS 362 + GAO 611 = **973** |
| Canonical rows | **973** |
| Pre-score exclusions | **0** |
| U+FFFD | **2 rows / 2 source characters**，原樣保留 |
| Canonical sentences | **286,080** |
| Health validation | **pass** |
| Predictions / scores | **0 / 0** |

Frozen identity：

- Policy：`configs/data_policies/govreport_test_v1.json`
- Canonical SHA-256：`3ff10e66ec902b20f0aee0ca1a36a66a2827d24d353473f54c9f7a8df7026dfc`
- Dataset fingerprint：`c5ae4fbb2595d0f27abe8894edf163563a7bc00dbf7713d386ba29987c6cb9c6`
- Health evidence：`docs/research/evidence/govreport_test_canonical_health_v1.json`
- Authorization：`configs/data_policies/govreport_test_authorization_v1.json`

## Stage B 準備進度

- 已完成九系統 one-shot runner 與 official evaluator 的 test-only fail-closed 接線。
- 已凍結 exact scientific commit `36919f222a697fd84d25600b23ba1633ff908ae9`、
  environment、commands 與 output paths。
- 執行凍結檔：`configs/preregistrations/govreport_final_execution_freeze_v1.json`。
- Score-free dry run 已通過：973 IDs／973 references、九個 system families、
  10 個 random seeds 與 official evaluator resources 全部通過，且仍為零
  prediction／零 score。
- Stage B activation 已釘住 freeze 與 dry-run evidence，`ready_for_test=true`。
- 尚未執行：唯一一次 GovReport official-test 正式計分。

此凍結當下 `test_predictions_generated_at_freeze=false` 且
`test_scores_observed_at_freeze=false`；最終分數不得回頭影響方法、baseline、
長度、排除規則或 evaluator。

Stage A 完成只代表資料 policy 可以使用，不代表可以在 runner 未凍結時臨時執行方法。
