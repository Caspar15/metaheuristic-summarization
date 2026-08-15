# Validation development partitions

這裡的 manifest 只在 canonical upstream `validation` 內定義 dev／dev-test membership；
不改 row 的 `split`，也不取代 `configs/data_policies/` 的完整資料驗證。

- dev：可反覆搜尋。
- dev-test：每個 configuration hash 只可評估一次。
- test：freeze 簽字前禁止。

產生與驗證方式見 `scripts/audit/freeze_validation_partitions.py` 與
`src/data/partitions.py`。兩份 v1 manifest 都已在任何 optimization score 前完成：

- Multi-News：seed 3407，dev 3,935／dev-test 1,686；
- GovReport：reference-blind frozen membership，dev 681／dev-test 292。

v2 只允許 GovReport frozen dev 執行 E1～E3 evidence completion。現有 dev-test 不再
用於新候選選擇；test 不在本目錄，且 GovReport test policy 目前刻意尚未 materialize。
