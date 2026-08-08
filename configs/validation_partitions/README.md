# Validation development partitions

這裡的 manifest 只在 canonical upstream `validation` 內定義 dev／dev-test membership；
不改 row 的 `split`，也不取代 `configs/data_policies/` 的完整資料驗證。

- dev：可反覆搜尋。
- dev-test：每個 configuration hash 只可評估一次。
- test：freeze 簽字前禁止。

產生與驗證方式見 `scripts/audit/freeze_validation_partitions.py` 與
`src/data/partitions.py`。Multi-News v1 使用 seed 3407；GovReport 必須在首次
optimization score 前另建 manifest。
