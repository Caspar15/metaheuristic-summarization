# 讀懂輸出（runs/ 與 tune_summary）

## 單次 Run（runs/<stamp>/）
- `predictions.jsonl`：逐篇輸出
  - `id`, `selected_indices`, `summary`, `reference`
  - Stage2 的 `selected_indices` 是對「聯集檔（U）的 sentences 陣列」索引
- `metrics.csv`：`metric,value`（rouge1/rouge2/rougeL）
- `config_used.json`：本次使用的完整設定（方便回溯）

## 二階段的中間產物
- `data/processed/validation.stage2.union.k1_<K1>.k2_<K2>.cap_<cap>.<stamp>.jsonl`
  - `sentences`：聯集候選全集；供 Stage2 選 3 句

## 調參總表（tune_union_fusion.py）
- `runs/tune_summary_<stamp>.csv`
  - 欄位：`base_optimizer,k1,k2,cap,method,w_bert,alpha,rouge1,rouge2,rougeL,pred`
  - `base_optimizer`：Stage1 Base 的演算法
  - `method`：Stage2 最終方法（bert 或 fused）
  - `pred`：對應最終 `predictions.jsonl` 的路徑

## 快速對比範例（Python）
```python
import pandas as pd, glob
df = pd.concat([pd.read_csv(p) for p in glob.glob('runs/tune_summary_*.csv')], ignore_index=True)
print(df.sort_values(['rouge1','rouge2','rougeL'], ascending=False).head(10))
```

