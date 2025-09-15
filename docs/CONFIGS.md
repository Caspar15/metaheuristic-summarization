# 設定檔與用途

## 常用檔
- `configs/features_basic.yaml`：預設（單階段 tokens 上限）。
- `configs/features_3sent.yaml`：單階段三句摘要（`unit: sentences, max_sentences: 3`）。
- `configs/features_20sent.yaml`：取 20 句（常作 Stage1 K 設定模板）。
- `configs/features_bert_20sent.yaml`：Stage1 BERT 路徑（取 K2 句；不使用候選/相似度）。
- `configs/features_fused_3sent.yaml`：Stage2 fused（三句；內部用 BERT 向量做相似度）。

## 重要鍵值
- `length_control`: `unit: tokens|sentences`, `max_tokens`, `max_sentences`
- `representations`: `use: true|false`, `method: tfidf|sbert`
- `candidates`: `use`, `k`, `mode: hard|soft`, `sources: [score|position|centrality]`, `recall_target`
- `optimizer.method`: `greedy|grasp|nsga2|bert|fused`（或 `fast`）
- `fusion`: `w_base`, `w_bert`（Stage2=fused）
- `bert.model_name`: 預設 `google-bert/bert-base-uncased`

## 建議搭配
- Stage1 Base 三句：`features_3sent.yaml` + `--optimizer greedy|grasp|nsga2`
- Stage1 BERT K2：`features_bert_20sent.yaml` + `--optimizer bert`
- Stage2 bert（最終三句）：`features_bert_3sent.yaml` + `--optimizer bert`
- Stage2 fused（三句+MMR）：`features_fused_3sent.yaml` + `--optimizer fused`

