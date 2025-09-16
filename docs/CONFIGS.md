# 設定檔指南（CONFIGS）

## 常用模板
- `configs/stage1/base/k20.yaml`：單段基線（tokens 上限）
- `configs/stage2/fast/3sent.yaml`：三句摘要（`unit: sentences, max_sentences: 3`）
- `configs/stage1/base/k20.yaml`：Stage1 Base Top‑K 模板（K1）
- `configs/stage1/llm/bert/k20.yaml`：Stage1 編碼器（BERT）Top‑K 模板（K2）
- `configs/stage1/llm/roberta/k20.yaml`：Stage1 編碼器（RoBERTa）Top‑K 模板（K2）
- `configs/stage1/llm/xlnet/k20.yaml`：Stage1 編碼器（XLNet）Top‑K 模板（K2）
- `configs/_generated/**`：由調參腳本輸出的變體（包含歷史 bert/fused Stage2，現行僅使用 fast 系列）

## 主要欄位
- `length_control`: `unit: tokens|sentences`, `max_tokens`, `max_sentences`
- `representations`: `use: true|false`, `method: tfidf|sbert`
- `candidates`: `use`, `k`, `mode: hard|soft`, `sources: [score|position|centrality]`, `recall_target`, `soft_boost`
- `optimizer.method`: `greedy|grasp|nsga2|bert|roberta|xlnet|fast|fast_grasp|fast_nsga2`
- `fusion`: `w_base`, `w_bert`（在 fast* 中 `w_bert` 代表 TF‑IDF 語義權重）
- `bert.model_name`: HuggingFace 模型名（未指定時依方法給預設）

## LLM 選句（編碼器排序）
- 三擇一：`optimizer.method = bert | roberta | xlnet`（統一路徑）
- 流程：AutoModel 最後層均值池化 → 與文件重心 cosine 排序 → 依 `length_control` 取前 N。
- 模型指定：優先讀取 `bert.model_name`；未指定時預設：
  - `bert` → `bert-base-uncased`
  - `roberta` → `roberta-base`
  - `xlnet` → `xlnet-base-cased`

## 建議搭配
- Stage1 Base K1：`stage1/base/k20.yaml` + `--optimizer greedy|grasp|nsga2`
- Stage1 LLM K2：`features_{bert|roberta|xlnet}_20sent.yaml` + `--optimizer bert|roberta|xlnet`
- Stage2 三句：`stage2/fast/3sent.yaml` + `--optimizer fast|fast_grasp|fast_nsga2`


