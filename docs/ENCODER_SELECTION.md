# Stage1 編碼器選擇（LLM 句子）

本專案提供一條統一的「編碼器選擇」流程，支援三種 backbone，彼此可以互換：
- `optimizer.method = bert | roberta | xlnet`

## 流程
- 以所選 backbone 對每一句子進行編碼，取最後層的句向量。
- 以句向與文件中心向量的 cosine similarity 進行排序。
- 由 `length_control` 控制輸出長度（句數或 token 數），取得 Top-K 候選。

## 模型與參數
- 透過 HuggingFace `transformers` + `torch` 的 AutoModel；模型名稱由 `bert.model_name` 指定。
- 未指定時的預設值：
  - `bert` → `bert-base-uncased`
  - `roberta` → `roberta-base`
  - `xlnet` → `xlnet-base-cased`

## 範例設定
- BERT K20: `configs/stage1/llm/bert/k20.yaml` + `--optimizer bert`
- RoBERTa K20: `configs/stage1/llm/roberta/k20.yaml` + `--optimizer roberta`
- XLNet K20: `configs/stage1/llm/xlnet/k20.yaml` + `--optimizer xlnet`

## 實作位置
- 三種編碼器共用同一套流程，差異僅在 backbone，詳見 `src/models/extractive/encoder_rank.py`。