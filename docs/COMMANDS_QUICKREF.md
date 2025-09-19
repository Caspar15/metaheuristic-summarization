# 指令快速參考（Two-Stage Pipeline）

以下以 Windows PowerShell 為例。

## 前置
- 啟用環境: `.venv\Scripts\activate`
- 安裝: `python -m pip install -r requirements.txt`

## 單步（單一階段）

1) Stage1 Base（K 可切換）
```
python -m src.pipeline.select_sentences \
  --config configs/stage1/base/k10.yaml \
  --split validation \
  --input data/processed/validation.full.jsonl \
  --run_dir runs \
  --optimizer nsga2 \
  --stamp stage1-nsga2-k10-full
```

2) Stage1 LLM（BERT / RoBERTa / XLNet；K=10）
```
# BERT
python -m src.pipeline.select_sentences --config configs/stage1/llm/bert/k10.yaml \
  --split validation --input data/processed/validation.full.jsonl --run_dir runs \
  --optimizer bert --stamp stage1-bert-k10-full

# RoBERTa
python -m src.pipeline.select_sentences --config configs/stage1/llm/roberta/k10.yaml \
  --split validation --input data/processed/validation.full.jsonl --run_dir runs \
  --optimizer roberta --stamp stage1-roberta-k10-full

# XLNet（建議先安裝 sentencepiece/protobuf）
python -m pip install sentencepiece protobuf
python -m src.pipeline.select_sentences --config configs/stage1/llm/xlnet/k10.yaml \
  --split validation --input data/processed/validation.full.jsonl --run_dir runs \
  --optimizer xlnet --stamp stage1-xlnet-k10-full
```

3) Union + Stage2（fast / fast_grasp / fast_nsga2）
```
python scripts/build_union_stage2.py \
  --input data/processed/validation.full.jsonl \
  --base_pred runs/<stage1-base>/predictions.jsonl \
  --bert_pred runs/<stage1-llm>/predictions.jsonl \
  --out data/processed/validation.full.stage2.union.jsonl \
  --cap 15

python -m src.pipeline.select_sentences \
  --config configs/stage2/fast/3sent.yaml \
  --split validation \
  --input data/processed/validation.full.stage2.union.jsonl \
  --run_dir runs \
  --optimizer fast \
  --stamp stage2-fast-3sent-full

python -m src.pipeline.evaluate \
  --pred runs/stage2-fast-3sent-full/predictions.jsonl \
  --out  runs/stage2-fast-3sent-full/metrics.csv
```

## 一鍵（含時間彙總）
```
python scripts/run_two_stage_timed.py \
  --input data/processed/validation.full.jsonl \
  --run_dir runs \
  --base_cfg configs/stage1/base/k10.yaml --opt1 nsga2 \
  --llm_cfg  configs/stage1/llm/roberta/k10.yaml --opt2 roberta \
  --cap 15 \
  --stage2_cfg configs/stage2/fast/3sent.yaml --opt3 fast \
  --stamp_prefix k10-full
```

## 離線建議（避免 429）
```
$env:TRANSFORMERS_OFFLINE = "1"; $env:HF_HUB_OFFLINE = "1"
# 可選: 指定快取根目錄
$env:HF_HOME = "C:\\Users\\<user>\\.cache\\huggingface"
```

或將 `configs/stage1/llm/{bert|roberta|xlnet}/k{K}.yaml` 的 `bert.model_name` 指向本機快照路徑（例如 `C:\\Users\\<user>\\.cache\\huggingface\\hub\\models--roberta-base\\snapshots\\<SHA>`）。

