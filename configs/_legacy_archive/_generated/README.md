本目錄包含由調參或舊流程產生的設定檔。自 2025-09 起，建議改用下列結構：

- Stage1 Base：`configs/stage1/base/k{K}.yaml`
- Stage1 LLM：`configs/stage1/llm/{bert|roberta|xlnet}/k{K}.yaml`
- Stage2（fast）：`configs/stage2/fast/3sent.yaml`

如需產生新的 Stage1 K 變體，請使用：

```
python scripts/gen_stage1_cfg.py --type base --k 7
python scripts/gen_stage1_cfg.py --type llm --k 7 --model roberta
```

舊有產物（含 `stage2_bert*`、`stage2_fused*` 等）皆已移至 `configs/_archive/`，僅供歷史參考。

