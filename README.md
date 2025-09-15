# 抽取式摘要系統（Extractive Summarization）

本專案的精簡文件已整理到 `docs/`：
- Pipeline 說明與兩階段流程：`docs/PIPELINE.md`
- 輸出與結果閱讀（runs/ 與 tune_summary）：`docs/RUNS.md`
- 設定檔總覽與鍵值意義：`docs/CONFIGS.md`
- 清理/整理建議：`docs/CLEANUP.md`

以下為原始說明（保留）。

本專案是一個模組化抽取式摘要系統，預設以 CNN/DailyMail 類型資料為測試場景。提供基線規則特徵（TF‑ISF、長度、句位）與多種選句優化（Greedy/GRASP/NSGA‑II），並新增不訓練的 BERT 排序器與「分數融合 + MMR」選句（fused）。已支援以「句數」為單位的長度控制（例如強制輸出 3 句摘要）。

目前統一以「前處理 → 選句 → 評估」三段式命令使用。監督式（supervised）相關模組已移除；候選池 top‑k 與 `representations.use` 的行為已落地；新增 `length_control.unit: sentences` 與 `length_control.max_sentences` 以嚴格限制輸出句數。另提供「兩路各取 Top‑K → 聯集 → 最終決策」的二階段流程與調參腳本。

## 功能總覽
- 抽取式基線：規則特徵加權，Greedy + 相似度冗餘控制。
- 進階優化：GRASP 隨機化貪婪 + 局部搜尋；NSGA‑II 多目標（重要性/覆蓋/冗餘）。
- 句向量與相似度：TF‑IDF 或 SBERT 表示，cosine 相似度矩陣。
- 候選池：支援以分數 `top‑k` 限定可選句集合（已生效）。
- 表示開關：`representations.use=false` 時跳過向量/相似度；NSGA‑II 無相似度時自動退回 Greedy。
- 生成式（可選）：BART CNN、Pegasus CNN/DM（需 `transformers`/`torch`）。
- 不訓練 BERT 排序：`optimizer: bert`，以句向量對文件重心的 cosine 排序，遵守三句/長度約束。
- 分數融合 + MMR：`optimizer: fused`，在候選上以 `S = w_base*base + w_bert*bert` 融合後做 MMR 去冗餘。
- 二階段流程（做法 B）：第一層（greedy/GRASP/NSGA‑II）與第二層（BERT）各取 Top‑K，做聯集 U 後再決策最終 3 句。

## 專案結構
```
configs/
data/
  raw/
  processed/
runs/
scripts/
  benchmark_small.py
  split_dataset.py
  jsonl_to_csv.py
  build_union_stage2.py
  tune_union_fusion.py
src/
  data/
  features/
  models/
    extractive/
      bert_rank.py
      fused.py
  pipeline/
  representations/
  selection/
  eval/
```

## 實驗性功能與路徑調整
- 生成式摘要（BART/Pegasus）：移至 `experimental/abstractive/`，屬可選功能，需 `transformers`/`torch`。
- Cross-encoder Rerank（預留樣板）：移至 `experimental/rerank/` 與 `experimental/pipeline/rerank.py`。
- 以上模組目前不影響主流程（前處理 → 選句 → 評估），僅在後續里程碑接入時再啟用。

```
experimental/
  abstractive/
    bart_cnn.py
    pegasus_cnn_dm.py
  rerank/
    cross_encoder.py
  pipeline/
    rerank.py
```

## 安裝與環境
- Python 3.10 以上
- 建立虛擬環境：
  - Windows: `python -m venv .venv && .\.venv\Scripts\activate`
  - Unix: `python -m venv .venv && source .venv/bin/activate`
- 安裝依賴：`pip install -r requirements.txt`
- 可選依賴：
  - 評估：`rouge-score`
  - SBERT/生成式：`torch`, `transformers`, `sentence-transformers`
  - NSGA‑II：`pymoo==0.6.1.1`

## 數據準備
- 預期 CSV 欄位：`id, article, highlights`
- 若你有一個完整 CSV，可用腳本切分：
```
python scripts/split_dataset.py --input path/to/all.csv --out_dir data/raw
```
- 也可用 `scripts/jsonl_to_csv.py` 將 JSONL 轉成 CSV（欄位自動對齊）。

## 快速開始
1) 前處理（分句、過濾短句）
```
python -m src.data.preprocess \
  --input data/raw/validation.csv \
  --split validation \
  --out data/processed/validation.jsonl \
  --max_sentences 25
```

2) 選句與產生摘要
```
python -m src.pipeline.select_sentences \
  --config configs/features_basic.yaml \
  --split validation \
  --input data/processed/validation.jsonl \
  --run_dir runs \
  --optimizer greedy
```

3) 評估 ROUGE
```
python -m src.pipeline.evaluate \
  --pred runs/<stamp>/predictions.jsonl \
  --out  runs/<stamp>/metrics.csv
```

## 三句摘要（常用範例）
- 設定檔：`configs/features_3sent.yaml`（已提供），重點：
  - `length_control.unit: sentences`
  - `length_control.max_sentences: 3`
- 執行（以 validation 為例）：
```
python -m src.pipeline.select_sentences \
  --config configs/features_3sent.yaml \
  --split validation \
  --input data/processed/validation.jsonl \
  --run_dir runs \
  --optimizer nsga2   # 或 greedy / grasp
```
- 注意：NSGA‑II 需 `pymoo`；若 `representations.use=false` 或無相似度則自動退回 Greedy。

## 全量 CNN/DailyMail validation 一鍵示例
```
# 前處理（不抽樣；每篇最多 25 句候選以控資源）
python -m src.data.preprocess \
  --input data/raw/cnn_dailymail/validation.csv \
  --split validation_full \
  --out data/processed/validation.full.jsonl \
  --max_sentences 25

# 以三句設定跑不同選句器
python -m src.pipeline.select_sentences --config configs/features_3sent.yaml --split validation --input data/processed/validation.full.jsonl --run_dir runs --optimizer greedy
python -m src.pipeline.select_sentences --config configs/features_3sent.yaml --split validation --input data/processed/validation.full.jsonl --run_dir runs --optimizer grasp
python -m src.pipeline.select_sentences --config configs/features_3sent.yaml --split validation --input data/processed/validation.full.jsonl --run_dir runs --optimizer nsga2

# 評估（各 run 目錄下）
python -m src.pipeline.evaluate --pred runs/<stamp>/predictions.jsonl --out runs/<stamp>/metrics.csv
```

## 設定說明（configs/features_basic.yaml）
```yaml
objectives:
  lambda_importance: 1.0    # 重要性權重
  lambda_coverage: 0.8      # 覆蓋度權重（NSGA-II 標量化）
  lambda_redundancy: 0.7    # 冗餘懲罰權重（NSGA-II 標量化）
  length_penalty: 2.0       # 保留欄位（目前未直接使用）
representations:
  use: true                 # 為 false 時完全跳過向量/相似度計算
  method: "tfidf"           # tfidf | sbert
  cache: true               # （預留）
candidates:
  use: true
  k: 15                     # 候選池大小（依基線分數）
  mode: "hard"              # hard | soft（soft 僅加權偏好，不封鎖）
  sources: ["score"]        # 聯合集合：score | position | centrality
  soft_boost: 1.05          # mode=soft 時對候選的分數乘法增益
  recall_target: null       # 例 0.95；可選，需有 reference 才能估算 oracle 召回
redundancy:
  method: "mmr"
  lambda: 0.7               # MMR 調和係數（alpha）
  sim_metric: "cosine"
length_control:
  unit: "tokens"            # tokens | sentences
  max_tokens: 100
optimizer:
  method: "greedy"          # greedy | grasp | nsga2 | bart | pegasus
seed: 2024
```

行為補充：
- 候選池已生效：若 `candidates.use=true`，選句只會在 `top‑k` 句內進行。
- `representations.use=false` 時：Greedy/GRASP 可跑（冗餘項視 0）；NSGA‑II 會自動退回 Greedy。
 - 句數長度控制：若 `length_control.unit: sentences` 且設定 `max_sentences`，則最終摘要最多僅含該數量句子（與前處理的 `--max_sentences` 不同）。

## 選句方法
- Greedy：以 `alpha*score - (1-alpha)*max_sim` 的 MMR 型式逐步選句，受長度上限約束。
- GRASP：隨機化貪婪建構（RCL）+ swap/add/drop 局部搜尋，取最佳解。
- NSGA‑II：多目標（重要性最大、覆蓋最大、冗餘最小），最後以 `lambda_*` 標量化挑代表解；需相似度矩陣。

## 腳本工具
- 小樣本基準整套流程（抽樣 train/valid/test + 多種 optimizer + 評估 + 彙總）：
```
python scripts/benchmark_small.py \
  --config configs/features_basic.yaml \
  --raw_dir data/raw \
  --train_n 200 --valid_n 50 --test_n 50 \
  --max_sentences 25
```
- 多分割一鍵跑：
```
python scripts/run_7_2_1.py \
  --config configs/features_basic.yaml \
  --raw_dir data/raw \
  --processed_dir data/processed \
  --run_dir runs \
  --optimizer greedy \
  --max_sentences 25
```

## 二階段 Rerank（預留）
- 目的：對多個候選摘要（由第一層演算法產生）使用 Hugging Face cross-encoder 進行摘要級打分，並與第一層分數做校準後融合。
- 位置：
  - 模型介面：`src/models/rerank/cross_encoder.py`
  - Pipeline：`src/pipeline/rerank.py`（讀入候選、計分、融合、輸出最佳）
- 設定（configs/features_basic.yaml）：
```
rerank:
  enabled: false
  model_name: "cross-encoder/ms-marco-MiniLM-L-6-v2"
  top_n: 20
  normalize: "minmax"   # minmax | zscore
  weights:
    ce: 1.0
    base: 0.0
```
- 待辦：
  - 依你選定的開源模型實作 cross-encoder 打分（batch 推理）。
  - 在 validation 上尋找融合權重（或用小的 meta-learner）。
  - 若配合 LLM rerank，建議 `candidates.mode: soft` 與較大 k，以確保候選召回。

## 設計概覽（Pipeline Overview）
- Prepare（A）
  - P0: 選擇資料與範圍（`scripts/split_dataset.py` 或直接放置 CSV）
  - P1: 清理與分句（`src/data/preprocess.py`）
  - P2: ADM #1（丟棄低價值 tokens）：目前以句級最小長度替代，token 級待擴展
- Represent（B）
  - R1: 句向量（`src/representations/sent_vectors.py`）
  - R2: 相似度矩陣（`src/representations/similarity.py`）
- Select（C）
  - R4: 候選池 top‑k（已生效，`src/pipeline/select_sentences.py`）
  - R5: 選句演算法（`src/models/extractive/{greedy,grasp,nsga2}.py`）
  - R3: 冗餘/覆蓋（Greedy/GRASP 以 MMR 抑制冗餘；NSGA‑II 顧及覆蓋/冗餘）
- Score & Output（D）
  - D1: 第二階段摘要級打分（預計以 cross‑encoder；待加入）
  - D2: 分數融合與長度控制（`features/compose.py`, `selection/length_controller.py`）
  - D3: 匯出摘要（`runs/<stamp>/predictions.jsonl`）
  - D4: 評估（`src/pipeline/evaluate.py`：ROUGE；時間/ablation 待補）

備註與優化方向：
- 可將 R4（top‑k）提前在向量化之前先做，以僅對候選計算向量/相似度，節省計算；若考慮 LLM rerank 的上限，建議使用 `mode: soft` 與較大的 k，或以多來源（sources）提高召回。
- `representations.use=false` 時：Greedy/GRASP 正常、NSGA‑II 退回 Greedy（已落地）。
- 後續將新增 D1（二階段 rerank）與（可選）RL 前置搜尋，提升最終摘要品質。

## 兩種「句數上限」的差別
- 前處理 `--max_sentences`：控制每篇文在進入選句前最多保留多少候選句（效能/記憶體考量；會影響可選集合大小）。
- 摘要長度 `length_control.max_sentences`：控制最終摘要輸出最多包含幾句（不影響候選集合大小）。

## 疑難排解（Troubleshooting）
- 缺少 rouge-score：
  - 現象：`RuntimeError: 請先安裝 rouge-score ...`
  - 解法：`pip install -r requirements.txt` 或單獨安裝 `rouge-score`。
- NSGA‑II 無法使用或 ImportError：
  - 現象：`pymoo` 未安裝或環境不符；或 `representations.use=false` 導致無相似度。
  - 行為：會自動退回 Greedy（程式已處理）。
  - 建議：需要 NSGA‑II 時，確保 `pymoo` 就緒，且 `representations.use=true`。
- SBERT/transformers 未安裝：
  - 現象：選用 `representations.method: sbert` 時報錯。
  - 解法：安裝 `torch`, `sentence-transformers`；或改回 `tfidf`。
- Windows/路徑編碼：
  - 現象：非 ASCII 路徑導致讀寫錯誤或亂碼。
  - 解法：在 PowerShell 下執行並確保編碼為 UTF‑8；資料路徑盡量使用 ASCII。
- 記憶體壓力（相似度矩陣太大）：
  - 方案：降低 `--max_sentences`，或開啟候選池並減小 `candidates.k`；未來可啟用「僅對候選向量化」優化。
- 中文長度控制失真：
  - 原因：tokens 以空白切分，中文常視為 1 token。
  - 臨時解法：降低 `max_tokens`；或改用 `length_control.unit: sentences`；後續將提供 `unit: chars` 與 `tokenizer` 選項。

## 常見問題（FAQ）
- 未安裝 `rouge-score` 導致評估報錯：請先 `pip install -r requirements.txt`。
- 使用 SBERT：請安裝 `torch`, `sentence-transformers`，並在設定檔 `representations.method: sbert`。
- 長度單位：目前以 token 計數（以空白切分），中文文本建議先做斷詞再導入，或降低 `max_tokens`。
- Windows JSON 編碼：程式以 `utf-8` 讀寫；若路徑含非 ASCII 字元，建議在 PowerShell 下執行並確保編碼為 UTF‑8。

## 變更記錄（重點）
- 2025‑08‑29：新增 Project_status 提供後續協作處理。
- 2025‑09‑01：移除監督式模組；候選池生效；尊重 `representations.use` 並在 NSGA‑II 無相似度時退回 Greedy；清理未用檔案。
- 2025‑09‑03：新增 `length_control.unit: sentences` 與 `max_sentences`，可直接指定輸出句數（如 3 句）；補充 README/PROJECT_STATUS 範例與指引。

## 授權
- 本專案未附帶明確授權條款。如需發布或再利用，請先於內部確認。
