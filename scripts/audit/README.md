# scripts/audit — 稽核診斷腳本

這些腳本把先前只存在暫存目錄的稽核分析**版本化**，讓 `CODE_AUDIT_IEEE_Access.md`
與 `STRATEGY_ASSESSMENT.md` 引用的數字可以被獨立重現。

> ⚠️ **這些是 diagnostic，不是論文結果。**
> 全部使用 `src.eval.rouge` 的**內部多句 Lsum 協定**，與 published Perl ROUGE 數字
> 不保證可比。greedy reference **不是** exact upper bound，也不是任何資料集的官方 oracle 協定。
> 正式結果必須走 `ACTION_PLAN.md` Phase 2–4 的鎖定流程。

> 2026-08-02 狀態：PR #10 已把 production Lead 移到 `src.baselines.cli`；
> 本目錄的 `lead_vs_system.py` 仍只用來重現 test-tuned legacy F-0，不是 Phase 2
> baseline runner，也不會因 Lead 程式已合併而自動變成投稿級結果。

執行位置：`metaheuristic-summarization/`（模組路徑需要 repo root 在 `sys.path`）

---

## `lead_vs_system.py` — F-0：系統 vs 本地 Lead

在同資料、同 evaluator、ID 對齊的條件下比較某個 run 與 Lead。

```bash
.venv/Scripts/python.exe -m scripts.audit.lead_vs_system \
  --data data/processed/multi_news_test.jsonl \
  --pred runs/tuning_experiments/ExpB_K20_Max_Coverage/predictions.jsonl \
  --budget 245
```

**已重現的輸出**（5,622 篇，2026-07-26）：

| system | R-1 | R-2 | R-Lsum | words |
|---|---|---|---|---|
| System run（legacy ExpB） | 0.4352 | 0.1405 | 0.3880 | 241.3 |
| Lead, 245-word budget | 0.4331 | **0.1453** | **0.3901** | 228.0 |
| Lead, per-doc length-matched | 0.4325 | **0.1449** | **0.3895** | 225.8 |

差值：R-1 `+0.0021` / R-2 `−0.0048` / R-Lsum `−0.0021`。
**未做 paired significance test**，小差距不可宣稱勝負。

⛔ **R-Lsum 那一欄已過期**（2026-07-30, PR #9）：evaluator 的分句器換成
`src/data/sentence_split.py` 的共用 Punkt tokenizer，實測 R-Lsum 位移 **+0.0032**
（R-1 / R-2 為 +0.0000，不受影響，`+0.0021` 與 `−0.0048` 仍有效）。
表中三個 Lsum 值與 `−0.0021` 的差距都必須重跑本腳本才能再引用。

---

## `selection_diagnostics.py` — 病因：選句位置與重疊率

```bash
.venv/Scripts/python.exe -m scripts.audit.selection_diagnostics \
  --data data/processed/multi_news_test.jsonl \
  --pred runs/tuning_experiments/ExpB_K20_Max_Coverage/predictions.jsonl \
  --budget 245 --limit 200
```

**已重現的輸出**（200 篇，2026-07-26）：

| 誰在選 | 位置中位數 | 前 25% 佔比 |
|---|---|---|
| Greedy reference（目標） | 0.462 | 31.3% |
| Lead | 0.082 | 86.9% |
| System run | 0.143 | 67.6% |

- System 選句也被 **Lead** 選中：**61.7%**
- System 選句也被 **greedy reference** 選中：**22.8%**

> 這是「系統行為像昂貴版 Lead」的量化依據。
> `22.8%` **不是** official oracle recall；新 pipeline 必須以 validated oracle 重做。

---

## `dataset_headroom.py` — 主場選擇：Lead 之上還有多少空間

```bash
# Multi-News（word budget）
.venv/Scripts/python.exe -m scripts.audit.dataset_headroom \
  --data data/processed/multi_news_test.jsonl --budget 245 --limit 200

# CNN/DailyMail（3 句）
.venv/Scripts/python.exe -m scripts.audit.dataset_headroom \
  --data data/processed/_archive_legacy/cnn_dm_test.jsonl --lead_sentences 3 --limit 200

# SciTLDR-AIC（1 句）
.venv/Scripts/python.exe -m scripts.audit.dataset_headroom \
  --data data/processed/_archive_legacy/scitldr_test.jsonl --lead_sentences 1 --limit 200
```

**已重現的輸出**（各 200 篇抽樣，2026-07-26）：

| 資料集 | Lead R-1 | Greedy ref R-1 | Headroom | 位置中位數 | 前 25% |
|---|---|---|---|---|---|
| Multi-News | 0.4383 | 0.5901 | 0.1518 | 0.46 | 31.3% |
| CNN/DailyMail | 0.4003 | 0.5709 | 0.1707 | 0.21 | **57.4%** |
| SciTLDR-AIC | 0.1979 | 0.3876 | 0.1897 | 0.49 | 33.0% |

> 前 25% 佔比高 = lead bias 強 = Lead 難以擊敗。CNN/DM 最不適合當主場。
> 抽樣值，非全集；不可作正式結果引用。

---

## `plm_timing.py` — PLM 成本分解（載入 vs 推論）

```bash
.venv/Scripts/python.exe -m scripts.audit.plm_timing --sentences 40 --repeats 3
```

**兩次量測（同機器、不同 thread 數與磁碟快取狀態）：**

| 量測 | BERT 載入佔比 | 純推論 BERT/RoBERTa | 載入+推論 BERT/RoBERTa |
|---|---|---|---|
| 2026-07-26 第一次（20 threads） | 78.1% | 1.04× | 2.64× |
| 2026-07-26 第二次（14 threads） | 92.7% | 1.02× | 3.28× |

> ⚠️ **載入佔比在不同執行間差異極大（78% ↔ 93%），不可引用特定百分比。**
> 穩定的只有兩件事：
> 1. **純推論比值 ≈ 1.0** —— 兩個架構等價的 encoder 本來就該如此（直接回答 R4 的疑問）
> 2. **載入時間遠大於推論時間**，因此舊稿的 per-article 計時主要在量模型重複建構
>
> **正式數字必須依鎖定的 runtime protocol 重測**：固定硬體、thread/batch、
> 排除 warm-up、≥5 次重複、報 median/mean/std/P95。

---

## `random_baseline_min_words.py` — Random baseline 的 `apply_min_words` 決策

```bash
python -m scripts.audit.random_baseline_min_words \
  --data data/processed/multi_news_validation_canonical.jsonl \
  --max_words 250 --min_words 200 --seeds 0,1,42,9999
```

支撐 `src/baselines/random_baseline.py` 把 `apply_min_words` 改為 `False` 的決策
（PR #11 review "Blocking 1"）。取代舊有的 400-row 量測 —— PR #11 review 發現
`validation_4576`（`source_capacity_words=244`、`min_words_relaxed=False`）在
skip-tolerant selector 下對 seed 0 與 42 都失敗,與舊 docstring「四個 seed、400
篇零失敗」的說法矛盾。

**已重現的輸出**（全 5,621 篇 validation split，2026-08-03）：

| selector | seed 0 | seed 1 | seed 42 | seed 9999 |
|---|---|---|---|---|
| 樸素 stop-at-first-miss（結構等同 Lead 的停止規則，只是換成隨機排列） | 2.60% (146) | 2.51% (141) | 2.38% (134) | 2.31% (130) |
| skip-tolerant（`random_baseline.py` 實際實作的 `_select_random`） | 0.07% (4) | 0.04% (2) | 0.05% (3) | 0.05% (3) |

`validation_4576` 在 skip-tolerant selector 下四個 seed **全部**失敗
（實際選到 180-191 字，`effective_min_words=200`）—— 這一列在
`tests/fixtures/multi_news_validation_diagnostic_sample.jsonl` 中被單獨釘住
（見 `tests/test_baselines_random.py`）。

Pool 與選中句子的字數分布（同一次全量重跑）：

| | 平均字數/句 | 平均選中句數/篇 |
|---|---|---|
| Eligible pool（全部候選句） | 21.55 | -- |
| 選中（seed 0/1/42/9999） | 18.83 / 18.75 / 18.74 / 18.82 | 13.11 / 13.16 / 13.17 / 13.12 |

> skip-tolerant selector **不是零失敗**（舊 docstring 的說法在全量下不成立），
> 且系統性偏好較短句子（選中平均字數 < pool 平均字數）——見
> `random_baseline.py`docstring 的 "NAMING HONESTY" 一節。

---

## `length_matched_lead.py` — 長度括弧（F-18b）

比系統多用字數就可能贏 R-1／R-Lsum，這正是稽核批評舊稿的那一點。句子粒度使精確等長不可能，所以**兩側都要報**：

```bash
python -m scripts.audit.length_matched_lead \
  --data data/processed/multi_news_validation_canonical.jsonl \
  --pred <run>/predictions.jsonl \
  --out_dir <run>/length_bracket
```

輸出 `lead_undershoot.jsonl`（≤ 系統字數）與 `lead_overshoot.jsonl`（≥ 系統字數），各自以 `src.pipeline.evaluate` 評分。

**已重現的輸出**（`greedy + length_normalized`，5,613 篇，2026-08-03）：

| | R-1 | R-2 | R-Lsum | 字/篇 |
|---|---|---|---|---|
| Lead（不足） | 0.4324 | 0.1460 | 0.3931 | 229.4 |
| Lead（250 字預算） | 0.4333 | 0.1468 | 0.3941 | 233.6 |
| **系統** | 0.4347 | 0.1354 | 0.3960 | 244.0 |
| Lead（超過） | **0.4354** | **0.1495** | **0.3965** | 258.8 |

> R-1 與 R-Lsum 隨字數單調遞增，系統位置對應其字數 —— **領先由長度解釋，不是選句品質**。

---

## `selection_overlap.py` — 選句重疊率（F-18e）

```bash
python -m scripts.audit.selection_overlap \
  --a <system>/predictions.jsonl \
  --b runs_v2/gate2_lead_document_order_validation/predictions.jsonl
```

以 `sentence_id` 比對（不受排序影響）。**已重現**（全量，2026-08-03）：greedy+`mean` 27.5%、greedy+`length_normalized` 24.3%、NSGA-II+`mean` 27.6%。

> ⚠️ legacy 的 **61.7%** 來自不同 split、200 篇抽樣、test-tuned artifact，**方向可比、數值不可相減**。

---

## `paired_run_intersection.py` — 共同 feasible denominator（F-17/F-18）

正式 primary evaluation 必須計分 all rows；本工具只產生跨方法共同 feasible
intersection 的 paired sensitivity，不能取代 primary。它會拒絕 duplicate IDs、
未知 gold IDs、缺列的 post-F-17 artifact、混合 schema 與未明示的 evaluator protocol，
並在 `intersection_report.json` 保存 input SHA-256、每個 run 自身的 infeasible
IDs/reasons、legacy 缺列，以及純粹為了配對而排除的 feasible IDs。

```bash
python -m scripts.audit.paired_run_intersection \
  --pred <run-a>/predictions.jsonl <run-b>/predictions.jsonl \
  --gold data/processed/multi_news_validation_canonical.jsonl \
  --out_dir runs_v2/audit/<name> \
  --protocol multisentence_lsum
```

只有 pre-F-17 artifact 才可另加 `--assume-legacy-feasible`；這是未驗證假設，
產出的數字維持 diagnostic。

---

## 與文件的對應

| 腳本 | 支撐的結論 | 文件位置 |
|---|---|---|
| `lead_vs_system.py` | F-0 系統未贏 Lead | `CODE_AUDIT_IEEE_Access.md` F-0 |
| `length_matched_lead.py` | 長度括弧：領先由字數解釋 | `CODE_AUDIT_IEEE_Access.md` F-18(b) |
| `selection_overlap.py` | 漏斗打開但未轉化為品質 | `CODE_AUDIT_IEEE_Access.md` F-18(e) |
| `paired_run_intersection.py` | 共同 feasible denominator 與 per-example scores | `CODE_AUDIT_IEEE_Access.md` F-17/F-18 |
| `selection_diagnostics.py` | 病因：像昂貴版 Lead | `STRATEGY_ASSESSMENT.md` §1.2 |
| `dataset_headroom.py` | 主場資料集選擇 | `STRATEGY_ASSESSMENT.md` §1.1 / §2 |
| `plm_timing.py` | F-4 計時是載入 overhead | `CODE_AUDIT_IEEE_Access.md` F-4 |
| `random_baseline_min_words.py` | Random baseline `apply_min_words=False` 決策 | `src/baselines/random_baseline.py` 模組 docstring |

---

## Matched selector pilot（2026-08-06）

先凍結 reference-blind manifest；凍結後才可跑品質比較：

```bash
python -m scripts.audit.freeze_selector_pilot \
  --input data/processed/multi_news_validation_canonical.jsonl \
  --output configs/pilot_manifests/multinews_selector_pilot_v1.json \
  --sample_size 200 \
  --salt multinews-selector-pilot-v1

python -m scripts.audit.run_selector_comparison \
  --input data/processed/multi_news_validation_canonical.jsonl \
  --config configs/selector_comparison_multinews.yaml \
  --manifest configs/pilot_manifests/multinews_selector_pilot_v1.json \
  --output_dir runs/selector_pilot_v1 \
  --methods greedy mmr nsga2 \
  --nsga_seeds 2024 \
  --bootstrap_resamples 10000
```

Runner 先驗 full 5,621-row frozen policy，再只取 manifest IDs；逐方法輸出完整
predictions、per-example ROUGE、平均字／句數，並強制檢查 candidate、salience、
similarity、coverage fingerprints 完全一致。pilot 的單一 NSGA-II seed 只供方向
判斷；正式結論仍須至少五個預先固定 seeds 與 full validation。
