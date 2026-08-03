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

## 與文件的對應

| 腳本 | 支撐的結論 | 文件位置 |
|---|---|---|
| `lead_vs_system.py` | F-0 系統未贏 Lead | `CODE_AUDIT_IEEE_Access.md` F-0 |
| `selection_diagnostics.py` | 病因：像昂貴版 Lead | `STRATEGY_ASSESSMENT.md` §1.2 |
| `dataset_headroom.py` | 主場資料集選擇 | `STRATEGY_ASSESSMENT.md` §1.1 / §2 |
| `plm_timing.py` | F-4 計時是載入 overhead | `CODE_AUDIT_IEEE_Access.md` F-4 |
| `random_baseline_min_words.py` | Random baseline `apply_min_words=False` 決策 | `src/baselines/random_baseline.py` 模組 docstring |
