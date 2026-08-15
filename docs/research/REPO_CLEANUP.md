# 專案整理計畫

> 對應 `ACTION_PLAN.md` 的 Phase 0。以下是提案，不是已授權的刪除清單；任何 move/delete/tag 前都要先核對 dirty worktree、legacy 重現需求與精確目標。
> 初始盤點數字保留為 2026-07-26 snapshot；整理狀態覆核至 2026-08-15，應以 §7 為準。
> 本文件不是研究實驗排程；目前 E1～E3 比剩餘低優先度 cleanup 更重要。

---

## 結論先講：你的程式碼**沒有你想的那麼糟**

我實際量過了：

| 項目 | 數量 | 評價 |
|---|---|---|
| git 追蹤檔案 | **231** | ✅ 很乾淨。`.venv` / `node_modules` **沒有**被追蹤 |
| **`.git` 大小** | **3.2 MB** | ✅ 版本庫非常小，代表歷史沒有塞大檔 |
| `src/` 核心程式 | 397 KB，約 **2,000 行** | ✅ 模組化合理，不是義大利麵 |
| `.gitignore` | 有，且正確 | ✅ |

磁碟佔用（都是本機檔案，**沒有進版本庫**）：

| 目錄 | 大小 | 處理 |
|---|---|---|
| `.venv` | 3.7 GB | 正常，不用管 |
| `data/` | 2.5 GB | 資料集，保留（`multi_news_train.jsonl` 單檔 566 MB，本研究用不到可移走） |
| `runs/` | **1.2 GB** | ⚠️ 值得壓縮歸檔，且內容已污染 |
| `frontend/` | 97 MB | 多為 `node_modules`，移出研究主線 |
| configs | 112 個，其中 **100 個在 `_legacy_archive`** | ⚠️ 89% 是垃圾但已隔離 |
| runs | 253 個目錄，其中 **237 個在 `_archive_2026_01_15`** | ⚠️ 94% 是垃圾但已隔離 |
| scripts | 40 個，其中 **35 個在 `_archive`** | ⚠️ 88% 是垃圾但已隔離 |

**真正的問題不是「垃圾多」，而是「分不出哪些結果是有效的」。**

`runs/tuning_experiments/` 底下那 11 個看起來很正式的 run，全部是 **test set 調參**的產物 ——
它們比垃圾更危險，因為它們看起來可信、而且已經被寫進論文了。

**所以整理的重點是「標記有效性」，不是「刪檔案」。**

---

## 1. 最重要的一件事：標記污染結果 🔴

```bash
cd metaheuristic-summarization
```

建立 `runs/README.md`：

```markdown
# ⚠️ 這個目錄的所有結果都不得用於 IEEE Access 投稿

原因：
- runs/tuning_experiments/ 底下 11 個 run 全部在 multi_news_test.jsonl（test set）上執行，
  論文再從中挑選 best config → test-set overfitting
- runs/full_100、full_benchmark_result 使用單序列 ROUGE-L；依目前多句內部協定應另算 ROUGE-Lsum，但 published-protocol parity 仍需驗證
- 所有 CNN/DM 相關 run 使用 13,368 筆 validation，非官方 11,490 筆 test

保留原因：response letter 需要重現舊數字。
新結果一律寫入 runs_v2/。
```

同時打 tag：

建議在確認後，以已核對的 commit 建立 annotated tag；不要標記目前 dirty working tree：

```bash
git tag -a legacy_ict_express 1b9fe6f -m "Legacy ICT Express code and results"
```

---

> 🚫 **不要歸檔或刪除 `scripts/audit/`** —— 那是新版本化的稽核診斷腳本
> （`lead_vs_system` / `selection_diagnostics` / `dataset_headroom` / `plm_timing`），
> 是 `CODE_AUDIT_IEEE_Access.md` 與 `STRATEGY_ASSESSMENT.md` 數字的重現依據。
> 要壓縮歸檔的是 `scripts/_archive/`，兩者不同。

---

## 2. 疑似死碼：先歸檔，不直接刪除

| 檔案 | 理由 |
|---|---|
| `src/pipeline/build_features.py` | 靜態搜尋未見引用，且與 `feature_builder.py` 功能重複；仍需檢查 CLI／歷史重現後才可歸檔 |
| `notebooks/` | 空目錄 |
| `scripts/__pycache__/` | 快取 |

建議先移入明確的 legacy archive，確認正式 pipeline 與舊結果重現都不依賴後，再另行決定是否刪除。

---

## 3. 與論文無關、應移出研究主線

這些不該出現在投稿用的公開 repo（研究主計畫 §4.2 也要求研究與產品分離）：

| 目錄 | 內容 | 建議 |
|---|---|---|
| `frontend/` | React demo | 移到另一個 repo，或投稿時排除 |
| `backend/` | FastAPI + Flask legacy | 同上。`flask_legacy.py` 是最大的單一原始碼檔（11.7 KB） |
| `experimental/` | BART/Pegasus 抽象式摘要、cross-encoder rerank | 與 extractive 主線無關，歸檔 |

> ⚠️ 不要直接刪 —— 這些是你的成果，只是不屬於這篇論文。移到 `_not_in_paper/` 或另開 repo。

---

## 4. 已隔離但可以再壓縮的

> ✅ **2026-07-29 覆核：版本庫隔離目標已達成；本機壓縮／移出工作區仍未執行。**
> `configs/_legacy_archive/`、`scripts/_archive/`、`runs/_archive_2026_01_15/`、
> `data/processed/_archive_legacy/` 已透過 `.gitignore` 的 `**/_archive*/`、`**/_legacy_*/`
> 規則**從版本庫排除**（repo 由 250 檔降到 131 檔）。
> 本機檔案完全未動，仍可隨時取用。下面的壓縮建議只影響本機磁碟空間，優先度低。

這些已經在 `_archive` 裡了，優先度低，有空再處理：

- `configs/_legacy_archive/`（100 個 yaml，含 `base_k20__1` ~ `__6` 這種明顯的重複試錯）
- `runs/_archive_2026_01_15/`（237 個目錄）
- `scripts/_archive/`（35 個檔案）
- `data/processed/_archive_legacy/`（**注意：SciTLDR 與 CNN/DM 資料在這裡，還會用到，不要刪**）

建議：整包封存並產生 checksum；是否從工作區移除需另行確認。

```bash
# 範例，執行前先確認
tar -czf _archive_20260726.tar.gz configs/_legacy_archive runs/_archive_2026_01_15 scripts/_archive
```

---

## 5. 必修的小問題

### `requirements.txt` 重複宣告、有效約束不清

```
pymoo==0.6.1.1          # 第 3 行
...
pymoo>=0.6.0            # 第 28 行  ← 重複；與 0.6.1.1 相容但會混淆 lock intent
scikit-learn>=1.2       # 第 12 行
scikit-learn>=1.3.0     # 第 27 行  ← 重複；實際有效下限為 1.3.0
```

- [x] 合併重複項，鎖定單一版本（2026-07-26：`pymoo` / `scikit-learn` 各只剩一條）
- [x] 補上 `pytest`（已在依賴與 CI；2026-08-11 本機 459 passed，PR #17 Linux CI 454 passed／5 skipped）
- [x] 拆出 `requirements-demo.txt`（fastapi / flask / uvicorn）與 `requirements-ci.txt`（輕量 CI）
- [x] 補上 `nltk`（canonical Multi-News 分句已使用 Punkt）
- [ ] 補上 `bert-score`（Phase 5 語意指標才需要）
- [ ] 產生 lockfile（`pip freeze > requirements.lock.txt`）

### 其他

- [x] `src/data/preprocess_scitldr.py` —— multi-reference 已改為 canonical list（2026-07-26）
- [ ] `_minmax_norm`（`fast_fused.py`）常數輸入回傳 0.0，`_minmax_normalize`（`compose.py`）回傳 0.5 —— 行為不一致
- [ ] `src/features/semantic.py` 的 `centrality` 與 `novelty` 數學上完全反相關（`centrality_norm = 1 - novelty_norm`），
      同時給獨立權重是退化的；且 `centrality` 錯誤地含對角線自身相似度
- [ ] **公開 repo 的授權／資料再散布決策**：目前 GitHub repository 是 public 且沒有
      LICENSE；Multi-News dataset card 標示 license 為 `other`。在 tests 中提交任何原始新聞
      文字 fixture 前，必須先確認可再散布依據，或改用結構等價的 synthetic／生成式 fixture，
      不能因檔案很小就默認可公開。

---

## 6. 建議的目標結構

```
metaheuristic-summarization/
├── CLAUDE.md                 → 在專案根目錄（上一層）
├── src/
│   ├── data/                 分句、split、schema 驗證、fingerprint
│   ├── candidates/           ★ 新增：lead / statistical / graph / plm 各 route 獨立排名
│   ├── fusion/               ★ 新增：score calibration、provenance union、routing
│   ├── selection/            greedy / GRASP / NSGA-II + Pareto 選解policy
│   ├── eval/                 rouge（已修）、oracle（已加）、bertscore、統計
│   └── baselines/            ★ 新增：lead / textrank / lexrank / pacsum / sbert / llm
├── configs/                  只留現行的，其餘壓縮歸檔
├── runs_v2/                  ★ 新增：只放重構後的有效結果
├── runs/                     舊結果，README 標明 invalid
├── tests/                    補齊 golden tests
└── _not_in_paper/            frontend / backend / experimental
```

---

## 7. 執行順序與目前進度

> ✅ 已完成 = 2026-07-26 實際驗證過；⬜ = 尚未做。

| # | 動作 | 狀態 |
|---|---|---|
| 1 | 🔴 寫 `runs/README.md` 標記污染結果 | ✅ 已完成（2026-07-26；分三類說明失效原因，含 smoke run 規模不足） |
| 2 | 對 legacy commit `1b9fe6f` 建 annotated tag | ✅ 已完成（`legacy_ict_express`，tag message 列出四項失效原因） |
| 3 | 疑似死碼（`src/pipeline/build_features.py`）移入 archive | ⬜ 尚未做，檔案仍在 |
| 4 | 修 `requirements.txt` + pytest | ✅ 已完成 |
| 5 | `frontend` / `backend` / `experimental` 移到 `_not_in_paper/` | ⬜ 尚未做（README 已標明可略過，影響已降低） |
| 6 | 歸檔 `_legacy_archive` / `_archive_2026_01_15` / `scripts/_archive` | ✅ **已由 `.gitignore` 的 `**/_archive*/`、`**/_legacy_*/` 排除於版本庫外**，本機檔案保留 |
| 7 | 退役過時 `docs/*.md` 至 `docs/_legacy_docs/` | ✅ 已完成（9 份） |
| 8 | 清除 config 中硬編碼的使用者絕對路徑 | ✅ 已完成（14 個檔案） |

**剩下的第 3、5 項優先度低**：`src/pipeline/build_features.py` 是零引用死碼，
`frontend` / `backend` / `experimental` 已在 README 標明可略過。兩者都不影響研究正確性，
可等 Phase 1 收尾再處理。

✅ `runs/README.md` 與 `runs_v2/README.md` 已明確區分 legacy invalid artifacts 與
governed evidence registry；新舊分界已完成。研究現行狀態以 `runs_v2/README.md`、
`ACTION_PLAN.md` 與 claim matrix 為準。

> 全部做完約 1 天。**第 1 步不要跳過** —— 那 11 個污染的 run 是目前最危險的東西。
