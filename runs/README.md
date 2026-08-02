# ⚠️ 這個目錄裡沒有任何可用於論文的結果

截至 2026-08-02，`runs/` 現存的 `metrics.csv` 與 `predictions.jsonl` **都不能作為新論文的證據**。
理由分三類，每一類的失效原因不同。

> 這個目錄本身已被 `.gitignore` 排除（只有這份 README 進版本庫）。
> 保留本機檔案是為了重現舊 artifact 與寫 response letter，不是為了引用數字。

---

## 1. 🔴 test-set 調參污染 —— `tuning_experiments/`

19 個子目錄，其中 11 個主要 run **全部在 5,622 筆 Multi-News test set 上執行**，
舊稿再從這些輸出裡挑出「best configuration」（Coverage, K=20）。

**這是 test-set overfitting。** 即使沒有梯度訓練，只要用 test 結果選 K、權重或 objective，
test 就已經被用於模型選擇，那個 split 對這批設定而言已經燒掉了。

產生它們的腳本是 `scripts/quick_tune*.ps1` 與 `run_missing_experiments.ps1`
（現已加上 guard，不帶 `-IAcknowledgeTestSetTuning` 會拒絕執行）。

代表性數字（**僅供診斷，不可引用**）：

| run | R-1 | R-2 | R-L (舊協定) |
|---|---|---|---|
| `ExpB_K20_Max_Coverage` | 0.4352 | 0.1405 | 0.2019 |

> 這組正是舊稿 Table 7 的數字。用同一 evaluator 在本地重跑的 Lead prefix 是
> `0.4331 / 0.1453 / 0.3901` —— **系統在 R-2 與 R-Lsum 上並未勝出**。
> 詳見 `docs/research/CODE_AUDIT_IEEE_Access.md` 的 F-0。

---

## 2. 🟠 評測協定已失效 —— `full_100/`、`full_benchmark_result/`、`sanity_check/`

這些 run 的 `metrics.csv` 記的是**單序列 `rougeL`**，不是多句摘要適用的 `rougeLsum`。
同一批 prediction 換成現行協定後：

| artifact | 舊 `rougeL` | 現行 `rougeLsum` |
|---|---|---|
| `full_benchmark_result` | 0.2014 | **0.3857** |
| `ExpB_K20_Max_Coverage` | 0.2019 | **0.3880** |

**這是 metric 定義改變，不是模型變好。** 而且這幾個目錄已無 `predictions.jsonl`，
無法重算，只能當歷史紀錄。

⛔ 右欄兩個 `rougeLsum` 值**連當歷史紀錄都要加注**（2026-07-30, PR #9）：
它們是在舊的 regex 分句器下算的，現行 evaluator 已改用共用 Punkt tokenizer，
實測位移約 **+0.003**。因為 `predictions.jsonl` 已不存在，**這兩個數字永遠無法在
新協定下重算**——任何情況下都不得寫進論文。

另外所有 CNN/DailyMail 相關的舊 run 使用 **13,368 筆 validation**，
不是官方 11,490 筆 test，因此不能與文獻 test 數字並排比較。

---

## 3. 🟡 規模不足的 smoke run —— `phase1-*-smoke-*/`

| run | 筆數 |
|---|---|
| `phase1-provenance-smoke-20260726` | **3** |
| `phase1-route-smoke-20260726` | **3** |
| `phase1-objective-smoke-20260726` | **3**（diagnostic：`min_words=0` 退化為單句） |
| `phase1-objective-min200-smoke-20260726` | **3**（200–250 word feasibility smoke） |
| `phase1-fullsource-objective-smoke-20260726` | **3**（full-source coverage correctness smoke） |
| `phase1-smooth-tfisf-smoke-20260727` | **3**（non-negative smoothed TF-ISF v2 smoke） |

這些是 Phase 1 重構的**接線／constraint 驗證**（confirming provenance reaches the selector,
routes rank the full input, shared objectives respect the declared length band）。**3 篇文件的 ROUGE 沒有任何統計意義**，
它們的存在只證明 pipeline 跑得通，不證明任何方法有效。

---

## 4. `_archive_2026_01_15/`

更早期的探索紀錄，同時具備上述多種問題。純歷史。

---

## 新結果要寫到哪裡

依 `docs/research/ACTION_PLAN.md`：

- 新的 governed validation 結果寫入 **`runs_v2/`**（尚未建立／追蹤）
- test split 只有在 configuration freeze 之後才解鎖，且只跑一次
- 每個正式 run 必須附 data fingerprint、effective config hash、commit、seed 與 manifest

PR #10 已提供 Lead baseline CLI，但其 `--run_dir` 預設仍是 `runs`。這個預設值不是
研究有效性的保證；正式 Phase 2 命令必須明確傳 `--run_dir runs_v2`。即使某個新檔案
意外寫進 `runs/`，也必須依 artifact 的 policy/preflight/config/commit 身分判定，不能
只看目錄名稱把它升格為有效結果。現在尚無任何兩個-primary完成的正式 Lead 結果。

## 引用數字前先確認三件事

1. 這個 run 的設定是在 **validation** 上選的，還是在 test 上選的？
2. 用的是哪一個 **evaluation protocol**（`multisentence_lsum`？舊 `rougeL`？）
3. 有沒有做 **paired significance test**？—— 目前 repo 裡**還沒有**統計模組，
   所以任何「贏了多少」的敘述都尚未經檢定

三個問題有任何一個答不出來，就不要把數字寫進論文。
