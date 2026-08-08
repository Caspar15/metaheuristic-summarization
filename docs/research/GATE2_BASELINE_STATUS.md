# Gate 2 baseline 狀態報告

## 2026-08-09 checkpoint：Multi-News PLM 2/27 與 execution-cache safety gate

- Multi-News PLM family 已完成 `sbert_centroid` 與 `sbert_mmr_lambda_0.1`，即 **2/27**；
  macro ROUGE 分別為 `0.316024`、`0.294505`。這是 dev point estimate，不是 finalist。
- `sbert_mmr_lambda_0.3` 在 family 命令達 3,600 秒外層限制時中斷；partial run 保留，
  下一次 `--resume` 必須先依 F-48 封存為 failed attempt 並寫入 `search_log.jsonl`。
- 觀察到每個 PLM candidate 重複編碼完全相同的 frozen-dev inputs（F-51）。已實作
  execution-only content-addressed cache，相關測試與完整回歸為 **382 passed**；但全量
  3,935-row cached-vs-uncached `selected_indices` 等價驗證尚未完成，因此目前不可續跑。
- 本 checkpoint 沒有讀 dev-test 或 test，也沒有依已看到的分數刪減預註冊 27-candidate
  網格。Gate 2 仍未通過。

## 2026-08-09 checkpoint：兩 primary non-PLM frozen-dev 完成

### 已完成

- `gate2-baseline-matrix-v1` 的 Multi-News／GovReport non-PLM families 各完成 **23/23**
  個事前註冊 candidates；每個 run 只讀 frozen dev 3,935／681 rows，`dev_test_accessed=false`、
  `test_split_accessed=false`。
- TextRank、LexRank 與 clean-room PacSum TF-IDF 共用各資料集 canonical input、A1 凍結的
  200–250／500–650 words contract 與 `multisentence_lsum` evaluator。
- 23 個成功 run 與一次 P07 外層中斷都已進 `runs_v2/search_log.jsonl`。中斷 artifact
  保留於 `attempt_01_interrupted/`，resume 後的 final run 另存，不覆寫失敗紀錄。
- GovReport 23 個 runs 全部一次完成；兩 family 的成功／失敗歷史都已寫入 registry。
- family 彙整器會驗 candidate count、每個 evidence 的 partition guards、結果完整性，且
  CLI 沒有 partition 參數。修正 GovReport A1 Lead path 後完整本地回歸為 **376 passed**。

### Multi-News frozen-dev 結果

| 方法／candidate | R-1 | R-2 | R-Lsum | Macro |
|---|---:|---:|---:|---:|
| PacSum TF-IDF P08（previous −0.8 / following +0.2） | **0.442862** | **0.150064** | **0.402293** | **0.331740** |
| PacSum TF-IDF P07（−0.7 / +0.3） | 0.442335 | 0.149944 | 0.400236 | 0.330838 |
| PacSum TF-IDF beta=1.0 | 0.436959 | 0.148948 | 0.398594 | 0.328167 |
| frozen Lead（A1 reuse） | 0.435033 | 0.148139 | 0.395701 | 0.326291 |
| LexRank | 0.431726 | 0.137147 | 0.390633 | 0.319835 |
| PacSum TF-IDF default | 0.430800 | 0.141651 | 0.385065 | 0.319172 |
| TextRank | 0.415192 | 0.130218 | 0.369876 | 0.305096 |

P08 比 frozen Lead 的 macro 高 `+0.005449`。相較目前 proposed S02b
（0.440759 / 0.139803 / 0.403670，macro 0.328077），P08 的 macro 高 `+0.003663`、
R-1 高約 `+0.002103`、R-2 高約 `+0.010261`，但 R-Lsum 低約 `−0.001377`。
這表示 proposed system **目前沒有勝過已完成的最強 baseline**，不能晉級。

### GovReport frozen-dev 結果

| 方法／candidate | R-1 | R-2 | R-Lsum | Macro |
|---|---:|---:|---:|---:|
| LexRank | **0.572517** | **0.241004** | **0.541340** | **0.451620** |
| TextRank | 0.552437 | 0.223466 | 0.516994 | 0.430966 |
| proposed S02b | 0.545091 | 0.196227 | 0.512267 | 0.417862 |
| PacSum TF-IDF P07（previous −0.7 / following +0.3） | 0.536944 | 0.208192 | 0.506363 | 0.417167 |
| PacSum TF-IDF default | 0.534998 | 0.202191 | 0.500017 | 0.412402 |
| frozen Lead（A1 reuse） | 0.514532 | 0.194032 | 0.489133 | 0.399232 |

LexRank 比 frozen Lead macro 高 `+0.052388`，比 proposed S02b 高 `+0.033758`；TextRank
也比 S02b 高 `+0.013104`。S02b 僅比最佳 PacSum P07 高約 `+0.000695`。因此 proposed
在 GovReport **明確沒有勝過已完成的 non-PLM graph baselines**。LexRank 與 frozen Lead
0/681 rows 的 selected indices 完全相同，平均 Jaccard `0.059722`，不是 Lead 複製品。
單次 selection wall time 為 TextRank `547.30 s`、LexRank `768.11 s`、P07 `19.50 s`；
既有 S02b 為 `1,106.27 s`。不同 process 的單次時間不能當精確 speedup，但 S02b 同時
比 LexRank 慢且低 `0.033758`；對 P07 則約 `56.7×` 時間只換 `+0.000695` macro，
目前 quality–cost 也不支持 S02b。

### 重要解讀限制

- beta=1.0 在 Multi-News 3,935/3,935、GovReport 681/681 rows 全部
  `score_degenerate=true`：所有 centrality scores
  同分，結果其實是 frozen canonical order 加上遇到不合長度句子可跳過的 fill policy。
  它不是 directed centrality 有效的證據；P08 才是本 family 的非退化 dev winner。
- P08 仍有 1/3,935 row score-degenerate。它與 frozen Lead 只有 166/3,935 rows 的
  `selected_indices` 完全相同，平均 selection Jaccard `0.617715`，並非單純複製 Lead。
- 目前只有 point estimates。尚未完成兩資料集 PLM family、greedy reference、
  Multi-News clean sensitivity、完整 paired bootstrap／selection correction，因此
  **Gate 2 仍未通過，也不讀 dev-test**。

### 證據

- 預註冊：`configs/preregistrations/gate2_baseline_matrix_v1.json`
- family 原始彙整：`runs_v2/gate2_baseline_matrix_v1/multinews/dev/non_plm/family_summary.json`
- 驗證後解讀：`runs_v2/gate2_baseline_matrix_v1/multinews/dev/non_plm/analysis_summary.json`
- GovReport family：`runs_v2/gate2_baseline_matrix_v1/govreport/dev/non_plm/family_summary.json`
- GovReport 解讀：`runs_v2/gate2_baseline_matrix_v1/govreport/dev/non_plm/analysis_summary.json`
- runner：`scripts/audit/run_gate2_baseline_matrix.py`
- family verifier：`scripts/audit/summarize_gate2_baseline_family.py`

### 尚未完成（下一步）

1. 先完成並版本化 F-51 全量等價驗證；通過後續跑 Multi-News PLM 剩餘 25 candidates，
   再跑 GovReport PLM 27 candidates。
2. 兩 primary 的 metric-specific greedy reference 與既有 Lead／Random integrity reuse。
3. 全 baseline finalists 與 proposed candidates 的 paired inference、headroom 與成本比較。
4. 針對兩資料集都輸 strongest non-PLM baseline 的現況，在 dev 做已預註冊的 selector／salience
   搜尋；若搜尋空間耗盡仍無顯著優勢，依停止條件寫重新定位建議。
5. 只有完成上述項目、決定 proposed selector／route 後，才可依事前規則做一次 dev-test；
   test 仍鎖定至 freeze 簽字。
