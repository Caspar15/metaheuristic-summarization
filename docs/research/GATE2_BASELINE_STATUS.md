# Gate 2 baseline 狀態報告

## 2026-08-09 checkpoint：Multi-News non-PLM frozen-dev 完成

### 已完成

- `gate2-baseline-matrix-v1` 的 Multi-News non-PLM family 已完成 **23/23** 個事前註冊
  candidates；每個 run 都只讀 frozen dev 3,935 rows，`dev_test_accessed=false`、
  `test_split_accessed=false`。
- TextRank、LexRank 與 clean-room PacSum TF-IDF 共用同一 canonical input、A1 凍結的
  200–250 words contract 與 `multisentence_lsum` evaluator。
- 23 個成功 run 與一次 P07 外層中斷都已進 `runs_v2/search_log.jsonl`。中斷 artifact
  保留於 `attempt_01_interrupted/`，resume 後的 final run 另存，不覆寫失敗紀錄。
- family 彙整器會驗 candidate count、每個 evidence 的 partition guards、結果完整性，且
  CLI 沒有 partition 參數。完整本地回歸為 **375 passed**。

### frozen-dev 結果

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

### 重要解讀限制

- beta=1.0 的 3,935/3,935 rows 全部 `score_degenerate=true`：所有 centrality scores
  同分，結果其實是 frozen canonical order 加上遇到不合長度句子可跳過的 fill policy。
  它不是 directed centrality 有效的證據；P08 才是本 family 的非退化 dev winner。
- P08 仍有 1/3,935 row score-degenerate。它與 frozen Lead 只有 166/3,935 rows 的
  `selected_indices` 完全相同，平均 selection Jaccard `0.617715`，並非單純複製 Lead。
- 目前只有 point estimates。尚未完成 PLM family、GovReport family、greedy reference、
  Multi-News clean sensitivity、完整 paired bootstrap／selection correction，因此
  **Gate 2 仍未通過，也不讀 dev-test**。

### 證據

- 預註冊：`configs/preregistrations/gate2_baseline_matrix_v1.json`
- family 原始彙整：`runs_v2/gate2_baseline_matrix_v1/multinews/dev/non_plm/family_summary.json`
- 驗證後解讀：`runs_v2/gate2_baseline_matrix_v1/multinews/dev/non_plm/analysis_summary.json`
- runner：`scripts/audit/run_gate2_baseline_matrix.py`
- family verifier：`scripts/audit/summarize_gate2_baseline_family.py`

### 尚未完成（下一步）

1. GovReport non-PLM 23 candidates。
2. Multi-News 與 GovReport PLM families，各 27 candidates。
3. 兩 primary 的 metric-specific greedy reference 與既有 Lead／Random integrity reuse。
4. 全 baseline finalists 與 proposed candidates 的 paired inference、headroom 與成本比較。
5. 只有完成上述項目、決定 proposed selector／route 後，才可依事前規則做一次 dev-test；
   test 仍鎖定至 freeze 簽字。
