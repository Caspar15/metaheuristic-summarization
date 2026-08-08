# Gate 2 baseline 狀態報告

## 2026-08-09 checkpoint：GovReport PLM 27/27 完成

- 事前註冊的 GovReport PLM family 已完成 **27/27**，final failure 0。第一次外層命令
  完成 22 個 candidate 後中斷；PacSum-SBERT P06 partial 已依 F-48 封存並登錄 failed
  attempt，resume 後 P06～P10 全部完成。
- family winner 是 full-source SBERT-MMR λ=`0.9`：R-1/R-2/R-Lsum
  `0.575010/0.241576/0.541718`，macro `0.452768`。它在三項 point estimate 都高於
  non-PLM LexRank，macro 只高 `0.001148`；尚未 paired bootstrap，不能宣稱顯著勝出。
- SBERT centroid macro `0.451441`，只低 LexRank `0.000179`；最佳 PacSum-SBERT
  beta=`0.5` macro `0.451099`。這表示 GovReport 的 full-source semantic ranking 很強，
  但目前多種強 baseline 彼此非常接近，必須以逐篇 paired inference 決定 finalist。
- winner 比 proposed S02b macro 高 `0.034906`，三項都較高；因此 S02b 目前仍未通過
  strongest-completed-baseline gate。這不直接觸發 semantic route 刪除，因為 full-source
  external baseline 與方法內 capacity-matched route ablation 回答不同問題。
- F-53 驗證 27×681=`18,387` row accesses：680 個 unique cold `miss_written`、17,707 hits。
  第一個 run 內有一筆內容 key 重複而直接命中，所有 run 仍各覆蓋 681 rows；contract、
  ordered-key digest 與 PLM dependencies 一致。dev-test/test 未讀。

## 2026-08-09 checkpoint：Multi-News PLM 27/27 完成

- 事前註冊的 Multi-News PLM family 已完成 **27/27**：SBERT centroid 1、full-source
  SBERT-MMR 5 個 λ、PacSum-SBERT 21 個 OFAT 點。沒有 final candidate failure。
- family winner 是非退化 `pacsum_sbert_P03_previous_-0.3`：R-1/R-2/R-Lsum
  `0.442111/0.150742/0.401520`，macro `0.331458`。它比 proposed S02b macro 高
  `0.003381`，但仍比 non-PLM PacSum TF-IDF P08 低 `0.000282`。這只是 dev point
  estimate；差距尚未做 paired inference，不能宣稱 TF-IDF 或 SBERT representation 勝出。
- 最佳 full-source SBERT-MMR 是 λ=`0.7`，macro `0.322581`，比 proposed S02b 低
  `0.005496`；SBERT centroid macro `0.316024`。這類 full-source baseline 與方法內
  candidate-matched selector swap 是不同實驗，不能用 200-row matched pilot 的 MMR
  優勢覆蓋本結果。
- 原 `sbert_mmr_lambda_0.3` 中斷已依 F-48 封存為 `attempt_01_interrupted` 並登錄失敗；
  final retry 另存且成功，不覆寫歷史。
- F-51 cache 的全量 exact audit 先於 cached rerun 完成。後續 25 個 run 共驗得
  98,375 hits、相同 ordered-key digest 與完整 PLM dependency versions；最初兩個 uncached
  run 明列 legacy。F-53 family verifier 對列數、狀態、contract、digest 與版本 fail loud。
- 本 checkpoint 沒有讀 dev-test 或 test，也沒有依中途分數刪減網格。兩資料集 PLM
  現均已完成；Gate 2 仍未通過，因兩 primary greedy reference 與完整 paired inference 尚未完成。

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
  CLI 沒有 partition 參數。目前加入 F-51～F-55 guards 後完整本地回歸為 **392 passed**。

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
- 目前只有 point estimates。兩 primary PLM 已完成；尚未完成 greedy reference、
  Multi-News clean sensitivity、完整 paired bootstrap／selection correction，因此
  **Gate 2 仍未通過，也不讀 dev-test**。

### 證據

- 預註冊：`configs/preregistrations/gate2_baseline_matrix_v1.json`
- family 原始彙整：`runs_v2/gate2_baseline_matrix_v1/multinews/dev/non_plm/family_summary.json`
- 驗證後解讀：`runs_v2/gate2_baseline_matrix_v1/multinews/dev/non_plm/analysis_summary.json`
- GovReport family：`runs_v2/gate2_baseline_matrix_v1/govreport/dev/non_plm/family_summary.json`
- GovReport 解讀：`runs_v2/gate2_baseline_matrix_v1/govreport/dev/non_plm/analysis_summary.json`
- Multi-News PLM family：`runs_v2/gate2_baseline_matrix_v1/multinews/dev/plm/family_summary.json`
- Multi-News PLM 解讀：`runs_v2/gate2_baseline_matrix_v1/multinews/dev/plm/analysis_summary.json`
- GovReport PLM family：`runs_v2/gate2_baseline_matrix_v1/govreport/dev/plm/family_summary.json`
- GovReport PLM 解讀：`runs_v2/gate2_baseline_matrix_v1/govreport/dev/plm/analysis_summary.json`
- runner：`scripts/audit/run_gate2_baseline_matrix.py`
- family verifier：`scripts/audit/summarize_gate2_baseline_family.py`

### 尚未完成（下一步）

1. 兩 primary 的 metric-specific greedy reference 與既有 Lead／Random integrity reuse。
   `gate2-greedy-reference-v1` 已在分數前凍結，governed runner 與 6 個 configs 待執行。
2. 全 baseline finalists 與 proposed candidates 的 paired inference、headroom 與成本比較；
   GovReport MMR 與 LexRank 的 `0.001148` 差距在推論前只能稱 point estimate。
3. 針對 Multi-News 仍輸 strongest completed baseline、GovReport 仍輸 strongest completed
   baseline 的現況，在 dev 做已預註冊的 selector／salience
   搜尋；若搜尋空間耗盡仍無顯著優勢，依停止條件寫重新定位建議。
4. 只有完成上述項目、決定 proposed selector／route 後，才可依事前規則做一次 dev-test；
   test 仍鎖定至 freeze 簽字。
