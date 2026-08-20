# E3 route/provenance ablation status

## Multi-News final completion（2026-08-20）

Multi-News final E3 使用 frozen dev 3,935 rows、100,000 paired bootstrap、固定
base seed 20260920，並對 5 variants × 4 metrics 做 Holm-20。正式 parity run 與 frozen
anchor 的 Python／NumPy／scikit-learn／NLTK／ROUGE／PyYAML／Torch／Transformers／
SentenceTransformers 版本完全一致；dev-test/test 均未存取。

| Variant | Macro | Full − variant macro | 95% CI | Holm-20 p |
|---|---:|---:|---:|---:|
| **C01 full proposed** | **0.330417** | — | — | — |
| A01 no semantic | 0.328500 | +0.001917 | [+0.001149,+0.002685] | 0.000400 |
| A02 no graph | 0.326033 | +0.004385 | [+0.003466,+0.005311] | 0.000400 |
| A03 lexical only, capacity 80 | 0.319939 | +0.010478 | [+0.009364,+0.011618] | 0.000400 |
| A04 exact C01 pool, equal-weight RRF | 0.329675 | +0.000742 | [+0.000135,+0.001353] | 0.049440 |
| A05 exact C01 pool, lexical selector salience | 0.323278 | +0.007139 | [+0.006188,+0.008099] | 0.000400 |

五項預註冊 macro claims 全部通過。A04 位於 Holm 門檻邊緣，論文只能寫成小幅、
task-specific weighting effect；A05 則直接證明 route provenance 必須進 selector，
不能只決定候選池 membership。最初一套完整 attempt 使用 scikit-learn 1.5.1／NLTK
3.9.1／Torch 2.5.1，和 anchor 不同；A02/A04 selected indices 確實變動，因此該 attempt
完整保留但排除。runner 現會在任何分數前 fail-loud 檢查 dependency parity。

正式 evidence：`runs_v2/multinews_e3_route_provenance_v1/`。以下 GovReport 結果仍有效。

## 結論（2026-08-15）

E3 已完成。五個 variants 都是在任何 E3 分數前凍結，僅使用 GovReport frozen dev
681 rows；每案 681/681 feasible，dev-test/test 均未存取。完整 C01 相對五個消融案的
macro 差異，其 95% paired-bootstrap CI 都全正，20-endpoint Holm 校正後皆顯著；因此
semantic、graph、capacity-matched non-lexical routes、weighted route fusion，以及把
route provenance 真正送入 selector 的五項預註冊主張均未觸發降級條件。

## 內部 evaluator 結果

| Variant | R-1 | R-2 | R-Lsum | Macro | C01 − variant macro | 95% CI | Holm-20 p |
|---|---:|---:|---:|---:|---:|---:|---:|
| **C01 full proposed** | **0.579106** | **0.249380** | **0.543725** | **0.457404** | — | — | — |
| A01 no semantic | 0.564764 | 0.232524 | 0.528778 | 0.442022 | +0.015382 | [+0.013057,+0.017719] | 0.000400 |
| A02 no graph | 0.573256 | 0.234656 | 0.537198 | 0.448370 | +0.009034 | [+0.006964,+0.011131] | 0.000400 |
| A03 lexical only, capacity 80 | 0.485142 | 0.146490 | 0.446759 | 0.359463 | +0.097940 | [+0.093473,+0.102404] | 0.000400 |
| A04 exact C01 pool, equal-weight RRF | 0.573253 | 0.239399 | 0.536172 | 0.449608 | +0.007796 | [+0.006154,+0.009420] | 0.000400 |
| A05 exact C01 pool, lexical selector salience | 0.486286 | 0.144090 | 0.448332 | 0.359569 | +0.097834 | [+0.093306,+0.102323] | 0.000400 |

本表使用預註冊的內部 Google `rouge_score` multi-sentence Lsum protocol，用途是 matched
ablation。不可把它與 E1 的 Stanza + Perl ROUGE-1.5.5 官方尺度混成同一張勝負表；E1
完整方法官方 macro 是 `0.458257`。

## 如何解讀

- A01/A02 顯示，在凍結的 GovReport 架構中，semantic 與 sparse graph 各自都有獨立
  quality contribution；semantic removal 的下降較大，但不能據此聲稱 universal ranking。
- A03 已把 lexical-only pool 容量補到 80，仍大幅落後，排除「完整方法只因候選數量較多」
  的主要替代解釋。
- A04 固定每篇與 C01 完全相同的 candidate membership，只把 weighted RRF 改成等權；
  顯著下降支持 route weighting 的作用。
- A05 也固定完全相同的 C01 candidate membership，但 selector 只吃 lexical percentile；
  結果幾乎退化到 A03。這直接證明 route provenance 必須影響 selector salience，不能只把
  semantic/graph 當作「進池資格」。
- 以上是 GovReport 長篇單文件的 frozen-dev component evidence，不外推到 Multi-News，
  也不是 untouched-test 結果。

## 執行與失敗紀錄

- 正式執行：16 bounded workers、每 worker 一個 BLAS thread；五案各約 73–83 秒。
- 16-worker process-tree peak RSS 約 3.22–7.26 GB；這是整個平行研究 job 的峰值，不是
  單篇部署記憶體。單篇 controlled cost 以 E2 為準。
- attempt 01 在任何 worker／分數前因 runner 讀錯 preregistration schema 而 `KeyError`；
  修正只改 manifest identity guard，另加三個 regression tests。
- attempt 02 因 Windows sandbox 禁止 worker pipe 而在任何 prediction／分數前失敗；
  相同 frozen config 在 sandbox 外成功重跑。兩個 failures 都保留並寫入 search log。

## Evidence

- Analysis：`runs_v2/govreport_e3_route_provenance_v1/analysis.json`
- Study summary：`runs_v2/govreport_e3_route_provenance_v1/study_summary.json`
- Per-variant evidence：`runs_v2/govreport_e3_route_provenance_v1/dev/`
- Failed attempts：`runs_v2/govreport_e3_route_provenance_v1/attempts/`
- Execution addendum：`configs/preregistrations/govreport_e3_execution_addendum_v1.json`

## 2026-08-15 當時尚未完成（歷史）

- ICACT 六頁技術內容已逐頁核對；正式 DOI／獎項證明／similarity report 仍待投稿前補齊。
- Pre-test freeze audit 已完成並發現 frozen-policy ordering conflict；GovReport test data
  policy、exact execution package 與老師／完整作者群兩階段簽字仍未完成。

E3 從未用來自動解鎖 test，也沒有授權新增 dev 搜尋。兩資料集 final test 其後依各自
human-authorized frozen one-shot protocol 完成；E3 結論不得回頭改配置。
