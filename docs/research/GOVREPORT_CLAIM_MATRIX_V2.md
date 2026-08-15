# GovReport-centered Claim Matrix v2

> **執行狀態覆核：2026-08-15。** Candidate、主張範圍與 E1～E3 protocol 已凍結；
> E1 已完成且官方尺度 primary comparison 通過；E2 cost/memory/scaling 與 E3
> route/provenance ablation 也已完成，ICACT 六頁 technical extension audit 已核對。
> Test-policy ordering、immutable test policy、exact execution package 與 final signatures
> 仍未完成。因此現在是 pre-test governance resolution，不是可執行 final test 的狀態。

## 決策狀態

- 決策時間：2026-08-10（系統 UTC 時間記錄於 machine-readable addendum）。
- 作者端方向：**選項 A 已批准**。
- 老師／完整作者群簽字：**待完成**。
- protected split：**仍鎖定**；本決策沒有授權再讀 dev-test 或 test。
- 機器可讀權威：
  `configs/data_policies/govreport_centered_repositioning_v2.json`。

這份 v2 只改變 IEEE Access 修訂稿的資料集角色、投稿主張與 promotion gate。它不修改
GovReport／Multi-News canonical bytes、既有 validation policies、partition membership、
A1 長度協定、預註冊或歷史 evidence。原本「兩 primary 都必須通過」的 D3b gate
仍誠實記為失敗；v2 是失敗後的窄化定位，不是把失敗改寫成成功。

## 最終資料集角色

| 角色 | 資料集 | 新稿用途 | 明確限制 |
|---|---|---|---|
| 唯一主要品質資料集 | **GovReport** | 驗證 long single-document、multi-sentence、training-free extractive summarization | frozen dev 的 official evaluator 已通過；test 尚未執行 |
| Boundary condition | **Multi-News** | 報告 multi-document profile 的失敗邊界 | 不再作共同 promotion gate，不新增 protected-split 搜尋，也不能隱藏 R-2 顯著失敗 |
| 未納入 | CNN/DailyMail、SciTLDR、PubMed | 本次定案不新增 | 不能為了增加資料集數量而在 freeze 後臨時加入 |

## 論文主張矩陣

| ID | 主張 | 目前可寫到哪裡 | 投稿前還缺什麼 | 失敗時怎麼辦 |
|---|---|---|---|---|
| C-GOV-QUALITY | Frozen proposed 在 GovReport 優於 full-source SBERT+MMR λ=0.9 | 可寫「frozen dev、官方 evaluator、預註冊 macro 顯著」；R-L 單項未顯著 | freeze 簽字與 one-shot final evaluation | test 若失敗即撤回 superiority；不得調參救分 |
| C-PROVENANCE | 每句保留 lexical／semantic／graph route、rank、fusion 與 selector provenance | fixed-pool A04/A05 已證明 weighted fusion 與 selector provenance 的 macro 增益均 Holm-20 顯著 | freeze 簽字與 one-shot final evaluation | test 不改 component 結論，但整體品質 claim 依 test 決定 |
| C-ROUTES | semantic／graph 對長篇單文件提供增量 | E3 A01/A02/A03 的 capacity-aware removal 均顯著；E2 成本與 memory 已量測 | freeze 簽字與 one-shot final evaluation | 僅限 GovReport profile，不外推 Multi-News |
| C-NSGA | NSGA-II 是 matched comparator，不是最佳 selector | pilot 與 full-dev 均支持負結果 | 如實報告，不需要再跑 test 搜尋 | 保留負結果／附錄，不能放回標題 |
| C-BOUNDARY | GovReport 優勢不外推到 Multi-News | frozen dev 已支持 | 摘要、限制與結論都要明說 | 不適用；這本身就是 scope 結論 |

## Frozen candidate

- Candidate：`C01_combined_salience_route_weight`
- Profile：single document + multi-sentence
- Selector：TF-IDF MMR，λ=0.7
- Length：500–650 whitespace words
- Routes：lexical + semantic + sparse graph
- Candidate budget：route top-40、每路 requested reservation 20、total 80
- Salience：weighted RRF；lexical route weight 0.5
- Config SHA-256：`c2088ca4...bd481`

Frozen dev 結果是 macro `0.457404`，相對 strongest local SBERT+MMR
`+0.004636`，95% CI `[+0.002507,+0.006745]`；這不是 official-evaluator 或 test
結果。除 selected-index-equivalent correctness fix 與預註冊 evidence instrumentation 外，
不得再依品質結果改 candidate、route weight、MMR λ、budget 或長度。

## 投稿定位

建議主句：

> 本研究提出一個免任務微調、保留逐句 provenance 的多路抽取框架，並以嚴格的
> matched-input selector comparison 說明昂貴 metaheuristic search 不必然帶來品質增益。
> 在 GovReport 長篇單文件上，凍結配置對強 no-task-training baseline 顯示正證據；
> Multi-News 的負結果則界定方法不適用於所有 multi-document summarization 情境。

不可使用的寫法：跨資料集 SOTA、全面勝過所有 extractive baselines、NSGA-II 最佳、
global optimum、已在 test 勝出，或把不同 split／evaluator 的文獻數字當本地公平勝負。

## 解鎖 final evaluation 前的 DoD

- [x] GovReport 作者官方 Stanza + Perl ROUGE-1.5.5 protocol 已版本化並完成 parity/ranking check。
- [x] Frozen candidate、SBERT+MMR 與必要 baselines 的 CPU cold/warm runtime、peak process-tree RSS、scaling 完成；GPU VRAM 不適用於本 CPU-only 主報告。
- [x] Route-removal 與 provenance-removal ablation 依預註冊完成；沒有新增搜尋。
- [x] ICACT → IEEE Access technical extension table 已由六頁 PDF 逐頁核對；DOI／獎項證明／similarity report 是投稿行政待辦。
- [ ] GovReport test data policy 在**未看分數**的狀態下建立並 pin 完成。
- [!] 老師／完整作者群簽署 final freeze：目前 frozen policy ordering conflict 待決議；建議採 policy-materialization／execution 兩階段簽核。

任何一項未完成，都不能執行 test。
