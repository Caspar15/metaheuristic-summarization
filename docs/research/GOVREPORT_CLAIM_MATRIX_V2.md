# GovReport-centered Claim Matrix v2

> **2026-08-20 final-result addendum（現行權威）**：GovReport final official test 已完成並
> 保留 scoped superiority claim；Multi-News 其後經作者同意，以獨立 frozen secondary
> protocol 完成 official test，macro 排名第一但不顯著勝 PacSum，且 R-2 顯著較差。
> 兩資料集 E2/E3 均完成。下方 pre-test 欄位保留為治理沿革；現行主張以
> `FINAL_EXPERIMENT_STATUS_2026_08_20.md` 為準。

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
| 唯一主要 confirmatory 資料集 | **GovReport** | 驗證 long single-document、multi-sentence、training-free extractive summarization | official test 已完成且主要 macro superiority 通過；scope 不外推 |
| Secondary multi-document benchmark | **Multi-News** | 報告跨資料型態穩健性與 metric trade-off | official macro 排名第一但與 PacSum 統計同級；必須揭露 R-2 顯著較差，不作共同 promotion gate |
| 未納入 | CNN/DailyMail、SciTLDR、PubMed | 本次定案不新增 | 不能為了增加資料集數量而在 freeze 後臨時加入 |

## 論文主張矩陣

| ID | 主張 | 目前可寫到哪裡 | 投稿前還缺什麼 | 失敗時怎麼辦 |
|---|---|---|---|---|
| C-GOV-QUALITY | Frozen proposed 在 GovReport 優於 full-source SBERT+MMR λ=0.9 | official test macro `+0.003700`、95% CI 全正、`p=0.000040`；R-L 單項未顯著 | 寫入主文並維持 scope/limitations | 不得 post-test 調參或擴張成跨資料集 SOTA |
| C-MN-QUALITY | Multi-News secondary benchmark 排名與取捨 | official macro 第一但對 PacSum-TFIDF `p=0.907111`；R-1/R-L 顯著較高、R-2 顯著較低 | 三分項、CI 與 nonsignificant macro 全報 | 不可只報排名、R-1 或 R-L 隱藏 R-2 |
| C-PROVENANCE | 每句保留 lexical／semantic／graph route、rank、fusion 與 selector provenance | 兩資料集 fixed-pool A04/A05 支持 provenance；Multi-News A04 效果小且門檻邊緣 | 保守描述 task-specific route weighting | 不宣稱 universal optimal weights |
| C-ROUTES | semantic／graph 提供 capacity-aware 增量 | GovReport 與 Multi-News A01/A02/A03 均顯著；兩資料集 E2 成本與 memory 已量測 | 與 cold/warm cost 一起報 | 不宣稱免費或所有資料集效果相同 |
| C-NSGA | NSGA-II 是 matched comparator，不是最佳 selector | 品質低於 Greedy；Multi-News cold/warm `91.69/50.07s` vs Proposed `47.85/9.23s` | 如實報告／附錄，連結 ICACT 延伸 | 不能放回標題或宣稱 metaheuristic superiority |

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

## 解鎖 final evaluation 前的 DoD（歷史；兩次 one-shot 已完成）

- [x] GovReport 作者官方 Stanza + Perl ROUGE-1.5.5 protocol 已版本化並完成 parity/ranking check。
- [x] Frozen candidate、SBERT+MMR 與必要 baselines 的 CPU cold/warm runtime、peak process-tree RSS、scaling 完成；GPU VRAM 不適用於本 CPU-only 主報告。
- [x] Route-removal 與 provenance-removal ablation 依預註冊完成；沒有新增搜尋。
- [x] ICACT → IEEE Access technical extension table 已由六頁 PDF 逐頁核對；DOI／獎項證明／similarity report 是投稿行政待辦。
- [x] GovReport test data policy 已在**未產生 prediction、未看分數**的狀態建立並 pin 完成。
- [~] 兩階段核准已由提出請求的作者轉述；Stage B 仍須 exact runner／environment／commands freeze 完成才啟動。

任何一項未完成，都不能執行 test。
