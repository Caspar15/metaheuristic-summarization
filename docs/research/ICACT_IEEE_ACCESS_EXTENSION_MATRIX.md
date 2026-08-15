# ICACT → IEEE Access Extension Matrix

> **2026-08-15 狀態**：技術差異草案已與 GovReport-centered v2 對齊，但所有
> camera-ready 核對欄仍未完成；E1 official evaluator、E2 cost/scaling、E3 ablation
> 已產生可填入新稿的凍結 dev 表格，但仍不是 final test。此文件目前不能當成
> extension 合規已完成的證明。

## 使用方式

這是投稿前的差異證據表，不是宣傳稿。ICACT camera-ready 與 ICT Express 被拒稿全文
不在 repo 內，因此「ICACT 已有內容」目前只填入 repo 文件可確認的高層範圍；頁碼、
公式與表號必須由作者拿 camera-ready 逐項核對後才能勾選完成。不能把「程式已做」直接
等同「期刊稿具有足夠新增比例」，正式 similarity 仍以投稿檢查為準。

## Extension matrix

| 面向 | ICACT／ICT Express 既有範圍 | IEEE Access 實質新增 | 證據位置 | Camera-ready 核對 |
|---|---|---|---|---|
| 研究問題 | 結合 metaheuristic、graph centrality 與 PLM semantics 的抽取式摘要 | 改為 GovReport-centered、training-free、provenance-preserving long-document framework，並明列 Multi-News boundary | `GOVREPORT_CLAIM_MATRIX_V2.md` | [ ] 頁碼／原主張 |
| 資料治理 | 舊稿存在 test-tuned Multi-News runs，不能再引用 | Canonical schema、pinned revisions、health manifests、GovReport official archive、reference-blind dev/dev-test partitions、test lock | `configs/data_policies/`、F-17/F-71 | [ ] 舊資料表 |
| 候選架構 | 三路結果以 index union 組合，route evidence 會遺失 | Independent lexical／semantic／sparse-graph proposals、route reservation、weighted RRF、逐句 provenance | `ARCHITECTURE.md`、prediction schema | [ ] 舊流程圖 |
| Selector | NSGA-II 是主要敘事，參數／fallback 曾與論文不一致 | Greedy／MMR／NSGA-II matched-input isolation；NSGA-II 因負證據降為 comparator | D2、selector pilot、F-63–F-65 | [ ] 舊 NSGA 公式／表格 |
| Objective／constraint | 舊 importance、coverage、redundancy 語義與實作不完全一致 | Shared objective factory、length-normalized importance、source/candidate feasibility、fail-loud output contract | `src/objectives/`、F-14/F-17 | [ ] 舊目標函式 |
| Baselines | baseline 不完整，部分數字來自不同 split／evaluator | 本地同 pipeline 的 Lead、Random、TextRank、LexRank、PacSum TF-IDF/SBERT、SBERT centroid/MMR | `GATE2_BASELINE_STATUS.md` | [ ] 舊 baseline 表 |
| 統計 | 缺少完整 paired inference 與搜尋機會校正 | Per-example artifacts、paired bootstrap、Holm、selection-aware Bonferroni、失敗 profile 保留 | Gate 2、D1–D3 evidence | [ ] 舊統計段落 |
| 評測協定 | 舊 ROUGE 定義與句界不足以支持 published comparison | Internal ROUGE golden + GovReport official Stanza/Perl ROUGE parity（E1 完成） | `E1_OFFICIAL_EVALUATOR_STATUS.md` | [ ] 舊 evaluator 描述 |
| 效能與成本 | 元件計時曾被誤外推為完整 pipeline | Cold/warm end-to-end latency、peak RAM、cache state、document-length scaling（CPU E2 完成；GPU 不適用主報告） | `E2_COST_SCALING_STATUS.md` | [ ] 舊 timing 表 |
| Ablation | 無法分離 route、candidate pool、selector 的效果 | Capacity-aware route removal + fixed-pool weighted/index-only provenance removal（E3 完成） | `E3_ROUTE_PROVENANCE_ABLATION_STATUS.md` | [ ] 舊 ablation 表 |
| 適用範圍 | 容易被解讀為一般性品質優勢 | GovReport 正證據與 Multi-News R-2 負結果並列，限制寫入摘要／結論 | D3b paired summary、claim matrix | [ ] 舊結論 |
| 可重現性 | 缺乏完整 artifact chain | Config/data/manifest SHA、dependency pins、failed-attempt retention、CI、provenance verifier | `runs_v2/`、F-51/F-70/F-71 | [ ] 舊 availability statement |

## 投稿前必填

- [ ] ICACT 完整書目與 DOI。
- [ ] ICACT「傑出論文獎（Outstanding Paper Award）」的正式獎項名稱與可引用證明。
- [ ] ICACT camera-ready 每項 contribution 的頁碼、公式、表格與圖號。
- [ ] IEEE Access 新稿對應章節、表格、圖號。
- [ ] Cover letter 逐項說明 substantial extension，而非只寫「增加更多實驗」。
- [ ] 正式 similarity report；不得以本地 n-gram diagnostic 代替。
- [ ] 新稿引用 ICACT prior publication，並按最新 IEEE 規定處理 AI assistance disclosure。

## 建議 cover-letter 核心句（草案）

> This manuscript substantially extends our ICACT conference paper by replacing the
> test-tuned evaluation with frozen data governance, introducing provenance-preserving
> multi-route fusion and matched selector isolation, adding locally reproduced strong
> baselines and multiplicity-aware paired inference, and reporting a GovReport-centered
> result together with a negative Multi-News boundary condition. NSGA-II is retained as
> a controlled comparator rather than being presented as the quality-leading method.

此句必須在 camera-ready 核對與全部 evidence 完成後再定稿。
