# ICACT → IEEE Access Extension Matrix

> **2026-08-15 狀態**：已逐頁核對作者提供的 6-page ICACT PDF（raw SHA-256
> `fa4c0c18...a4577`），並與 GovReport-centered v2、E1 official evaluator、E2
> cost/scaling、E3 ablation 對齊。技術內容 audit 已完成；正式 DOI／書目、Outstanding
> Paper Award 證明、新稿頁碼與 similarity report 仍待補，因此目前仍不是投稿合規完成證明。

## 使用方式

這是投稿前的差異證據表，不是宣傳稿。ICACT PDF 位於 repository 上層 `../ICACT/`，
內容 audit 與逐頁 mapping 已記錄於 `evidence/icact_camera_ready_audit_v1.json`。ICT Express
PDF 是被拒稿、不是 prior publication；它可供 reviewer-response 修訂，但 conference
extension 的法規比較基準只應是 ICACT。不能把「程式已做」直接等同「期刊稿具有足夠
新增比例」，正式 similarity 仍以投稿檢查為準。

## Extension matrix

| 面向 | ICACT 既有範圍 | IEEE Access 實質新增 | 證據位置 | ICACT 頁碼／項目 |
|---|---|---|---|---|
| 研究問題 | NSGA-II/metaheuristics 對 LM-only selector 的效率與品質優勢，並以 LM 輔助 ensemble | 改為 GovReport-centered、training-free、provenance-preserving long-document framework，並明列 Multi-News boundary | `GOVREPORT_CLAIM_MATRIX_V2.md` | [x] p.1 Abstract/Introduction；p.5 Conclusion |
| 資料治理 | CNN/DailyMail 三句與 SciTLDR-AIC 單句；只述相同 preprocess，無 pinned split/artifact policy | Canonical schema、pinned revisions、health manifests、GovReport official archive、reference-blind partitions、test lock | `configs/data_policies/`、F-17/F-71 | [x] pp.2–4 dataset/evaluation descriptions；Tables 1/4/5 |
| 候選架構 | NSGA-II top-10 與 BERT/RoBERTa/XLNet top-10 做 index union，再以 MMR 選三句 | Independent lexical／semantic／sparse-graph proposals、route reservation、weighted RRF、逐句 provenance | `ARCHITECTURE.md`、prediction schema | [x] p.2 Method steps 3–5；p.3 Eq. (12)–(13) |
| Selector | NSGA-II 是主要品質敘事；MMR 是 union 後 final selector | Greedy／MMR／NSGA-II matched-input isolation；NSGA-II 因負證據降為 comparator | D2、selector pilot、F-63–F-65 | [x] pp.2–3 Eq. (7)–(13)；p.3 Tables 1–3 |
| Objective／constraint | TF-ISF、length、position 與三個 NSGA-II objectives | Shared objective factory、length-normalized importance、source/candidate feasibility、fail-loud output contract | `src/objectives/`、F-14/F-17 | [x] pp.2–3 Eq. (1)–(10) |
| Baselines | BERT/RoBERTa/XLNet similarity、Greedy、GRASP、NSGA-II；無 Lead/LexRank/PacSum/SBERT-MMR 強矩陣 | 本地同 pipeline 的 Lead、Random、TextRank、LexRank、PacSum TF-IDF/SBERT、SBERT centroid/MMR | `GATE2_BASELINE_STATUS.md` | [x] p.3 Table 1 |
| 統計 | Tables 1/4/5 只有 aggregate point estimates，無 paired CI 或 multiplicity correction | Per-example artifacts、paired bootstrap、Holm、selection-aware Bonferroni、失敗 profile 保留 | Gate 2、D1–E3 evidence | [x] pp.3–4 Tables 1/4/5 |
| 評測協定 | 只寫 ROUGE-1/2/L F1；沒有 published-evaluator parity | Internal ROUGE golden + GovReport official Stanza/Perl ROUGE parity（E1 完成） | `E1_OFFICIAL_EVALUATOR_STATUS.md` | [x] p.2 evaluation paragraph；p.3 Section IV |
| 效能與成本 | Per-article timing；Tables 2–3 標題寫 microseconds、cells 卻是 ms，另有 SciTLDR Table 6 | Cold/warm end-to-end latency、peak RAM、cache state、document-length scaling（CPU E2 完成） | `E2_COST_SCALING_STATUS.md` | [x] p.3 Tables 2–3；p.4 Table 6 |
| Ablation | Table 1 比 component methods、Table 4 比三個 ensembles，但無 matched route/provenance isolation | Capacity-aware route removal + fixed-pool weighted/index-only provenance removal（E3 完成） | `E3_ROUTE_PROVENANCE_ABLATION_STATUS.md` | [x] pp.3–4 Tables 1/4 |
| 適用範圍 | 結論宣稱 robust evidence／generalization；CNN/DailyMail 與 SciTLDR-AIC | GovReport 正證據與 Multi-News R-2 負結果並列，限制寫入摘要／結論 | D3b paired summary、claim matrix | [x] pp.4–5 SciTLDR discussion與Conclusion |
| 可重現性 | 無 config/data/artifact SHA、failed-attempt log 或 CI | Config/data/manifest SHA、dependency pins、failed-attempt retention、CI、provenance verifier | `runs_v2/`、F-51/F-70/F-71 | [x] 全文未見相應 artifact chain |

## 投稿前必填

- [ ] ICACT 完整書目與 DOI。
- [ ] ICACT「傑出論文獎（Outstanding Paper Award）」的正式獎項名稱與可引用證明。
- [x] ICACT 六頁 PDF 的每項 contribution、公式與表格已逐頁核對；見 evidence JSON。
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

技術內容與 E1～E3 已核對；此句仍須在 DOI／正式書目、獎項證明與 similarity report
完成後再定稿。
