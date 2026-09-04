# Research documentation hub

最後整理：2026-09-04（Asia/Taipei）  
目前階段：**正式 test 已 freeze；IEEE Access 中文獨立 Research Article 已補齊主文證據與負結果，下一步是英文稿、clean-clone/release 與作者行政簽核。**

這是 `docs/research/` 的唯一入口。此目錄刻意保留研究決策與失敗歷史，不能只看檔名或
文件內較早日期判斷現況；請依本頁的狀態與權威順序閱讀。

## 五分鐘閱讀路徑

1. [FINAL_EXPERIMENT_STATUS_2026_08_20.md](FINAL_EXPERIMENT_STATUS_2026_08_20.md)  
   最終實驗結論與可寫／不可寫的主張。
2. [IEEE_ACCESS_MANUSCRIPT_BLUEPRINT_2026_08_25.md](IEEE_ACCESS_MANUSCRIPT_BLUEPRINT_2026_08_25.md)  
   唯一現行論文／投稿主指南：章節、引用、表圖、頁數、readiness與合規。
3. [ACTION_PLAN.md](ACTION_PLAN.md) 的最上方 2026-08-25 區段  
   當前待辦；後續數十頁是完整 phase ledger，不是全部都要重做。
4. [MANUSCRIPT_SUPPLEMENTAL_EVIDENCE_2026_09_04.md](MANUSCRIPT_SUPPLEMENTAL_EVIDENCE_2026_09_04.md)  
   輸出長度、不可行列敏感度、配置量、provenance 案例及三個 post-freeze 機制診斷。

若只想知道「現在下一步是什麼」：**從 frozen JSON 自動產生主文表圖，先寫
Experimental Design 與 Results，再完成 clean-clone/release/compliance。不要重開品質搜尋。**

## 現行事實

| 項目 | 現況 |
|---|---|
| 方法 | lexical + pinned semantic + sparse graph routes；reservation + weighted RRF + provenance-aware salience |
| Selector | GovReport：TF-IDF MMR λ=0.7；Multi-News：Greedy-TFIDF；NSGA-II 非 final 元件，僅在 Supplement／repository 保留負結果 |
| Primary | GovReport official test 973 rows；主要 paired Mean ROUGE 優勢顯著 |
| Secondary | Multi-News official test 5,621 rows；Mean 與 PacSum-TFIDF 統計同級，R-2 顯著較差 |
| 自動實驗 | 兩資料集九系統 official test、selector isolation、E2、E3 均完成 |
| 主文草稿 | 官方 IEEE Access class 中文版已補齊結果、限制與 availability；主文 selector 僅保留 Greedy／MMR，route weights 與 selector 的 development selection 已明列；正式英文、funding／ORCID 與最終作者簽核仍待完成 |
| 禁止事項 | 不再依 test 調方法／配置；不宣稱 universal SOTA；legacy `runs/` 不進新稿 |
| 投稿狀態 | 可以正式寫稿；尚未通過 clean-clone、release、similarity、作者行政與完整 manuscript gates |

## 文件狀態圖例

- **CURRENT**：現行規格或最終結論；寫稿時優先引用。
- **EVIDENCE**：某一實驗／稽核的詳細證據；不能取代 final status。
- **GOVERNANCE**：資料、freeze、claim 或投稿決策的可稽核紀錄。
- **HISTORICAL**：保留當時決策與負結果；文件內的「下一步」可能已過時。
- **OPERATIONS**：執行清單、環境或 repository 整理。

## A. 現行寫作與投稿規格

| 文件 | 狀態 | 用途 |
|---|---|---|
| [IEEE_ACCESS_MANUSCRIPT_BLUEPRINT_2026_08_25.md](IEEE_ACCESS_MANUSCRIPT_BLUEPRINT_2026_08_25.md) | **CURRENT / single master** | 主文章節、RQ、表圖、引用、reviewer mapping、readiness與compliance |
| [IEEE_ACCESS_SUBMISSION_READINESS_2026_08_24.md](IEEE_ACCESS_SUBMISSION_READINESS_2026_08_24.md) | **EVIDENCE snapshot** | 2026-08-24詳細投稿／誠信稽核；不再是另一份主指南 |
| [ICACT_IEEE_ACCESS_EXTENSION_MATRIX.md](ICACT_IEEE_ACCESS_EXTENSION_MATRIX.md) | **HISTORICAL / GOVERNANCE** | ICACT 與現稿差異；只供 similarity／prior-work 稽核，不作正文 extension blueprint |
| [GOVREPORT_CLAIM_MATRIX_V2.md](GOVREPORT_CLAIM_MATRIX_V2.md) | **CURRENT** | 可寫與禁止的 claim；較早 pre-test 欄位只作沿革 |

## B. 最終實驗結果

| 文件 | 狀態 | 用途 |
|---|---|---|
| [FINAL_EXPERIMENT_STATUS_2026_08_20.md](FINAL_EXPERIMENT_STATUS_2026_08_20.md) | **CURRENT / authoritative** | 兩資料集 final 結果、E2/E3、主張邊界 |
| [GOVREPORT_FINAL_TEST_RESULTS.md](GOVREPORT_FINAL_TEST_RESULTS.md) | **EVIDENCE** | GovReport 973-row official test 與 paired inference |
| [MULTINEWS_FINAL_TEST_RESULTS.md](MULTINEWS_FINAL_TEST_RESULTS.md) | **EVIDENCE** | Multi-News 5,621-row official test 與 component trade-off |
| [E2_COST_SCALING_STATUS.md](E2_COST_SCALING_STATUS.md) | **EVIDENCE** | 兩資料集 cold/warm wall、CPU、RSS、scaling |
| [E3_ROUTE_PROVENANCE_ABLATION_STATUS.md](E3_ROUTE_PROVENANCE_ABLATION_STATUS.md) | **EVIDENCE** | 兩資料集 route/provenance ablation |
| [MANUSCRIPT_SUPPLEMENTAL_EVIDENCE_2026_09_04.md](MANUSCRIPT_SUPPLEMENTAL_EVIDENCE_2026_09_04.md) | **CURRENT / EVIDENCE** | 輸出長度、敏感度、配置量、固定案例與 post-freeze mechanism checks |

## C. 現行技術與研究治理

| 文件 | 狀態 | 用途 |
|---|---|---|
| [ARCHITECTURE.md](ARCHITECTURE.md) | **CURRENT** | Target Architecture v2、資料契約、route/fusion/selector 規格 |
| [paper_revision_plan_IEEE_Access.md](paper_revision_plan_IEEE_Access.md) | **GOVERNANCE** | 從拒稿到新 pipeline 的完整研究治理；較早執行狀態已被增補取代 |
| [CODE_AUDIT_IEEE_Access.md](CODE_AUDIT_IEEE_Access.md) | **EVIDENCE ledger** | F 編號、legacy 缺陷、重現方式與修正證據 |
| [COMPUTE_ENVIRONMENT.md](COMPUTE_ENVIRONMENT.md) | **OPERATIONS** | 計算環境、tokenizer、cache、硬體與 timing contract |
| [STRATEGY_ASSESSMENT.md](STRATEGY_ASSESSMENT.md) | **GOVERNANCE** | 投稿可行性與定位；以頂部 2026-08-20 assessment 為現況 |

## D. 已完成的 development-stage 證據

這些文件不是待跑清單，也不能用其中 dev 分數取代 official test。

| 文件 | 狀態 | 用途 |
|---|---|---|
| [D1_SENSITIVITY_STATUS.md](D1_SENSITIVITY_STATUS.md) | **HISTORICAL / EVIDENCE** | dev sensitivity 與 route deletion 診斷 |
| [GATE2_BASELINE_STATUS.md](GATE2_BASELINE_STATUS.md) | **HISTORICAL / EVIDENCE** | baseline family dev 搜尋與 quality-gate 負結果 |
| [SELECTOR_COMPARISON_PROTOCOL.md](SELECTOR_COMPARISON_PROTOCOL.md) | **HISTORICAL / EVIDENCE** | Greedy/MMR/NSGA-II pilot、D2 與 matched-input 規格 |
| [E1_OFFICIAL_EVALUATOR_STATUS.md](E1_OFFICIAL_EVALUATOR_STATUS.md) | **HISTORICAL / EVIDENCE** | GovReport frozen-dev evaluator parity；不是 final test |

## E. Freeze 與決策沿革

| 文件 | 狀態 | 用途 |
|---|---|---|
| [REPOSITIONING_RECOMMENDATION.md](REPOSITIONING_RECOMMENDATION.md) | **HISTORICAL / GOVERNANCE** | v1 gate 失敗到 GovReport-centered option A 的決策形成 |
| [GOVREPORT_PRETEST_FREEZE_RECOMMENDATION.md](GOVREPORT_PRETEST_FREEZE_RECOMMENDATION.md) | **HISTORICAL / GOVERNANCE** | Stage A/B freeze 建議與簽核沿革；已完成，不是待辦 |
| [GOVREPORT_TEST_POLICY_STATUS.md](GOVREPORT_TEST_POLICY_STATUS.md) | **HISTORICAL / GOVERNANCE** | score-free test policy materialization 與 dry-run 狀態 |

## F. 執行與導航文件

| 文件 | 狀態 | 用途 |
|---|---|---|
| [ACTION_PLAN.md](ACTION_PLAN.md) | **OPERATIONS ledger** | 最上方是當前待辦；其餘保留 Phase −1～6 完整歷史 |
| [REPO_CLEANUP.md](REPO_CLEANUP.md) | **OPERATIONS backlog** | repository 整理提案；不授權刪除／搬移 |
| [INDEX.md](INDEX.md) | **HISTORICAL index** | 依日期累積的詳細索引；新讀者應先看本頁 |
| [evidence/README.md](evidence/README.md) | **EVIDENCE index** | 小型 audit artifacts 的用途與權威邊界 |

## 權威順序

內容衝突時按以下順序處理：

1. final frozen artifacts／`FINAL_EXPERIMENT_STATUS_2026_08_20.md`；
2. `IEEE_ACCESS_MANUSCRIPT_BLUEPRINT_2026_08_25.md`（寫作、投稿 gates與引用政策）；
3. `ARCHITECTURE.md`（技術契約）；
4. claim／extension matrices；
5. submission-readiness snapshot、E1/E2/E3、Gate 2、D1、selector與F-ledger evidence；
6. 早期計畫、freeze建議與chronological index。

較新的狀態增補可以取代舊文件的「現在／下一步」，但不能抹除當時分數、失敗或決策。

## 目錄整理政策

- 現階段**不物理搬動**既有 Markdown：它們有大量 repository 內交叉引用與 evidence path。
- 不因文件過時就刪除；改以 CURRENT/EVIDENCE/HISTORICAL 標記。
- 新的主文／投稿文件放在 `docs/research/`，檔名加日期或版本。
- 小型機器可讀稽核產物放 `docs/research/evidence/`；大型 runs 仍放 `runs_v2/`。
- 不在多份文件手抄同一新數字；final 數字只以 frozen artifact 與 final status 為源頭。
- 待 manuscript/release 完成後，才另做一次全 repository link migration，再考慮建立
  `history/`、`results/`、`submission/` 實體子目錄。

## 目前待辦

| 待辦 | 白話意思 | 完成標準 |
|---|---|---|
| [x] frozen evidence → 中文稿 Tables I–X | 主文數值已逐項對 frozen artifacts，並完成 PDF 逐頁檢查；Fig. 1 是必要架構圖，其他 paired／cost 圖列為英文稿可選 | 中文工作稿表格、補充證據與 frozen artifacts 一致 |
| [ ] frozen JSON → tables 自動生成 | 把目前已核對的人工排版改為可重建數值區塊，降低英文改稿時的抄錄風險 | 同一指令可重建 Tables I–X 的數值區塊；可選圖僅在實際納入時生成 |
| [ ] 撰寫 Experimental Design、Results、Method | 中文完整工作稿已建立並編譯；下一步將它改寫成正式學術英文，而不是逐句直譯 | 三章英文初稿完成，每個數字、公式與config都有來源 |
| [ ] Related Work bibliography／retraction／ICACT資料 | 已建立 15 筆 primary-source 初始書目，ICACT DOI 與官方 Outstanding Paper 頁面已確認；仍須逐筆做 retraction 與 metadata 終審 | 引用可逐筆驗證，沒有錯作者／錯年份／已撤稿文獻 |
| [ ] clean-clone／environment lock／release tag | 模擬審稿人從全新資料夾下載repo，照說明能安裝、測試與重現；把環境與投稿版本固定 | 乾淨clone驗證通過，有lock/container、確切release tag與reproduction commands |
| [ ] qualitative fixed-rule cases | 按事先固定規則挑成功／失敗摘要，讓人看懂ROUGE數字背後好在哪、錯在哪 | 規則先固定、正反案例都報；若不做，Limitations明寫沒有質性／人評 |
| [ ] 投稿行政與終審 | 做similarity、作者／ORCID／bios／funding／AI揭露、英文、PDF/source一致性 | IEEE Access submission package逐項通過且作者群簽認 |
