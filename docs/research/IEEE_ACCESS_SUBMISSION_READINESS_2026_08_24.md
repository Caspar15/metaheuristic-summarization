# IEEE Access 投稿前詳細稽核快照（2026-08-24）

> **文件角色已調整：** 本文件保留 2026-08-24 的實驗完整性、研究誠信例外、artifact 與
> IEEE Access規則逐項稽核。現行論文章節、投稿 gates、ICACT／ICT Express引用政策與
> 共同作者工作規格已合併到 `IEEE_ACCESS_MANUSCRIPT_BLUEPRINT_2026_08_25.md`；新讀者只需
> 先讀該主指南，本文件僅在查核細節時使用。

> **2026-08-25 consolidation：** 若本文件的寫作架構或待辦狀態與主指南衝突，以
> `IEEE_ACCESS_MANUSCRIPT_BLUEPRINT_2026_08_25.md` 為準；本文件中的 frozen 數字與
> audit observations仍保留為證據，不因角色調整而刪除。

> **2026-09-02 positioning override：** 指導教授決議將 IEEE Access 稿件寫成獨立
> Research Article，不在主文敘述 ICACT extension 或 ICT Express 拒稿。本文件的實驗
> 稽核仍有效；投稿定位、引用與 cover-letter 政策改以
> `IEEE_ACCESS_MANUSCRIPT_BLUEPRINT_2026_08_25.md` 的 2026-09-02 更新為準。

稽核日期：2026-08-24（Asia/Taipei）  
稽核版本：branch `research/govreport-centered-freeze`，HEAD `4dda9f8`  
歷史定位：ICACT conference paper 的實質期刊延伸；此定位已於 2026-09-02 被 standalone
Research Article 政策取代。ICT Express 稿件未出版，只作內部品質檢查依據。

## 1. 最終判定

| 問題 | 判定 | 說明 |
|---|---|---|
| frozen 自動實驗是否完成 | **是** | 兩資料集 official test、九系統 baseline、selector isolation、E2 成本、E3 消融均完成；不再調參 |
| 結果是否足以開始寫論文 | **是** | GovReport 有範圍明確且顯著的正結果；Multi-News 提供誠實的 boundary/trade-off |
| 是否發現造假、竄改分數或挑 run | **未發現** | freeze chronology、config/artifact hash、failed-attempt retention 與 search log 支持現行結果 |
| 是否可以宣稱跨資料集 SOTA | **不可以** | Multi-News 對 PacSum-TFIDF 的 Mean ROUGE 無顯著優勢，且 R-2 顯著較差 |
| 是否可以今天直接投稿 | **還不可以** | 英文主文、clean-clone、similarity、prior-work disclosure、作者資料與 AI disclosure 尚未完成 |

一句話：**科學實驗已 freeze，可以開始正式寫稿；投稿包本身尚未完成。**

## 2. 實際驗證範圍

### 2.1 程式與 repository

- 最新完整 pytest：`496 passed, 5 subtests passed`（2026-09-05）；原 2026-08-24
  checkpoint 為 `488 passed, 5 subtests passed`。
- `python -m compileall -q -f src tests scripts`：通過。
- 稽核開始時工作樹乾淨，HEAD 與 origin branch 對齊；本報告其後形成預期的文件變更。
- GovReport freeze-package verifier：`status=pass`，11/11 本機必要 evidence 完成。
- provenance verifier：468 checked，0 genuine fail；但 468 份皆屬下述 CRLF legacy 類別。
- `search_log.jsonl`：455 筆 development/search 紀錄的 `test_split_accessed` 全為 `false`；
  正式 test 由獨立 one-shot evidence 記錄，沒有混入搜尋 log。
- 未找到晚於稽核日期的 experiment timestamp。

### 2.2 正式實驗矩陣

| 工作包 | 範圍 | 完整性 | 統計／證據 |
|---|---:|---|---|
| GovReport official test | 973 rows、9 systems | 973/973；Proposed 973 feasible | Stanza + Perl ROUGE-1.5.5；100,000 paired bootstrap；Holm |
| Multi-News official test | 5,621 rows、9 systems | 5,621/5,621；12 infeasible rows 留在分母 | 同一 official scale；100,000 paired bootstrap；Holm |
| GovReport E3 | frozen dev 681 rows、5 ablations | 完成 | 20 endpoints、100,000 bootstrap、Holm-20 |
| Multi-News E3 | frozen dev 3,935 rows、5 ablations | 完成 | 20 endpoints、100,000 bootstrap、Holm-20 |
| GovReport E2 | reference-blind dev sample 30 docs、9 systems | 完成 | cold/warm、3 fresh-process repetitions、CPU/RSS/scaling |
| Multi-News E2 | reference-blind dev sample 30 docs、9 systems | 完成 | cold/warm、77/77 completed attempts、CPU/RSS/scaling |
| selector isolation | 兩資料集 frozen dev | 完成 | Greedy、MMR、NSGA-II matched-input；NSGA-II 為負結果 comparator |

兩份 final `analysis.json` 的現行 SHA-256 均與各自 `execution_evidence.json` 相符；
兩份 final config 現行 SHA-256 也仍與 pre-score freeze pin 相符。

### 2.3 最終可寫的核心數字

| Dataset | Proposed R-1 | R-2 | R-L | Mean ROUGE | 預註冊主要比較 |
|---|---:|---:|---:|---:|---|
| GovReport official test | 0.58374 | 0.24711 | 0.54898 | 0.459943 | vs SBERT+MMR：paired mean `+0.003700`，95% CI `[+0.002008,+0.005420]`，`p=0.000040` |
| Multi-News official test | 0.45011 | 0.14314 | 0.41351 | 0.335587 | vs PacSum-TFIDF：paired mean `+0.000091`，95% CI `[-0.001542,+0.001730]`，`p=0.907111` |

`Mean ROUGE` 是預先定義的 `(R-1 + R-2 + R-L) / 3`，不是第四個官方 ROUGE
metric。主表的 corpus-level 平均與 bootstrap 的 per-example paired mean 聚合方式不同，
最後幾位可能略有差異；主文必須在表註定義。

Multi-News 相對 PacSum-TFIDF：R-1 `+0.002389`、R-L `+0.004641`，但 R-2
`-0.006758`；三者在 Holm-3 後均顯著。因此只能寫 metric trade-off，不可只挑 R-1/R-L。

## 3. 「有沒有作弊」的逐項判斷

本節只能說「repository 可見證據中未發現」，不能形式上證明任何人絕不可能作弊。

| 風險 | 稽核結果 | 論文／artifact 處理 |
|---|---|---|
| legacy 用 test 調參 | **確實發生過**：舊 `runs/` 的 11 runs 作廢 | 不得進新稿數字、圖表或模型選擇敘事 |
| 新 pipeline 又偷看 test 調參 | **未發現** | final config hash 仍等於 freeze；test 後不得再改方法 |
| 只留下成功 run | **未發現** | interruption、environment mismatch、guard failure 均保留並說明排除理由 |
| GovReport recovery 偷重跑挑分 | **未發現** | 只由既有 R-1/R-2/R-L 產生缺少的 internal random macro，再完成分析；未重跑 prediction／official evaluator、未改 source score |
| Multi-News E3 看完 test 才設計 | **否** | 科學設計在 `multinews_secondary_evidence_completion_v1.json` 於 final test 前凍結；8/20 addendum 只在 E3 出分前固定執行細節 |
| E3 environment mismatch 後重跑 | **合理但要揭露** | 第一版完整保留並排除；最終只修到 frozen anchor dependency parity，不改 variant 或門檻 |
| 隱藏 Multi-News 負結果 | **沒有** | nonsignificant mean、R-2 退步、12 infeasible rows 均保存與報告 |
| 隨機 baseline 挑 seed | **未發現** | 10 seeds 在 freeze 中固定，報 seed mean |
| artifact hash 漂移 | **有工程債，非分數竄改證據** | 468 evidence 是 CRLF-era direct-hash legacy，靠版本化 LF-normalization errata 通過；仍應 clean-clone |
| 人員核准證明 | **只有作者轉述** | verifier 明寫 `reported_approved_by_requesting_author`，不可說 repository 有獨立簽章 |

結論：**現行 frozen program 沒有發現 fabrication、score tampering 或 post-test
configuration selection。** 最大風險是寫稿時誇大：把 GovReport 的範圍內優勢寫成
universal SOTA、隱藏 Multi-News R-2、或重新引用 legacy test-tuned 結果。

## 4. 尚未完成，但不是新的核心自動實驗

1. 從乾淨 clone 重建環境並跑最小 reproducibility path；重製比較逐篇
   `selected_indices`，而不是要求跨機器 raw byte hash 完全一致。
2. 建立完整 dependency lock 或 container；`requirements.txt` 不等於跨平台 lock。
3. 由 frozen JSON 一鍵產生主文表圖，避免手抄。
4. 做 equation → code → config → evidence → manuscript 一致性稽核。
5. 依固定選例規則補 qualitative success/failure cases。
6. 正式英文、引用、retraction、圖表解析度與 accessibility 終審。

人評不是 IEEE Access 的硬需求。若做人評，必須另固定 sample、盲測、評分項目與統計，
且不可回頭改方法。BERTScore、CNN/DailyMail、SciTLDR 也不是 frozen matrix 的缺漏，
不應在 test 後臨時加入。

## 5. IEEE Access 2026-08-24 官方規定

官方來源：

- [Submission Guidelines](https://ieeeaccess.ieee.org/authors/submission-guidelines/)
- [Preparing Your Article](https://ieeeaccess.ieee.org/authors/preparing-your-article/)
- [IEEE prior-publication policy](https://journals.ieeeauthorcenter.ieee.org/become-an-ieee-journal-author/publishing-ethics/guidelines-and-policies/submission-and-peer-review-policies/)
- [IEEE article structure guidance](https://journals.ieeeauthorcenter.ieee.org/create-your-ieee-journal-article/create-the-text-of-your-article/structure-your-article/)
- [IEEE Access APC](https://ieeeaccess.ieee.org/about/article-processing-charges/)

### 5.1 適合度

- 稿件類型建議選 **Research Article**。
- IEEE Access scope 涵蓋 IEEE fields、multidisciplinary/application-oriented 與傳統
  research articles；extractive summarization/NLP 在廣義 computer science scope 內。
- scope 合適不等於容易接受；官方目前列出的平均 acceptance rate 約 20%。
- 最大審稿風險是 novelty 與 claim calibration，不是資料集數量。

### 5.2 格式與行政需求

- 使用 IEEE Access template，double-column、single-spaced。
- 同時交 Word/LaTeX source 與內容完全一致的 PDF；每檔不超過 40 MB。
- 沒有硬性頁數上限，但強烈建議 **20 頁以下**；一般文章超過 20 頁須先向 EIC 詢問。
- submitting author 的 ORCID 必須公開且有資料；source/PDF 都要列完整作者。
- 所有作者 biography 放在 references 後；投稿選 3–10 keywords。
- 英文文法差可能直接退稿；references 要核對準確且未撤稿。
- 目前 APC 為 **USD 2,160 + applicable taxes**。

### 5.3 Standalone article 與相近 prior work

- 新稿以 standalone Research Article 撰寫；title footnote、Introduction、Discussion 與
  Conclusion 不使用 conference-extension 敘事。
- ICACT 在技術直接相關處作一般 Related Work 引用。IEEE general policy 要求作者揭露相似
  的既有出版品並說明差異，因此 cover letter 與投稿表單仍應誠實處理。
- IEEE Access 對「expanded conference article」規定 similarity 必須低於 35%。本稿雖不採
  該投稿定位，仍須做全文 similarity audit，且不得複製 ICACT 原句、圖或表。
- ICT Express 是未出版拒稿，不是 prior publication；不必把其 response letter 當 IEEE
  Access 必交文件。內部 response matrix 仍應確保四位 reviewer 的問題均已處理。
- Outstanding Paper Award 與本稿科學證據無關，預設不放主文。

### 5.4 AI 使用揭露

本專案已用 AI 協助 code review、測試、文件整理及可能的文字草擬。若主文含 AI 產生
內容，Acknowledgment 必須識別 AI system、指出涉及章節並簡述使用程度；公式、結果、
引用與最終文字仍由作者負責驗證。AI 不可列為作者。

## 6. 建議定位、名稱與貢獻

### 6.1 Working title

> **Provenance-Aware Multi-Route Extractive Summarization Without Task-Specific
> Training: Frozen Evidence From GovReport and Multi-News**

方法工作名可用 **PAMR-ES**，但正式定名前仍應做 acronym collision search。標題不要再
以 `Metaheuristic` 或 `NSGA-II` 為中心，因 frozen evidence 不支持它是最佳 selector。

### 6.2 四項可守住的 contributions

1. 不做 task-specific fine-tuning 的 multi-route extractive framework，讓 lexical、
   semantic、sparse-graph route 在完整輸入各自產生 proposal。
2. 逐句保存 route rank／score／agreement／reservation，以 weighted RRF 把 provenance
   真正送進 selector，而不是只做 index union。
3. matched-input Greedy／MMR／NSGA-II isolation 顯示 selector 依 task profile 而變，
   昂貴 NSGA-II 不帶來穩定品質優勢。
4. 兩資料集 frozen official evaluation、九個本地同 pipeline baselines、paired inference、
   route/provenance ablation 與 cold/warm quality-cost evidence，同時公開正結果與 boundary。

不要把 TF-ISF、SBERT、PageRank、MMR、RRF 或 NSGA-II 個別說成新演算法；新意落在
**provenance-preserving integration、route-to-selector contract、matched evidence 與可稽核設計**。

## 7. 主文架構與頁數預算

IEEE Access 沒有硬性頁數上限，但官方強烈建議主文低於 20 頁。現行中文工作稿為
12 頁；英文正式稿應以完整且精簡為原則，不把 16–19 頁誤寫成期刊要求。

| 區段 | 建議頁數 | 必須回答的問題 |
|---|---:|---|
| Title / Abstract / Index Terms | 1.0 | 問題、方法、兩資料集主要數字、Multi-News 限制；投稿系統要求 3–10 keywords（目前工作稿為 5 個） |
| I. Introduction | 1.5 | 問題、研究缺口、四項貢獻、ICACT extension |
| II. Related Work | 2.0 | centrality、embeddings、fusion/RRF、MMR、metaheuristics、long/multi-document |
| III. Proposed Method | 3.5 | 三 routes、reservation/RRF/provenance、task-profile selector、constraints |
| IV. Experimental Protocol | 2.5 | datasets/splits、length、9 baselines、official evaluator、statistics、hardware |
| V. Results | 4.5 | 兩 official test、selector isolation、E3、E2、qualitative cases |
| VI. Discussion and Limitations | 1.5 | GovReport、Multi-News R-2、NSGA-II、適用範圍與 threats |
| VII. Reproducibility and Availability | 0.5 | commit、config/data policy、data取得、artifact/commands |
| VIII. Conclusion | 0.5 | 只寫 scoped conclusion |
| References + biographies | 2.0–3.0 | 已核對 references；全作者 biography |

### 7.1 Method 順序

1. Problem formulation：source sentences、extractive subset、word budget、source-order output。
2. Lexical route：只寫現行有效訊號；未使用的 legacy feature 不寫入方法。
3. Semantic route：pinned sentence-transformer、batch encoding、完整輸入排名。
4. Sparse graph route：TF-IDF kNN graph、centrality、稀疏化與成本動機。
5. Reservation and weighted RRF：proposal depth、reservation、total cap、route guard。
6. Provenance-aware salience：route evidence 如何進入 selector。
7. Task-profile selection：GovReport 用 TF-IDF MMR；Multi-News 用 Greedy-TFIDF；
   NSGA-II 只作 comparator。
8. Feasibility/output：requested/effective minimum、shortfall/infeasible、原文順序輸出。

### 7.2 主文表格

1. **Table I — Dataset and frozen policy**：角色、source type、dev/dev-test/test rows、length、例外政策。
2. **Table II — System contracts**：9 systems 的 representation、scope、selector、training regime。
3. **Table III — GovReport official test**：9 systems 全部 R-1/R-2/R-L/Mean。
4. **Table IV — Multi-News official test**：同格式，不省略 R-2 負結果。
5. **Table V — Confirmatory paired inference**：delta、CI、raw/adjusted p；註明兩種 aggregation。
6. **Table VI — Matched selector isolation**：Greedy/MMR/NSGA-II，兩資料集 dev quality、length、time。
7. **Table VII — Route/provenance ablation**：兩資料集 delta、CI、Holm-20。
8. **Table VIII — Efficiency**：兩資料集 cold/warm wall、CPU、peak RSS；cold/warm 不交叉比。

### 7.3 主文圖

1. **Fig. 1 Architecture**：文章 → canonical sentences → 三 routes → reservation/weighted
   RRF/provenance → task-profile selector → source-order summary。
2. **Fig. 2 Quality–cost**：official quality 對 E2 cold/warm cost，註明不同固定 sample。
3. **Fig. 3 Scaling**：q10/q50/q90 或 sentence count 對 latency/RSS；只作描述性趨勢。
4. 可選 **Fig. 4 paired difference/CI**：GovReport 正結果與 Multi-News trade-off 並排。

### 7.4 Supplementary Material

- 32-endpoint secondary comparisons、完整 Holm 表。
- E2 q10/q50/q90、CPU、RSS、每 repetition。
- configs、preregistrations、failed attempts、hash/evidence schema。
- 搜尋紀錄與 legacy exclusion policy。
- ICACT extension matrix、更多 qualitative cases、重製命令。

## 8. 投稿前 Gates

### Gate A — 科學內容（已通過）

- [x] final configs frozen；不再依 test 調整。
- [x] 兩資料集九系統 official results。
- [x] paired CI、multiplicity、random seeds。
- [x] selector isolation、E2、E3。
- [x] Multi-News 負結果與 infeasible rows 保留。

### Gate B — artifact（尚未通過）

- [ ] clean-clone reproduction。
- [ ] lockfile/container 或等價完整環境方法。
- [ ] frozen JSON → tables/figures 一鍵生成。
- [ ] 468 legacy newline pins 的投稿版處理定稿。
- [ ] code/data availability 與 exact release tag/commit。

### Gate C — manuscript/compliance（尚未通過）

- [ ] IEEE Access template 內完成英文正式稿，並盡量維持低於 20 頁。
- [ ] ICACT citation/DOI、first-footnote、cover-letter extension table。
- [ ] 正式 similarity <35%。
- [ ] 正式獎項證明；無證明就不寫 award claim。
- [ ] ORCID、作者順序、affiliations、funding、bios、keywords。
- [ ] AI acknowledgment、reference/retraction check、英文終審。
- [ ] equation/code/config/table 一致性與 PDF/source 完全一致。

## 9. 下一步順序

1. 先由 frozen JSON 自動產生 Tables III–VIII 與 figures，禁止手抄。
2. 先寫 Protocol/Results，再寫 Method，讓每個 claim 先有 evidence。
3. 完成 Introduction/Related Work/Discussion 與 internal reviewer matrix。
4. 做 clean-clone artifact rehearsal 與一致性 audit。
5. 補 ICACT DOI、similarity、作者資料、AI disclosure、英文校稿。
6. 作者群逐頁簽認後提交；期間不重開品質搜尋。
