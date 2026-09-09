# IEEE Access 論文與投稿主指南

日期：2026-08-25；定位更新：2026-09-04（Asia/Taipei）  
適用版本：GovReport-centered frozen v2；official test、E2、E3 與 selector isolation 已完成  
用途：正文、Supplementary Material、cover letter、投稿 gates 與共同作者分工的**唯一現行主指南**

> 2026-08-25 consolidation：本文件已合併原本分散在 manuscript blueprint 與
> submission-readiness 的現行指引。`IEEE_ACCESS_SUBMISSION_READINESS_2026_08_24.md`
> 保留為 2026-08-24 的詳細稽核快照，不再與本文件競爭「現行權威」地位。

> **2026-09-02 manuscript-positioning override：** 依指導教授決議，IEEE Access 稿件
> 以獨立 Research Article 撰寫，不再使用「ICACT conference extension」作為正文敘事，
> 也不提 ICT Express 拒稿。ICACT 僅在技術直接相關處作一般文獻引用；考量 IEEE 要求
> 揭露相似的作者既有出版品，cover letter 仍應用一小段誠實說明差異。下方已依此更新。

> **2026-09-04 evidence override：** 主文已補輸出長度、matched infeasible-row
> sensitivity、configuration counts 與固定規則 provenance case。Post-freeze
> development-only checks 顯示 reservation 無品質增益、移除 lexical candidate route
> 反而較好；fixed-pool zero-lexical-weight 又把差異拆成 ranking vote 與 candidate
> membership。不得因此重跑 test 或改 final system，但必須刪除「三路各自正向」與
> 「reservation 提升 ROUGE」的過度主張。詳見
> `MANUSCRIPT_SUPPLEMENTAL_EVIDENCE_2026_09_04.md`。

> **2026-09-04 narrative simplification override：** NSGA-II 不屬於 final PAMR-ES，也不
> 支撐本文的 fusion/provenance 貢獻，因此移出主文的 selector table、runtime table、RQ、
> Discussion 與 Conclusion；既有 matched quality、runtime、seed/Pareto 負結果完整保留於
> Supplement 與 repository。主文 selector isolation 只比較 Greedy／MMR。另須明說
> dataset profile 不只包含 selector，也包含 development-selected route weights；正確主張是
> no task-specific fine-tuning，而不是 no tuning。

## 0. 本稿的核心定位

### 0.1 一句話定位

本稿不是宣稱發明 TF-ISF、SBERT、PageRank、RRF、MMR 或 NSGA-II，而是提出一個
**不做 task-specific fine-tuning、保留每條候選路徑來源證據的多路抽取式摘要框架**，
並以 frozen evaluation 檢驗它在長文件 GovReport 的有效性、在 Multi-News 的適用邊界，
以及不同 selector、route 與 provenance 的實際作用。

### 0.2 建議題目

> **Provenance-Aware Multi-Route Extractive Summarization Without Task-Specific
> Training: Frozen Evidence From GovReport and Multi-News**

方法暫名可用 **PAMR-ES**（Provenance-Aware Multi-Route Extractive Summarization），
正式投稿前仍須做名稱／縮寫撞名檢索。標題不要再以 `Metaheuristic`、`NSGA-II`、`LLM`
或 `Graph` 單獨作為主角，因 frozen evidence 不支持這些是獨立或主要創新。

### 0.3 主張層級

| 層級 | 本稿可以主張 | 本稿不可主張 |
|---|---|---|
| 主要 | GovReport frozen official test 上，Proposed 對預註冊主要比較 SBERT+MMR 的 Mean ROUGE paired difference 顯著為正 | universal SOTA、全面勝過所有方法、勝過所有 LLM |
| 次要 | Multi-News 上 Mean ROUGE 與最強 PacSum-TFIDF 統計同級，R-1/R-L 較高而 R-2 較低 | 跨資料集一致顯著優勝、隱藏 R-2 負結果 |
| 方法 | semantic／graph 與 weighted RRF ranking evidence 形成可稽核的 route-to-selector contract；reservation 只作來源平衡／紀錄 | route reservation 提高品質、lexical candidate route 或三條 routes 各自皆有正貢獻；TF-ISF、SBERT、PageRank、RRF、MMR 或 NSGA-II 本身是新演算法 |
| 實證 | matched selector、route/provenance ablation、official evaluator、paired statistics、cold/warm cost/scaling | NSGA-II 品質最佳、graph 是「entity relationship reasoning」、greedy reference 是 exact oracle |
| 訓練 | 不做 task-specific fine-tuning；允許使用固定的 pretrained sentence encoder | 完全不使用 pretrained model、方法是 LLM-based、zero-compute |

## 1. ICACT 與 ICT Express 的使用政策

### 1.1 ICACT：一般相關研究引用，不作正文沿革

本稿的研究問題、方法與實驗均應自成一體，不靠 ICACT 沿革才能理解。因此：

1. **不放 title-page extension footnote**，第一頁只留 funding、corresponding author 與
   template 必要資訊。
2. **Introduction 不寫投稿沿革**，直接從研究問題、缺口、方法與 contributions 展開。
3. **Related Work 正常引用 ICACT**：把它視為結合 LM-based selector 與 metaheuristic
   search 的相關方法，不用「作者先前版本」作為主要敘事。
4. **Discussion/Conclusion 不寫從 ICACT 修正而來**；NSGA-II 不進主文敘事，其既有
   matched selector 負結果保留於 Supplement 與 repository。
5. **Cover letter 保留一小段 prior-work disclosure**：IEEE 要求作者揭露相似的既有
   conference publication，並清楚說明新稿差異。這是誠信揭露，不是把正文定位成 extension。

正文不得複製 ICACT 的原句、圖或表。架構圖、公式敘事與表格均由現行程式及 frozen evidence
重新建立。若日後真的重用 copyrighted material，必須在對應 caption／段落標示來源並確認
permission。ICACT 的 DOI 與技術差異矩陣保留在內部投稿資料夾，以供 cover letter 與
similarity audit 使用，不進主文敘事。

### 1.2 ICT Express：目前不進 References

ICT Express 稿件被拒且未出版，因此在目前事實下：

- 不放進 References。
- 正文不寫 `Reviewer 1/2/4 suggested ...`，也不引用 reviewer comments。
- 不把 ICT Express response letter 當作 IEEE Access 新投稿的附件。
- 不必在新稿正文或 cover letter 主動敘述「曾被 ICT Express 拒稿」，除非投稿表單明確詢問。
- 審稿意見只作**內部 requirement matrix**，確保新稿已處理弱 baseline、方法不清、
  novelty overclaim、reproducibility、runtime、qualitative evidence 等問題。

例外：若 ICT Express 版本已公開成 arXiv、institutional repository、preprint 或有可公開
識別碼，它就成為 public prior version；此時投稿前必須重新判斷是否引用與揭露。

### 1.3 Outstanding Paper Award

獎項不是科學證據。只有在取得正式獎項名稱、主辦單位頁面或證書後，才可在 cover letter
或 author biography 簡短提及；不要放進 Abstract、Contributions 或用它代替 novelty 論證。

## 2. Front Matter

### 2.1 Title

- 明確出現 `Extractive Summarization`。
- 明確反映 `Provenance-Aware Multi-Route` 與 `Without Task-Specific Training`。
- 不使用 `State-of-the-Art`、`Novel Graph`、`LLM-based`、`Metaheuristic-Optimized`。

### 2.2 Abstract：建議 180–230 words，六句式

1. **Problem**：長文件／多文件抽取摘要需要同時處理重要性、語義代表性與結構訊號。
2. **Gap**：既有 training-free hybrid 通常只合併候選 index，路徑來源證據不會真的進 selector；
   selector 與 candidate changes 也常被混在一起比較。
3. **Method**：提出三路 proposal、weighted RRF、sentence provenance、作為來源平衡的
   route reservation 與 task-profile selector；不做 task-specific fine-tuning。
4. **Protocol**：GovReport 為 confirmatory primary，Multi-News 為 secondary boundary；
   九系統同 pipeline、official ROUGE、paired bootstrap、E2/E3。
5. **Main result**：寫 GovReport 的 R-1/R-2/R-L/Mean 與主要 paired delta、CI、p-value。
6. **Boundary/conclusion**：Multi-News Mean 與 PacSum 統計同級，明寫 R-2 trade-off；結論是
   provenance-aware integration 在特定長文件 profile 有效，而不是 universal superiority。

Abstract 不放引用、不塞完整 baseline 清單、不寫舊 ICACT／ICT Express 故事、不把 Mean
ROUGE 稱為官方 metric。

### 2.3 Index Terms

建議 6–8 個：`extractive summarization; long-document summarization; multi-document
summarization; training-free summarization; candidate fusion; provenance; graph centrality;
multi-objective optimization`。若 NSGA-II 只剩 comparator，可刪最後一個，避免誤導主題。

## 3. I. Introduction（約 1.5 頁，7–8 段）

### P1 — 問題與實務重要性

說明長政府報告與多新聞文件的資訊重複、跨段落議題與篇幅限制，使單純 Lead 或單一路徑
不一定能同時兼顧重要性、覆蓋與冗餘。不要一開始就講 NSGA-II。

### P2 — 為何選 training-free extractive setting

定義使用情境：沒有 task-specific fine-tuning、需要 traceable source sentences、資源有限，
但允許固定 pretrained sentence encoder。說明抽取式輸出可追溯，卻會受 abstractive
reference mismatch 影響，先埋下 limitation。

### P3 — 現有方法缺口

分三層寫：

- lexical／position 容易受表面字詞與 lead bias 影響；
- semantic embedding 可找語義代表性，但可能忽略文件結構與局部主題；
- graph centrality 擅長結構共識，但可能偏好重複、主流節點。

真正缺口不是「沒有人結合過三者」，而是：**候選被融合後，其 route rank、agreement、
reservation 與 unique contribution 常沒有進入最終 selector，也缺少 matched-input evidence。**

### P4 — 評測缺口

指出研究需要：同 preprocessing/evaluator 的強 baseline、鎖定資料角色、paired uncertainty、
route/provenance ablation 與 end-to-end cold/warm cost。不要指名批評 ICT reviewer。

### P5 — 方法總覽

用一段對應 Fig. 1：canonical sentences → lexical／semantic／sparse graph routes → route
reservation + weighted RRF/provenance → task-profile selector → source-order extract。

### P6 — Research Questions

- **RQ1**：在 frozen GovReport confirmatory protocol 下，方法能否勝過預先指定的主要
  no-task-specific-fine-tuning baseline（SBERT+MMR）？
- **RQ2**：同一多路徑設計在 Multi-News 是否仍有優勢；若沒有，差異落在哪些 ROUGE
  components？
- **RQ3**：lexical、semantic、graph routes、provenance-aware fusion 與 final selector 在
  fixed-development comparisons 中呈現何種效果，其時間與記憶體代價為何？

RQ1 不寫成「本文表現最佳的 baseline」，避免用結果反過來定義 comparator；selector 與
成本合併進 RQ3，讓主文維持三個研究問題，與實際章節及表格一一對應。

### P7 — Contributions（三點）

1. 一個不做 task-specific fine-tuning 的 lexical／semantic／sparse-graph multi-route
   extractive framework，以 weighted RRF 與 route reservation 建立容量受控候選池；
   reservation 的貢獻限定為來源平衡與可稽核性，不宣稱獨立品質增益。
2. 一個 sentence-level provenance contract，將 route evidence 傳入 selector，並以
   exact-pool ablation 分離候選成員與排序訊號的效果。
3. frozen GovReport/Multi-News official evaluation：九系統、paired inference、multiplicity、
   matched Greedy/MMR selector isolation、route/provenance ablation，以及 cold/warm
   runtime-memory analysis；非 final selector 的負結果另在 Supplement 完整保存。

### P8 — Article organization

用一到兩句說明後續章節組織；不放 conference relation paragraph。主文固定為九個編號
章節：Introduction、Related Work、Problem Definition and Research Questions、Method、
Experimental Design、Results、Discussion/Limitations、Reproducibility/Availability、Conclusion。

## 4. II. Related Work（約 2–2.5 頁）

### A. Classical and Graph-Based Extractive Summarization

包含 Lead、TextRank、LexRank、PacSum；區分 lexical graph、centrality 與 position bias。
本稿的 sparse graph route 不宣稱比 TextRank/LexRank 類方法「原理全新」，新意在它如何
與其他 routes 及 provenance contract 互動。

### B. Pretrained Representations and Extractive Selection

包含 Sentence-BERT、centroid ranking、MMR；將 BERTSumExt、MatchSum 放在 supervised
context，而不是假裝它們與 no-task-training 主表完全同資源。說清楚：本稿使用固定
pretrained encoder，但不對 GovReport／Multi-News 做 task-specific parameter updates。

### C. Long-Document and Multi-Document Summarization

介紹 GovReport、Multi-News 及相對應長文件／多文件困難。討論 G-SEEK、G-SEEK-2 等
graph-based extract-then-abstract 或 trained graph systems時，要標示它們的 output regime、
training data 與 generator 不同，因此屬方法定位比較，不是本地九系統主表的公平 baseline。

### D. Hybrid, Ensemble, and Decision Provenance

回顧候選融合、rank fusion 與 MMR，並在技術直接相關處引用 ICACT `[ICACT]`。
承認 hybrid idea 已存在，避免 `first ever combination`。最後指出仍缺少「provenance 真正進入
selector」與「candidate-matched selector isolation」的證據。

### E. Gap Synthesis

用一段收斂四個區別：

1. output 是 extractive，不是 abstractive；
2. training 是 no task-specific fine-tuning，不是 model-free；
3. graph 是 complementary route，不是唯一創新；
4. contribution 是 provenance-aware integration + frozen evidence，不是元件發明。

建議放 **Table I — Related-system taxonomy**：`Method / task-specific training / output type /
representation / graph / selector / candidate provenance / locally rerun?`。文獻數字不要抄進
本地主結果表；不同 evaluator/split/output regime 只做 taxonomy。

## 5. III. Problem Formulation and Scope（約 0.75–1 頁）

### A. Input and Extractive Output

定義文件集合、canonical sentence sequence、每句 word count、選取 subset 與最後依 source
order 排列。Multi-News 要保留 document identity／position scope，不能把多份文章當成一篇
無邊界文字。

### B. Task Profiles and Length Constraints

- GovReport：single-document long-form profile，500–650 words。
- Multi-News：multi-document news profile，200–250 words。
- 定義 requested/effective minimum、upper bound、shortfall 與 infeasible row 的處理。
- 明說 Multi-News final 12 infeasible rows仍留在 all-row denominator。

### C. Training and Comparison Scope

定義 `without task-specific training`；固定 pretrained SBERT 可以用，但不能用 reference
更新 encoder、fusion weight 或 selector。區分本稿主表的 no-task-training baselines 與只在
Related Work 討論的 supervised／generative／LLM systems。

### D. Objective and Research Scope

目標不是保證 exact global optimum，而是在固定 word budget 下平衡 salience、coverage、
redundancy。`greedy reference` 只作 reference-aware diagnostic，不叫 oracle／upper bound。

## 6. IV. Proposed PAMR-ES Framework（約 3–3.5 頁）

### A. Overview

對應 **Fig. 1 Architecture**。列出輸入、三 routes、fusion/provenance、selector、constraint、
source-order output。圖中不可出現實作未使用的 BERT/RoBERTa/XLNet ensemble。

### B. Canonical Sentence Representation

說明 stable sentence ID、document ID、original index、text、position、word count、route
records。這是 `canonical` 的正式定義，不把它寫成神祕演算法。

### C. Lexical Route

只寫 frozen configs 實際啟用的 TF-ISF、n-gram／position 與 importance aggregation。
未啟用或 legacy-only feature 不放進主方法。給 lexical salience equation 與符號表。

### D. Semantic Route

寫 pinned sentence-transformer checkpoint／revision、pooling、cosine score、batching、每句
truncation policy，以及完整 source ranking 到 route proposal 的過程。稱為 pretrained
sentence encoder 或 PLM route，不稱 LLM。

### E. Sparse Graph Route

定義 TF-IDF sentence vectors、k-nearest-neighbor／thresholded sparse adjacency、edge weight
與 centrality。說明 sparse 化的成本動機；不要寫不存在的 entity graph 或 knowledge graph。

### F. Route Reservation and Weighted RRF

這是主技術段落。必須清楚定義，但不得把 reservation 包裝成已證明的品質增益：

- 每路 top-K proposal；
- `min_per_route` reservation 如何保護 unique proposals；
- total candidate cap；
- weighted reciprocal-rank fusion 的公式；
- route rank、route score、agreement、reserved flag 與 fusion score 如何存進每句 provenance。

建議 **Fig. 2 — One-sentence provenance example**，用一個小例子顯示某句被 lexical rank 3、
semantic rank 8、graph 未選中，最後如何得到 fusion/provenance features。若頁數不足，移 Supplement。

### G. Provenance-Aware Candidate Salience

說明 selector 收到的不只是 lexical base score，也包含融合後的 evidence；這一段直接防止
舊版「semantic 只貢獻池子 membership」的問題。每個實際非零 feature 與 weight 都要能
對到 config 與程式；零權重 feature 不包裝成 active contribution。

### H. Task-Profile Selectors

- GovReport final profile：TF-IDF MMR，`lambda=0.7`。
- Multi-News final profile：Greedy-TFIDF。
- 其他非 final selector（包含既有 NSGA-II 實驗）：不進主文方法支線；品質、成本、seed、
  Pareto／mutation 與重現資訊完整移至 Supplement 與 repository。

要明說 selector 差異是在 dev 階段依 frozen task-profile policy決定，不是看 final test 後挑選。

### I. Feasibility and Output Construction

寫 upper bound、effective minimum、沒有 eligible sentence、oversized sentence、shortfall 與
infeasible row 的 fail-loud／record-and-continue contract。最終選句恢復原文順序；不是按
salience rank 拼接。

### J. Complexity

分 lexical、semantic encoding、sparse graph、fusion、Greedy/MMR 說明漸近與實測
成本關係。不要把單元件 timing 當整條 pipeline，也不要從 30-doc sample 推成所有硬體結論。

### 建議核心公式

1. lexical salience；
2. semantic similarity／centroid score；
3. sparse graph adjacency 與 centrality；
4. weighted RRF；
5. provenance-aware importance；
6. MMR criterion；
7. Greedy utility／marginal gain；
8. word-budget／feasibility constraint。

舊 ICACT 的 NSGA-II Eq. (7)–(10) 不搬進主文；其非 final selector 細節與負結果只保留於 Supplement。

## 7. V. Experimental Design（約 2.5–3 頁）

### A. Datasets and Frozen Roles

**Table II — Dataset and frozen policy** 至少包含：

| Dataset | Role | Source type | Dev | Dev-test | Official test | Length | Final exception |
|---|---|---|---:|---:|---:|---|---|
| GovReport | confirmatory primary | long single document | 681 | 292 | 973 | 500–650 words | official archive excluded one empty-reference validation row before partitioning |
| Multi-News | secondary boundary | multi-document news | 3,935 | 1,686 | 5,621 | 200–250 words | 12 infeasible final rows retained in denominator |

CNN/DailyMail、SciTLDR、PubMed 等不在 v2 dataset matrix，不寫成「尚未跑完」；它們只是
本次修訂未納入。

### B. Data Integrity and Preprocessing

交代 upstream revision/archive、checksum、sentence segmentation、canonical schema、異常列
政策、reference-blind partition 與 test freeze chronology。正文說原則，完整 SHA／manifest
放 Supplement／repository。

### C. Compared Systems

**Table III — System contracts**：Proposed、Lead、Random（10 frozen seeds mean）、TextRank、
LexRank、PacSum-TFIDF、PacSum-SBERT、SBERT-centroid、SBERT+MMR。欄位至少含 input
scope、representation、candidate construction、selector、task-specific training、cache state。

Greedy reference 另列為 reference-aware diagnostic，不能與 deployable systems 混成公平
baseline；NSGA-II 不放主文的 official ranking、selector 或 cost table，既有證據移 Supplement。

### C.1 Dataset-Specific Configuration Selection

必須明說「不做 task-specific fine-tuning」不等於「完全沒有 configuration selection」。
route weights 與 selector 都使用 validation 的 development partition 選定，並在 final test
前凍結：GovReport 為 `(lexical, semantic, graph)=(0.5,1,1)` 與 MMR（$\lambda=0.7$）；
Multi-News 為 `(1,1,2)` 與 Greedy。不得寫成「只調 selector」、`no tuning` 或兩資料集共用
完全相同的系統配置；同時須說明 SBERT 等神經模型參數全程固定、未做 fine-tuning。

### D. Evaluation Metrics

- Published-scale evaluator：Stanza sentence handling + Perl ROUGE-1.5.5 protocol。
- 報 ROUGE-1、ROUGE-2、ROUGE-L F1。
- `Mean ROUGE = (R-1 + R-2 + R-L)/3` 是 project-defined aggregate，不是官方第四個 ROUGE。
- 說明 corpus-level table aggregate 與 per-example paired mean 的最後小數可能不同。
- 抽取輸出對 abstractive reference 的 mismatch 放 threats；不可暗示 ROUGE 是完整人類品質。

### E. Statistical Analysis

寫主要 comparator、100,000 paired bootstrap、95% CI、raw／Holm-adjusted p-values、總比較
數量與 multiplicity family。不得只報顯著的 endpoints。區分 confirmatory GovReport 與
secondary Multi-News。

### F. E1/E2/E3 and Selector Protocols

- **E1**：official quality/evaluator parity。
- **E2**：reference-blind 30-document CPU sample；cold/warm 分開、3 fresh-process repetitions、
  wall time、CPU、peak process-tree RSS、q10/q50/q90 scaling。
- **E3**：兩資料集各五個事前指定 route/provenance ablations；固定 selector／pool contract。
- **Selector isolation**：同 candidates、salience、similarity、coverage inputs，只換 Greedy／
  MMR；使用 frozen dev，不當 final-test superiority evidence。既有 NSGA-II 完整結果移 Supplement。

### G. Implementation and Reproducibility

列 exact code commit/release tag、Python/dependencies、OS、CPU、RAM、worker／BLAS thread、
SBERT checkpoint revision、seed、commands、output paths。提醒 cold/warm cache state不可混比。

## 8. VI. Results（約 4–5 頁）

每個 subsection 都用相同順序：**研究問題 → 主數字 → paired uncertainty → 解釋 → 限制**。

### A. RQ1: GovReport Confirmatory Quality

放 **Table IV — GovReport official test**，九系統全部報 R-1/R-2/R-L/Mean。Proposed：
`0.58374 / 0.24711 / 0.54898 / 0.459943`。主要 paired comparator 是 SBERT+MMR，Mean
difference `+0.003700`，95% CI `[+0.002008,+0.005420]`，`p=0.000040`。必須同時說
R-L individual endpoint 沒有顯著，不能讓 macro 顯著代替所有 component 顯著。

### B. RQ2: Multi-News Boundary and Metric Trade-Off

放 **Table V — Multi-News official test**，同樣九系統與四欄。Proposed：
`0.45011 / 0.14314 / 0.41351 / 0.335587`。對 PacSum-TFIDF paired Mean
`+0.000091`，95% CI `[-0.001542,+0.001730]`，`p=0.907111`；R-1 `+0.002389`、
R-L `+0.004641`，R-2 `-0.006758`，三個 component 在 Holm-3 後均顯著。

正確結論是 trade-off／boundary，不是「排名第一所以勝出」。表註寫明 12 infeasible rows
仍在 5,621-row denominator。

### C. Confirmatory Paired Inference

放 **Table V — Primary paired comparisons**：dataset、comparator、endpoint、delta、CI、raw p、
adjusted p、decision。若英文稿空間允許，可另畫 paired-difference 95% CI 圖，GovReport
與 Multi-News 並排；該圖是閱讀輔助，不是尚未完成的必要證據。

### D. RQ3: Final-Selector Isolation

放 **Table VII — Greedy/MMR matched-input dev comparison**。清楚標 `development
evidence, not official-test result`。說明 GovReport 的 MMR 適合長文件 profile，Multi-News
則由 Greedy 較佳；不要把兩資料集 selector 差異寫成矛盾。NSGA-II 負結果移至 Supplement，
不是刪除或選擇性隱藏。

### E. RQ3: Route and Provenance Ablation

放 **Table VIII — E3 ablation**。主文至少列 full model 與五 variants 的 Mean ROUGE delta、
95% CI、Holm decision，兩資料集並列；R-1/R-2/R-L 全 endpoint表移 Supplement。逐項回答：

- 沒 lexical route 會怎樣？
- 沒 semantic route 會怎樣？
- 沒 graph route 會怎樣？
- 移除 weighted/provenance evidence、只留 index fusion 會怎樣？
- candidate/pool control variant 如何排除「只是候選數量變了」？

若某 route 增益小，寫 complementary/marginal contribution，不寫它是 dominant innovation。

### F. RQ3: End-to-End Efficiency and Scaling

放 **Table X — E2 cold/warm cost**。兩資料集都報，但把 q10/q50/q90、每 repetition 與
完整 scaling 細表移 Supplement；quality-cost/scaling 圖只在英文稿空間允許且不重複表格時
加入。正文必須說：

- 時間是固定 30-document sample 的整體／或 per-document 統計，依 evidence 欄位精確標示；
- cold 包含模型載入，warm 重用已載入資源；cold 只能和 cold 比、warm 只能和 warm 比；
- 主文只列 Lead、PacSum-TFIDF、SBERT+MMR 與 PAMR-ES (Ours)；其他 selector 成本移 Supplement；
- 所有結果是特定 CPU/hardware 的 measured evidence，不宣稱普遍速度倍數。

### G. Deterministic Provenance Walkthrough（主文已採用）

主文使用 frozen GovReport development prediction order 的第一筆文件，選例規則不參考
ROUGE 或人工品質；列出已選句與 reservation 保留但未選句的 route rank、agreement、
retention reason 與 final status。成功／失敗案例的人工作質性比較可留作補充或後續工作，
不可在看過結果後挑選幾個好看的例子再包裝成確認性證據。
cross-document evidence。不能看完案例後只挑好看的。

若投稿前不做，就不要虛構；把 `ROUGE-only and no fixed-rule qualitative/human assessment`
放入 Limitations。人評與 LLM baseline 不是已 frozen 自動實驗的一部分，若新增需另立
不改方法的 protocol。

## 9. VII. Discussion, Limitations, and Threats（約 1.5–2 頁）

### A. Why the Method Helps on GovReport

從長文件、跨 section、lead bias 較弱、MMR coverage／redundancy與 provenance 互補解釋；
只能說 evidence is consistent with，不把解釋寫成已證明的因果。

### B. Why the Advantage Does Not Transfer Uniformly to Multi-News

討論 news lead bias、短摘要、R-2 phrase coherence、graph×2／Greedy task profile，以及為何
R-1/R-L 改善不等於整體勝出。這是論文可信度的重要段落，不是需要藏起來的缺點。

### C. What Provenance Adds

區分「某句進 candidate pool」與「某 route evidence 真的影響排序」。用 E3 解釋 route
reservation、unique proposals 與 weighted RRF 的作用；不宣稱所有融合方法都沒 provenance。

### D. Dataset-Profile Selection Under Matched Conditions

直接由 matched frozen evidence 說明：GovReport 採 MMR、Multi-News 採 Greedy，是在相同
candidates／constraints 的 development comparison 下選定。route weights 與 selector 都是
dataset-specific profile 超參數，須一併揭露；討論聚焦本研究的實驗結論，不寫投稿版本沿革。

### E. Relationship to Supervised, Generative, and LLM Systems

說明 BERTSumExt、MatchSum、G-SEEK/G-SEEK-2、generative LLM 的 training/output/resource
regime 不同。不能拿本稿 ROUGE 直接宣稱勝過它們；也不要因沒有本地 LLM baseline 就稱
它們不公平。正確說法是本稿回答 no-task-specific-training extractive setting。

### F. Limitations and Threats to Validity

至少完整列出：

1. ROUGE 與 abstractive references 對 extractive output 的偏差；
2. 沒有 frozen human evaluation／semantic metric（若最後仍未補）；
3. 沒有同 protocol 的 generative LLM baseline，不能做 LLM superiority claim；
4. 只有 GovReport 與 Multi-News，外部效度有限；
5. fixed pretrained encoder 與 sentence truncation；
6. task-profile-specific selector／length policy；
7. E2 是 CPU、30-document fixed sample 與特定 hardware；
8. Mean ROUGE 是 project-defined aggregate；
9. Multi-News 12 infeasible rows與短文件／異常長句；
10. graph sensitivity／alternative centrality若只在 development screen 測過，不能包裝成
    confirmatory conclusion；
11. no universal SOTA claim。

## 10. VIII. Reproducibility, Availability, and Responsible Use（約 0.5–0.75 頁）

- code release URL、exact tag/commit、license／third-party notices；
- data只提供取得與 preprocessing 指令，不重散布不具授權的 corpus；
- configs、data policies、preregistrations、per-example predictions、analysis、failed attempts；
- evaluator、dependency/container、hardware、seed、one-shot commands；
- legacy test-tuned `runs/` 明確排除，不列入新稿 results；
- AI assistance disclosure 待作者群最後依實際使用程度定稿；AI 不列作者，作者負責驗證。

若 clean-clone 尚未完成，不可以在正文寫 `fully reproducible from a clean environment`；只能
寫 repository 目前實際提供到的程度。

## 11. IX. Conclusion（約 0.4–0.5 頁）

四段／四句即可：

1. 重述問題與 provenance-aware multi-route framework；
2. GovReport frozen confirmatory 正結果；
3. Multi-News metric trade-off 與 dataset-profile selector boundary；
4. 結論限縮為 task-profile-aware、no-task-specific-fine-tuning extractive setting，未來工作是更多 domains、
   human/semantic evaluation 與更有效的 selector，不重新宣稱 universal superiority。

Conclusion 不放新數字、不新增未測方法、不把 future work 當已完成貢獻。

## 12. 主文圖表總表與頁數取捨

### 12.1 必放表格

| 編號 | 內容 | 主文／補充 | 理由 |
|---|---|---|---|
| Table I | Dataset roles and partition sizes | 主文 | 固定資料角色與 split |
| Table II | Validation configuration counts | 主文 | 公開各方法實際觀察的搜尋預算 |
| Table III | GovReport R-1/R-2/R-L/Mean | 主文 | confirmatory core result |
| Table IV | Multi-News R-1/R-2/R-L/Mean | 主文 | boundary 與 R-2 負結果 |
| Table V | Paired CI and decisions | 主文 | 公正統計判斷 |
| Table VI | Output words/sentences/infeasible rows | 主文 | 排除只靠摘要較長提高 recall 的解釋 |
| Table VII | Selector isolation | 主文 | Greedy/MMR final-selector comparison；NSGA-II 完整負結果移 Supplement |
| Table VIII | Route/provenance ablation | 主文摘要＋補充完整 CI | 支撐並限制架構主張 |
| Table IX | Deterministic provenance walkthrough | 主文 | 展示 route-to-selector 可追蹤性 |
| Table X | Cold/warm runtime and memory | 主文摘要＋補充全表 | 回應效率與硬體質疑 |

這是 2026-09-04 中文工作稿的實際表格編排。若英文稿頁數吃緊，Table III/IV 可做成同一個
two-panel table，Table V 可併入兩個結果表的 paired rows；Table II 或 Table IX 可移至
Supplement。不可刪 Multi-News R-2、paired CI、輸出長度、消融或成本定義來省頁數。

### 12.2 必放圖

1. **Fig. 1** 完整架構圖。
2. Paired difference 95% CI 圖為英文稿的可選視覺化；目前 Table V 已完整呈現數值與判定，
   沒有該圖也不構成證據缺口。
3. Quality-cost/scaling 圖亦為可選；目前 Table X 與 Supplement 已提供 cold/warm、per-doc、
   RSS 與完整 scaling evidence。若加入，不能與表格重複到模糊焦點。

所有圖要 vector PDF/SVG 或足夠解析度，使用 color-blind-safe palette；不能只靠顏色區分，
caption 必須讓圖離開正文仍可理解。

### 12.3 Supplementary Material

- 完整 32-endpoint／Holm secondary comparisons；
- E3 所有 R-1/R-2/R-L/Mean endpoints、五 variants、兩資料集；
- E2 q10/q50/q90、三 repetitions、CPU/RSS、failure record；
- 完整 selector grid與 NSGA-II quality、runtime、seed stability／Pareto details；
- graph threshold／centrality development sensitivity（明標 development）；
- dataset checksums、manifest、config、prereg、artifact/evidence schema；
- failed attempts、legacy exclusion、reproduction commands；
- 額外 qualitative cases。

ICACT 差異矩陣只作內部 cover-letter 與 similarity 稽核資料，不放主文或 Supplement。

## 13. 內部品質問題 → 新稿處理矩陣

| 內部 concern | 新稿位置 | 現況／處理原則 |
|---|---|---|
| 三種 paradigm 與 NSGA-II 說不清 | Intro P3、Method A/H、selector table | 已可解決；NSGA-II 改 comparator並白話定義 |
| Related Work 太淺、缺 G-SEEK/G-SEEK-2 | Related Work A–D、taxonomy | 尚需完成正式 bibliography與逐篇事實核對 |
| code/reproducibility 不足 | Experiment G、Reproducibility、Supplement | evidence 已多；clean-clone/release tag仍是 blocker |
| extractive output對abstractive reference不公平 | Problem C、Metrics、Threats | 不可能消除；明確揭露與限縮 claim |
| ROUGE-only | Metrics、Qualitative、Limitations | 若不補人評/semantic metric，必須列 limitation |
| 缺 modern LLM comparison | Related Work E、Discussion E、Limitations | 不做 LLM superiority claim；本稿主表限 no-task-training extractive regime |
| 缺 qualitative/human analysis | Results G／Limitations | 建議 fixed-rule qualitative；human evaluation可選但須另凍結 |
| baselines 太弱 | System contracts、Tables IV/V | 已補 Lead/Random/TextRank/LexRank/PacSum/SBERT centroid/MMR |
| split/evaluator/checkpoint/hardware 不清 | Experimental Design A–G | 已有 evidence；主文必須真正寫出，不可只藏 repo |
| graph增益小／缺 isolation | E3、Discussion C | E3 已完成；小增益就寫 complementary，不誇大 |
| novelty overclaim／hybrid已存在 | Intro gap、Related gap、Contributions | 新意限 provenance contract + matched evidence，不說 first hybrid |
| oracle矛盾 | Problem D、Results | 全面改稱 metric-specific greedy reference diagnostic |
| runtime單位／公平性有問題 | E2、Table IX | cold/warm、完整 pipeline、hardware、CPU/RSS分開報 |
| PLM route幾乎沒真正影響排序 | Method F/G、E3 | 現行 provenance 已進selector；以 ablation證明，不靠敘事 |
| hyperparameters與 final Pareto selection不明 | Method H、Experiment G、Supplement | final selectors/config凍結；NSGA細節移 supplement |
| Stage 1/2訊號與三軌敘事不一致 | Fig. 1、Method C–I | 依現行程式重畫，不沿用舊流程圖 |
| graph不是 entity relation、也非 primary innovation | Method E、Discussion C | 明稱 sparse sentence-similarity graph/complementary route |
| dataset間方法不同 | Problem B、Method H、Experiment A | 明列 task profiles；不是事後挑 test，需寫凍結沿革 |

不是每個 reviewer 要求都必須機械式新增實驗。若研究 scope明確不比較 generative LLM，正確
處理是**限縮主張、說明 regime、列 limitation**；不能假裝已完成，也不能引用 reviewer 來辯解。

## 14. Cover Letter 架構

1. 稿名、Research Article、為何符合 IEEE Access scope。
2. 一句主結果與一個 boundary，不寫 universal SOTA。
3. 若投稿系統詢問相似的作者既有出版品，明確揭露 ICACT citation/DOI。
4. 用一小段說明本稿的獨立研究問題、現行架構、frozen datasets/evaluation 與 ICACT 的差異；
   不把整封信寫成 extension response。
5. 說明稿件未同時投稿、作者同意、利益衝突／data/code availability。
6. 若有正式可驗證 award，可最後一句提；沒有就不寫。

Cover letter 不提 ICT Express拒稿或逐條 reviewer意見，除非投稿系統明確要求 prior-submission history。

## 15. 頁數預算（目標 17–19 頁，含 references/bios）

| 區段 | 頁數 |
|---|---:|
| Front matter + Abstract | 1.0 |
| I. Introduction | 1.5 |
| II. Related Work | 2.0–2.5 |
| III–IV. Problem + Method | 4.0–4.5 |
| V. Experimental Design | 2.5 |
| VI. Results | 4.0–4.5 |
| VII–IX. Discussion/Repro/Conclusion | 2.0–2.5 |
| References + biographies | 2.0–2.5 |
| **Total** | **18–19.5** |

IEEE Access 雖非硬性 20 頁上限，但應以 20 頁以下為目標；超過時先移完整 endpoint、
config、Pareto、repetition 與額外案例到 Supplement，不可刪核心負結果或方法定義。

## 16. 建議實際寫作順序

1. 由 frozen JSON 自動產生 Tables I–X 的數值區塊；若保留可選 paired／cost 圖，亦由同一
   artifact 產生，鎖定數字來源。
2. 先寫 V Experimental Design 與 VI Results，逐句綁 evidence。
3. 再寫 IV Method，逐個 equation 對 code/config。
4. 寫 III Problem/Scope，固定 training/output/comparison regime。
5. 完成 II Related Work與taxonomy，逐篇查 primary source與retraction狀態。
6. 寫 VII Discussion/Limitations，先主動處理 Multi-News、ROUGE與LLM scope。
7. 最後寫 I Introduction、Abstract、Title與Conclusion，避免先寫出超過證據的故事。
8. 完成 funding first footnote 與 standalone cover letter；加入必要的 prior-work disclosure。
9. 做 clean-clone、equation-code-config-result audit、全文 similarity、英文與格式終審。

## 17. 投稿前逐句紅線檢查

每一個強動詞都要問：是 aggregate point estimate、paired statistical result，還是 interpretation？

- `outperforms`：只在 comparator、dataset、metric、protocol與統計都明確時使用。
- `significantly`：後面必須有 paired test、CI、p-value與 correction family。
- `efficient`：必須限定 E2 hardware、cold/warm與sample，不可泛稱。
- `robust/generalizable`：目前不足；改用 `shows a boundary across two task profiles`。
- `novel`：只指 provenance-preserving route-to-selector integration，不能指已知元件。
- `oracle`：禁止；改 `metric-specific greedy reference diagnostic`。
- `LLM`：本稿實際 semantic route是 sentence transformer／pretrained encoder，不是生成式 LLM。

## 18. 仍須完成的寫作 blockers

- [x] ICACT bibliography、DOI 與差異矩陣已保存，供 Related Work、cover letter 與
  similarity audit 使用；不作正文 extension 敘事。
- [ ] Related Work primary-source bibliography與retraction check。
- [ ] frozen artifacts 到 Tables I–X 數值區塊的自動生成；可選圖只有實際納入英文稿時才生成。
- [x] fixed-rule provenance walkthrough 已用 frozen GovReport development prediction order 第一列完成；不以 ROUGE 或案例好壞挑選，且 limitation 明列它不等於人評或 factuality 評估。
- [ ] clean-clone reproduction、environment lock/container、release tag。
- [ ] equation ↔ code ↔ config ↔ evidence ↔ manuscript一致性。
- [ ] source/PDF similarity audit、ORCID 與 funding；作者順序、affiliations、corresponding author、bios 與 5 個 keywords 已放入中文工作稿，仍待作者群簽認。
- [~] 中文工作稿已依實際用途加入 OpenAI Codex 揭露並列出受協助章節；英文定稿時仍須由作者群確認文字與引用形式。

完成上述 blockers 才是「可投稿」；目前則是**實驗已 freeze、可以照此藍圖正式寫稿**。

## 19. 投稿 readiness 與合規總表

### 19.1 現在能不能投稿

| Gate | 狀態 | 白話結論 |
|---|---|---|
| 科學實驗 | **通過** | GovReport／Multi-News official test、九系統、selector isolation、E2、E3 已完成，不再調參 |
| 研究誠信 | **repository 可見範圍未發現造假** | legacy test-tuned runs 已作廢；final configs、artifacts、負結果與 failed attempts 均保留 |
| 論文內容 | **進行中** | 中文獨立 Research Article 工作稿已更新，含固定規則 provenance case；英文正文、自動正式表圖與引用終審尚未完成 |
| Reproducibility artifact | **未通過** | clean-clone、完整 environment lock/container、release tag 與一鍵表圖仍待完成 |
| IEEE Access 行政 | **未通過** | 版型、作者順序、affiliations、corresponding author、bios 與 keywords 已進工作稿；similarity、prior-work disclosure、ORCID、funding 與作者群簽認仍待完成 |

所以現況是：**可以開始正式寫論文，但還不能直接按 Submit。**

### 19.2 已凍結且不得重開的內容

- GovReport 973-row official test 與 Multi-News 5,621-row official test。
- 九個本地同 pipeline systems。
- Greedy/MMR matched selector evidence；NSGA-II 負結果完整保留於 Supplement／repository。
- 兩資料集 E2 cold/warm runtime-memory-scaling。
- 兩資料集 E3 route/provenance ablation。
- final method、task profiles、length policies、主要 comparator與統計方法。

除非作者群正式重新定義一個獨立研究問題，不可因寫稿不方便或某一表不好看而重跑 test、
換 comparator、改 Mean ROUGE、刪除 Multi-News R-2 或排除 infeasible rows。

### 19.3 投稿包必須具備

- IEEE Access double-column source與內容一致 PDF；盡量控制 20 頁以下。
- 完整作者、公開 ORCID、affiliations、corresponding author、funding與 biographies。
- 投稿系統要求 3--10 個 keywords/phrases；目前中文工作稿使用 5 個精確關鍵詞。
- ICACT 在 Related Work 正常引用；cover letter 依 IEEE policy 揭露相似 prior work 與差異。
- 全文 similarity audit；不得複製舊稿文字、圖或表。
- code/data availability、exact release tag/commit與可重製環境。
- references accuracy／retraction check、英文與圖表 accessibility終審。
- AI assistance disclosure 已有符合目前實際用途的中文草案；英文投稿版仍由作者群定稿。

### 19.4 2026-09-04 中文工作稿版面實測

- 官方 IEEE Access LaTeX 雙欄版型可完整編譯；2026-09-04 最新中文工作稿共 12 頁、5.39 MB，逐頁 render 未見文字或表格重疊。
- 英文 abstract 為單一段落，以保守 tokenizer 計 233 words；Index Terms 為 5 個，分別符合最多 250 words 與投稿系統 3--10 keywords 的要求。
- LaTeX、BibTeX 與兩輪交叉引用重編譯完成，未出現 undefined citation/reference。
- 以上只代表工作稿的版型 gate 通過；英文定稿、source/PDF 完全一致、ORCID、funding、similarity、release tag 與作者群核准仍是投稿 blockers。

### 19.5 詳細稽核快照

2026-08-24 的逐項 experiment-integrity、artifact-hash、exception與 IEEE規則核對仍完整保留於
`IEEE_ACCESS_SUBMISSION_READINESS_2026_08_24.md`。它是**證據快照**；若其寫作架構、
待辦或狀態與本文件衝突，以本文件較新的 consolidation為準。
