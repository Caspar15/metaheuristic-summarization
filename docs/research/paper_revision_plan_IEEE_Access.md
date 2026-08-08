# IEEE Access 全面重建計畫

## 2026-08-06 selector-comparison update

已完成可執行的 Greedy／candidate-matched SBERT-MMR／NSGA-II selector swap，
以及 full-source SBERT centroid-only／SBERT-MMR baselines。新 pipeline 會重現
`all-MiniLM-L6-v2` 的 mean pooling + Normalize，並以 SHA-256 證明三 selector
收到相同候選與數值輸入。真實 Multi-News 3-row smoke 的 shared-objective 平均
utility 為 Greedy `0.3131`、NSGA-II `0.2990`、MMR `0.1111`；此處未計 ROUGE，
只能證明 NSGA-II 不會因較昂貴就自然優於 Greedy，不能作品質結論。正式去留須
依 `SELECTOR_COMPARISON_PROTOCOL.md` 在 Multi-News 與 GovReport validation
2026-08-06 frozen 200-row matched pilot 已提供第一個方向：candidate-matched MMR
在 R-1／R-2 顯著優於 Greedy，而 NSGA-II seed 2024 無顯著改善且總時間約 4.6×；
五 seed stability extension 後，NSGA-II 三個 ROUGE 的 seed mean 均低於 Greedy，
且選句 mean pairwise Jaccard 僅 0.639；selector 層因此確定以 MMR 為主線、
Greedy 為 reference、NSGA-II 為 comparator。這不等於整個方法已成立：仍須
full validation、full-source SBERT+MMR、PacSum 與第二 primary dataset 才能判斷
candidate router 是否足以構成 IEEE Access 的方法貢獻。

版本：2026-07-26 技術稽核版 ｜ 程式／資料狀態覆核：2026-08-02
適用範圍：ICACT 得獎論文的期刊擴充、ICT Express 拒稿稿件、metaheuristic-summarization 研究程式與既有實驗結果

文件治理：本文件是研究標準與投稿 gate 的唯一規範來源；`ACTION_PLAN.md` 是日常執行清單；`CODE_AUDIT_IEEE_Access.md` 與 `STRATEGY_ASSESSMENT.md` 只能作證據快照與衍生判斷。若數字衝突，以可重現 artifact、版本化程式、資料 fingerprint 與明確 evaluator protocol 為準，而不是以任何一份敘述文件為準。

## 0. 結論先行

目前版本不應直接改寫後投稿 IEEE Access。問題不是只有 related work、baseline 不夠或文字表達，而是論文主張、程式行為、資料切分與結果表之間存在多項實質矛盾。現有結果只能保留作探索紀錄，不能作為新稿主結果。

最嚴格且合理的處理方式是：

1. 凍結現有程式與結果為 legacy，不再追加表格。
2. 先重構研究 pipeline，使資料、候選生成、分數 provenance、最佳化、評估與 runtime 可被獨立驗證。
3. 所有超參數只在 validation set 決定；test set 鎖定後只執行一次正式評估。
4. 用一致的程式在本地重跑所有 baseline、ablation 與 proposed variants。
5. 只有當預先定義的效果、統計顯著性與 quality-cost Pareto gate 通過，才開始寫 IEEE Access 主文。

這不是「補實驗」而是一次 research reset。若不做 reset，高機率再次被拒，而且下一次 reviewer 可能會把問題定性為實驗有效性與研究可重現性不足。

目前唯一完成獨立重算的 F-0 是 legacy Multi-News 診斷：ExpB（K=20 Coverage）在相同 5,622 筆資料與目前的內部 evaluator 下為 R1/R2/R-Lsum = 0.4352/0.1405/0.3880；同一 245-whitespace-word 上限的 Lead prefix 為 0.4331/0.1453/0.3901。這表示舊系統只在 R1 高 0.0021，R2 與 R-Lsum 反而較低。這批 ExpB 是在 test set 選出的設定，故只能證明舊設計未通過 reality check，不能作為新稿結果。CNN/DailyMail 舊系統是 validation split，而文獻 Lead 是 test split，尚不能宣稱誰勝誰負。

證據狀態統一分為三層：

- **投稿級證據**：locked test、凍結設定、版本化 evaluator、完整 manifest 與可重現統計。
- **已重現的 legacy 診斷**：可用來找問題，但因 test tuning、舊 protocol 或研究設計已失效，不得進新稿主結果。
- **探索性估計**：非版本化 scratch script、抽樣 headroom、greedy reference 或跨論文數字；只能形成假設，不能形成結論。

## 1. 本次稽核依據

本計畫不是只轉述 reviewer，而是交叉檢查下列材料：

- Reviewer.docx：ICT Express 四位 reviewer 的完整意見。
- ICT Express 19 頁投稿 PDF：方法、公式、表格與結論。
- ICACT 6 頁得獎論文：期刊擴充的 prior publication 基線。
- metaheuristic-summarization/src、configs、scripts、tests。
- 既有 CNN/DailyMail、SciTLDR-AIC、Multi-News 資料與 run artifacts。
- IEEE Access 目前的投稿、重現性與 conference-extension 規範。

舊的初步提案、結案報告與「論文」資料夾依指示不作為技術判斷依據。

## 2. 已確認的 P0 致命問題

以下任一項未解決，都不應投稿。

### 2.0 ⚠️ 各 P0 的目前狀態（2026-08-02 逐項對 master 覆核）

> **以下 P0-01～P0-10 的內文描述的是稽核當時（commit `1b9fe6f`）的狀態，以現在式書寫。**
> 其中多項已在 Phase 1 重構中修好。沒有這張表，讀者會把已完成的項目當成待辦，
> 也可能誤以為投稿被 10 個問題同時擋住。
>
> 「新 pipeline」= `configs/phase1_mvp_multinews.yaml` 的 canonical 路徑；
> 「legacy」= `configs/1_*.yaml` / `2_*.yaml`，**刻意不修**，只供重現舊 artifact。

| # | 問題 | 狀態 | 依據 |
|---|---|---|---|
| **P0-01** | test set 上調參選模型 | 🟡 **legacy 成立；新 pipeline 已加守門** | `select_sentences.py` 的 `validation_pilot_only` 在 runtime 禁止 test split。**legacy 的 11 個 run 仍然作廢** |
| **P0-02** | CNN/DM 用 validation 卻比文獻 test | 🟡 **舊結果作廢；是否重建取決於實驗矩陣** | 尚無 CNN/DM canonical artifact。若保留為 sanity check，必須重建官方 test；若從新稿移除，則不得再引用舊 CNN/DM 勝負 |
| P0-03 | Stage-1 top-K 與實作不符 | ✅ **已修** | 各 route 在完整輸入排名；`route_top_k` 只定義 proposal depth，`min_per_route` 才是保留額 |
| P0-04 | Stage 2 沒有融合 BERT 分數 | ✅ **已修** | semantic route + `selector.salience_source: rrf_fusion`；實測改變 semantic 分數會改變選句 |
| P0-05 | SciTLDR oracle 與 multi-reference | ✅ **v1 已排除；重新納入才重開** | `preprocess_scitldr` 已存 `references: list`；v1 不跑 SciTLDR，因此 official conformance 不阻塞目前主線 |
| P0-06 | ROUGE-L 協定錯誤 | ✅ **已修**（parity 待驗） | `rougeLsum` + 手算 golden tests；與 published Perl ROUGE 的 parity 尚未驗證 |
| P0-07 | NSGA-II 參數與 config 不一致 | ✅ **已修** | `pop_size`/`n_gen`/`seed` 已接線、移除靜默 fallback、有 regression test |
| P0-08 | 公式與實作多處不一致 | 🟡 **部分修** | 已修：TF-ISF v2 改用非負平滑、graph 不再就地竄改、τ 已接線。**仍未修：`length_scores` 除以「文件內觀察最大值」而非論文的 `min(len/40,1)`；`centrality` 與 `novelty` 完全反相關** |
| P0-09 | Runtime protocol 不可重現 | 🟡 **程式已修，數字待重測** | `load_encoder()` 模型快取已修。**正式計時必須依鎖定 protocol 重測**（載入佔比在兩次量測間為 78% 與 93%，不穩定） |
| P0-10 | Data integrity 與 preprocessing | 🟡 **兩 primary validation 已修；freeze 後 splits 尚未讀取** | Multi-News 與 GovReport validation 的 canonical schema、pinned revision、health report、fingerprint、frozen policy、development partition 與防竄改守門已完成。test split 在人員簽署 freeze 前禁止；CNN/DM 是 Gate 3 後的條件式工作 |

**目前真正還擋著投稿的**：

1. 🔴 **Baseline Gate 2 尚未通過** —— Lead、Random、TextRank／LexRank 程式已進
   master；PR #15 centrality offline tokenizer hotfix、Linux CI 與 Multi-News 5,621-row
   TextRank／LexRank final-implementation rerun 已完成。但 PacSum、SBERT+MMR、
   GovReport 方法 runs、paired significance 及兩個 primary 的完整 baseline 矩陣仍缺。程式基礎與
   單一資料集實測不能替代完整 Gate 2，見 §9 Go/No-Go
2. 🔴 **第二個 primary 的資料層與 lexical D1 已建立，但證據矩陣仍不完整**；GovReport
   canonical validation／dev／dev-test 已凍結，A1 與 12-config lexical/objective dev
   screen 已完成。全文候選 macro `0.415585` 高於同協定 Lead／Random point estimate，
   但 graph／semantic、強 baseline、paired significance 與 dev-test promotion 仍為零
3. 🔴 **P0-08 的殘留** —— `length_scores` 與 `centrality`/`novelty` 兩項
4. 🟡 **P0-02 是條件式工作** —— 保留 CNN/DM sanity 才需重建 official test；不保留就必須刪除舊結果與主張
5. 🟡 P0-01 的 legacy artifact 永久作廢（不是待修，是既成事實）
6. 🟡 P0-06 的 published-protocol parity、P0-09 的正式計時，以及 validation-frozen Pareto/output policy

> ⚠️ **「P0 修好」不等於「可以投稿」。** §9 的 Go 條件要求相對強 baseline 有一致正向效果，
> D1 的兩個 lexical/objective cells 顯示明確 domain difference：Multi-News 全文候選
> macro `0.320912` 仍低於 Lead `0.326291`；GovReport 全文候選 `0.415585` 高於
> Lead `0.399232` 與 Random `0.408844` 的 dev point estimate。這支持候選召回是主要
> 病因，但全文 dense 搜尋不是 final architecture，且尚無 strong-baseline／paired
> significance。新增的 Multi-News cheap-multiroute 證據顯示 sparse graph G02 以
> mean/max `48.03/60` 的受控 pool 達到 macro `0.324305`，相對純 lexical base
> `+0.013952`、也高於全文 lexical，但仍低 Lead `0.001986`，主要缺口在 R-2。
> GovReport cap-aware section follow-up、兩資料集 semantic 與 governed Gate 2 未完成前
> 不能提前下最終結論。

> GovReport cheap-multiroute 隨後完成 11 個有效 runs；G02 hard pool macro `0.404282`
> 高於 Lead、低於 Random。G07 full/soft pool `0.414831` 但成本約 G00 `13.28×`，且仍
> 略低全文 lexical L10。uncapped section guard 因 mandatory 77 > cap 60 正確失敗；
> 原 failure 保留，容量推導的 cap-aware G11b 已獨立預註冊，不能覆寫原配置。
> G11b 隨後在 prereg commit 後完成 681/681 rows，pool max 60、macro `0.404182`；它只
> 證明 cap-aware section guard 可行且點估計小幅正向，不構成 promotion。

> Multi-News semantic S00 隨後得到 macro `0.322404`：有 semantic unique candidates，
> 但仍低 graph G02 `0.001900`，selection 成本約 graph G00 `10.37×`；S01 直接 semantic
> selector 更低 `0.022771`，該接法刪除。原 S02 因三路 reservations + guard 超過 cap
> 失敗；容量推導的兩-primary S02b 已在 GovReport semantic 分數前獨立預註冊。
> Multi-News S02b 隨後完成：macro `0.328077`，相對 graph G02 `+0.003772`、Lead
> `+0.001786`，pool mean/max `52.09/80`。這是首次受控池配置在該 dev 點估計高於 Lead，
> 但 S02b 同時改 total/guard cap、R-2 仍落後，尚不能把提升歸因 semantic 或宣稱勝出。
>
> GovReport semantic 原 family 也已完成：S00 macro `0.407203`，比 graph G02 高
> `0.002921`，但仍低 Random `0.001641`、全文 lexical L10 `0.008382`，selection 約
> graph G00 `15.15×`。S01 又比 S00 低 `0.057371`，因此 semantic-direct selector
> 接法已有跨資料集刪除證據；整條 semantic route 只因 S02b 正訊號暫留，仍需 paired
> evidence 與 capacity-matched route ablation。
> GovReport S02b 隨後得到 macro `0.417862`，高全文 lexical L10 `0.002277`、graph
> G07 `0.003031` 與 Random `0.009018`。兩 primary 的 S02b 都是目前 proposed 最高
> point estimate，支持繼續做 paired/matched ablation；但 GovReport 成本約 graph G00
> `18.28×`，且尚未對 strong baseline，不能直接凍結成最終架構。
> Multi-News capacity-matched route removal 隨後得到：S02b 相對移除 semantic
> `+0.003868`、相對移除 graph `+0.005358`。這排除「只因 cap 變大」的完整解釋，
> GovReport 同比較為 `+0.014268/+0.011481`，兩路跨資料集 point estimate 同號；
> 但仍須 31-config multiplicity-corrected paired inference。GovReport 無 semantic 只需
> `163.77 s`、完整 S02b `1,106.27 s`，semantic 必須另證明 quality-cost 值得 always-on。

### P0-01. Multi-News 在 test set 上調參與選模型

證據：

- scripts/quick_tune.ps1 直接以 data/processed/multi_news_test.jsonl 執行 ExpA、ExpB、ExpC。
- scripts/run_missing_experiments.ps1 也直接在同一 test set 比 K、ablation 與 fusion variants。
- 論文再從這些 test 結果選出 Coverage、K=20 作為 best configuration。

這是 test-set overfitting。即使沒有梯度訓練，只要用 test 結果選 K、權重、objective 或模型，test 就已被用於模型選擇。

必做修正：

- Multi-News validation 只用於開發、調參與選模型。
- test 在 configuration freeze 後才解鎖。
- 新增 immutable experiment manifest，記錄 split fingerprint、config hash、commit hash 與執行時間。
- 正式 test 不允許 rerun 後挑最好 seed；所有 seed 與 aggregation rule 必須預先固定。

驗收條件：

- 每張主表可追溯到一個 locked config。
- log 能證明該 config 在 test 執行前已由 validation 選定。
- test 結果不再回饋超參數調整。

### P0-02. CNN/DailyMail 使用 validation，卻和文獻 test 數字直接比較

證據：

- Table 1 的 NSGA-II ROUGE-1 0.3208 對應 runs 中 13,368 筆 validation_full。
- Table 4 的 0.3513 也對應 13,368 筆 validation run。
- 本地正式 CNN/DailyMail split 為 train 287,113、validation 13,368、test 11,490。
- 論文將 validation 結果與 BERTSumExt、Lead-3 的 published test 結果放在同一比較論述中。

必做修正：

- 重新建立官方 split，validation 用於調參，11,490 筆 test 用於正式結果。
- Lead-3、TextRank、PacSum 等可執行 baseline 必須在同一份 test input、同一 segmentation、同一 length rule 與同一 evaluator 下重跑。
- 只引用文獻數字的 supervised 方法必須放在「reference-only」區塊，不可做顯著性比較或宣稱直接勝負。

驗收條件：

- 主表每一列都有 split、樣本數、輸出長度與來源欄。
- 不再混用 validation 與 test 結果。

### P0-03. 論文的 Stage-1 top-K 與程式實作不是同一個方法

論文宣稱：

- NSGA-II 從 Pareto-front occurrence frequency 取 top-K。
- PLM 在全文排名後取 top-K。
- Graph 在全文排名後取 top-K。
- 三路使用相同 K，union 最多為 3K。

實際程式：

- select_sentences.py 先用 base statistical score 把輸入裁成 hard candidate pool。
- NSGA-II、BERT 與 Graph 隨後只在這個 pool 上運作，並輸出 length-bounded summary subset，不是 top-K ranking。
- Stage-1 的 max_tokens=300，因此三路實際平均選出的句數不同。
- full Multi-News run：Base 平均 14.08 句、BERT 8.36 句、Graph 8.92 句。
- K=40 的 full run union 平均只有 23.71 句、最大 49，不是論文所述約 120 句。
- utils_fusion.py 的 src_k 是對已按原文位置排序的 selected_indices 切前 K；它取得的是較前面的句子，不是 route ranking 的 top-K。

這會讓 K=20 與 K=40 的實驗解釋失效，也使「三路獨立互補」不成立。

必做修正：

- 將 candidate generation 與 final summarization 拆成兩個不同 API。
- 每一路都必須在相同的完整輸入集合上輸出 ranked candidate records。
- candidate record 至少包含 original_index、route、raw_score、normalized_score、rank、model/checkpoint。
- NSGA-II 若要採 Pareto occurrence frequency，必須真正保留完整 front 並計算 frequency；不可拿 scalarized final subset 代替。
- union 必須依 original_index 合併，保留所有 route provenance 與 score。
- K 必須在 rank 尚未按原文位置重排前截取。

驗收條件：

- 每篇文件每一路恰好輸出 min(K, N) 個不同候選，除非有明確的資料品質例外。
- union size 滿足 max(K, available) 至 3K 的可解釋範圍。
- 測試可驗證 K 截取依 rank，而不是依原文位置。

### P0-04. Stage 2 沒有融合實際 BERT 分數

論文 Equation 19 與文字宣稱 Stage 2 融合 statistical score 與 PLM semantic score。

實際程式：

- fast_fused.py 在 Stage 2 重新計算 TF-IDF document-centroid similarity。
- optimizer_dispatch.py 將這個 TF-IDF 分數命名為 w_bert 或 w_sem。
- Stage-1 BERT 的 raw score 沒有寫入 predictions，也沒有送到 Stage 2。
- BERT 的唯一作用是決定某句是否進入 union；Stage 2 不知道它的 BERT score 或 rank。

因此目前的 w_bert ablation 不是 BERT weight ablation，Table 8 的 Semantic strategy 也不是論文所描述的 PLM semantic weighting。

必做修正：

- 將 w_bert 改為語意明確的 channel 名稱，例如 w_plm 或 w_tfidf，不得混用。
- Stage-1 PLM route 必須輸出每句實際 embedding-based score。
- Stage 2 必須明確選擇：
  - provenance score fusion：直接融合各 route 保存的校準分數；或
  - Stage-2 re-encoding：在 union 上重新用指定 sentence encoder 計算語意分數。
- 兩者不可在公式與程式中混稱。
- 若採 sentence encoder，使用適合 sentence similarity 的 checkpoint，不再把 raw BERT centroid 當成強 PLM summarizer。

驗收條件：

- 單元測試可人工構造 BERT score，證明調整 w_plm 會按預期改變 final importance。
- run artifact 可逐句追溯 final score 的每個分量。

### P0-05. SciTLDR oracle 定義錯誤，且 multi-reference 被錯誤串接

論文所稱的 oracle 是「所有 source sentences 與 reference 的 ROUGE-1 F1」。這不是 oracle，只是一個 full-source baseline。因為加入大量無關文字會降低 precision，較短的 extraction 當然可能超過它，不存在邏輯矛盾。

必須先區分兩件事：實際系統輸給 oracle 本來就正常，oracle 代表在指定輸出限制下的可達上界；真正致命的是目前以錯誤 oracle 和錯誤 multi-reference 評估，把 0.2391 解讀成有競爭力的結果，因而掩蓋了系統可能連公平的無監督 baseline 都沒有追上。

程式另有第二個問題：

- preprocess_scitldr.py 原先把 target list 串成單一 reference；已於 2026-07-26 修正為 canonical `references: list[str]`，並加入防回歸測試。SciTLDR 官方 evaluator wrapper 尚未完成，因此程式會拒絕把一般 ROUGE 誤標成 `scitldr_official`。
- production inference 已不再輸出或使用 gold reference；evaluation 必須另給 `--gold`，以唯一 ID 完整對齊，缺列、多列或不一致都直接失敗。舊的 `candidates.recall_target` 因依賴 gold 決定 K，已禁止在 production selector 使用。
- SciTLDR 的 target entries 是多個 reference／annotation，不應視為同一篇長摘要。
- legacy scitldr_test.jsonl 共 618 篇；直接取每篇既存 rouge_scores 的最大值後再平均為 0.4311。這仍不是 official evaluator 的 oracle，只能證明資料中本來有 sentence-level candidate signal，而稿件另算的 full-source 0.1357 不能稱為 oracle。

官方 SciTLDR 論文 Table 3 的 AIC test max-ROUGE 結果如下。官方 `cal-rouge.py` 對每篇 prediction 與各 gold TLDR 計分，以最高 ROUGE-1 選定一個 reference，再從該同一 reference 報 R1、R2、RL；不是三個 metric 各自取最樂觀 reference：

| Method | Training regime | R1 | R2 | RL |
|---|---|---:|---:|---:|
| Extractive oracle | Oracle upper bound | 52.4 | 29.0 | 42.9 |
| PACSUM | Unsupervised extractive | 28.7 | 9.8 | 21.9 |
| BERTSUMEXT | Supervised extractive | 36.2 | 14.7 | 28.5 |
| MatchSum | Supervised extractive | 38.6 | 16.4 | 30.1 |
| CATTS | Supervised abstractive | 44.9 | 22.6 | 37.3 |

來源：Cachola et al., 2020, Table 3，https://aclanthology.org/2020.findings-emnlp.428/ 。官方 extractive oracle 是在每篇文章中選一個 source sentence，使其對任一 gold TLDR 的 ROUGE 最大，不是把全文當摘要。

目前稿件的最佳 SciTLDR-AIC R1 為 0.2391，即百分制 23.91。若它能在完全相同協定下成立，則診斷差距為：

- 對 official extractive oracle：-28.49 R1 points，只達 oracle 的 45.6%。
- 對同為無監督 extractive 的 PACSUM：-4.79 R1 points。
- 對 BERTSUMEXT：-12.29 R1 points；對 MatchSum：-14.69 R1 points。

但這些差距目前只能標為 red-flag diagnostic，不能當正式勝負結論，因為現有 23.91 使用了錯誤的 reference 串接，而且輸出句數／長度、files2rouge 與官方 max-R1-reference protocol 尚未對齊。修正後分數可能上升或下降，必須重跑才知道。

這張官方表同時否定「SciTLDR 只能報 ROUGE-1」的說法。正式結果必須報 R1、R2、RL，並清楚註明同一 reference 由最大 R1 選定；若另報 mean-over-references，只能作 supplement，不能和 Table 3 混比。

必做修正：

- 依 SciTLDR 官方資料與評估程式確認 A、AIC 的 split、target、source_labels、rouge_scores 語義。
- multi-reference 評估採官方 aggregation rule；不可自行串接。
- 主表採官方的一句 source-sentence extraction constraint、files2rouge 與 max-R1-reference protocol；另做不同 budget 時必須改名為獨立設定。
- 以同一 evaluator 本地重跑 PACSUM、Lead-1、BERTSUMEXT／MatchSum（能可靠重現者）與 proposed method；不可只把稿件的 23.91 和 published numbers 並排。
- 至少報 R1、R2、RL，並保存每篇、每個 reference 的原始分數及最後 aggregation 結果。
- extractive oracle 必須在和系統相同的句數或 token budget 下定義：
  - 小文件可 exhaustive best subset；
  - 大文件可使用 greedy reference，但名稱不得含 oracle，也不得稱 exact upper bound；
  - 若 dataset 提供 oracle labels，優先遵循官方 protocol。
- 將 full-source score 改名為 full-source baseline，或直接刪除。

驗收條件：

- 系統不可能超過同一 metric、同一 budget 下的 exact extractive oracle。
- reference 數量與官方資料一致。
- 論文清楚區分 exhaustive exact oracle、metric-specific greedy reference 與 full-source baseline。
- official test split 上重現 AIC one-sentence oracle R1 約 52.4；若無法在合理 rounding tolerance 內重現，SciTLDR 實驗不得進主文。
- proposed method 在相同 no-task-training 條件下至少不低於重跑的 PACSUM；否則 SciTLDR 不得支持「跨領域有效」或「具競爭力」主張。

### P0-06. ROUGE-L 協定錯誤，既有跨論文比較無效

src/eval/rouge.py 使用 rougeL，而非多句摘要慣用的 rougeLsum。既有 CNN/DailyMail R-1 與 R-L 的異常落差正是警訊。

診斷結果：

- 在既有 Multi-News full predictions 的前 1,000 筆上，以同一 Google rouge-score 套件重算：
  - rougeL 約 0.2070。
  - rougeLsum 約 0.3928。
- 這是診斷值，不是新主結果，但足以證明 metric choice 會實質改變結論。

其他問題：

- 系統摘要只以空白串接，沒有保留句界；rougeLsum 必須保留 newline boundaries。
- 引用他人表格數字時，沒有證明 tokenizer、stemming、length 與 ROUGE implementation 相同。

必做修正：

- 主報告使用 ROUGE-1/2/Lsum F1，並保留 sentence boundary。
- 若為和特定舊文比較而額外報 ROUGE-L，須另列，不能混用。
- 固定 evaluator version、use_stemmer、tokenizer、multi-reference rule。
- 每個 baseline 都用同一 evaluator 重算。
- 產出 per-example scores 以供 paired bootstrap。

驗收條件：

- evaluator golden tests 覆蓋 rougeL 與 rougeLsum 的已知差異。
- paper、code、metrics.csv 的 metric 名稱完全一致。

### P0-07. NSGA-II 實際參數與保存的 config 不一致

證據：

- configs/1_Base_NSGA2.yaml 設 pop_size=40、n_gen=50、seed=2024。
- optimizer_dispatch.py 呼叫 nsga2_select 時沒有傳 pop_size、n_gen 或 seed。
- nsga2_select 因此使用函式預設 100、100、seed=None。
- config_used.json 保存的是 40、50、2024，但它不是 effective configuration。
- fast_nsga2 也沒有從 config 傳入 population、generation 與 seed。
- NSGA-II 失敗時會 catch 所有 Exception 並靜默 fallback 到 greedy；artifact 仍可能被標為 NSGA-II。

必做修正：

- 參數必須由 validated config 明確傳入。
- 分開記錄 requested config 與 effective config；兩者不一致即 fail。
- 禁止 silent fallback。研究模式下任何 optimizer failure 必須終止 run。
- 若產品 demo 需要 fallback，輸出需標記 actual_method、fallback_reason，且不得進入研究結果。
- 明確記錄 crossover type、crossover probability、mutation probability per individual、mutation probability per variable、duplicate elimination、termination rule。
- 至少 5 個預先固定 seed，報 mean、standard deviation 與 paired uncertainty。

驗收條件：

- 同 seed 重跑得到相同 indices。
- 不同 seed 產生可追蹤的 stochastic distribution。
- 故意移除 pymoo 時研究 run 必須 fail，不得產生假 NSGA-II 結果。

### P0-08. 公式與 feature implementation 多處不一致

TF-ISF：

- 論文 TF 是 term count 除以句長；程式 v1 使用 raw count。
- 論文 ISF 是 log(S/st)；程式使用 log(N/(1+sf))，常見詞可能得到負值。
- 程式再以文件內最大 sentence score 相除，並非論文公式。

Length：

- 論文是 min(sentence_length/40, 1)。
- 程式先 cap 40，再除以該文件觀察到的最大長度；若文件最長句只有 20 詞，該句仍得到 1.0。

Similarity：

- 論文 Stage 1 說用 TF-ISF vectors。
- 程式實際用 scikit-learn TF-IDF vectors。

Graph：

- threshold 沒有由 graph_params 傳到 feature_builder 的 graph score。
- PageRank adjacency 保留 diagonal self-similarity 1.0，self-loop 會主導 sparse graph。
- threshold function 會原地改寫 similarity matrix，可能連帶改變後續 optimizer 的 coverage 與 redundancy。
- dangling-node 的註解說改為 uniform row，但程式並未真正填 uniform distribution。

Stage 2：

- 論文說 coverage 與 redundancy 使用 PLM embeddings。
- fast_nsga2 實際使用 TF-IDF similarity。

必做修正：

- 先建立一份唯一的 method specification，再依 specification 重寫程式與公式。
- 每個公式都要有對應的 unit test 與數值 example。
- 不保留 v1/v2 模糊雙軌；正式研究只能有一個 canonical implementation。
- 若選擇 TF-IDF，就誠實寫 TF-IDF；若選擇 TF-ISF，需實作一致的 vectorizer。
- Graph 應 copy matrix、移除 diagonal、正確處理 dangling nodes，並以 NetworkX 或獨立 oracle 實作做 parity test。

驗收條件：

- method-to-code traceability table 中，每個 equation 都能指向單一函式、config key 與測試。
- 用手算小例子逐項核對。

### P0-09. Runtime protocol 不公平且無法重現

目前問題：

- encoder_rank.py 每篇文件重新載入 tokenizer 與 model，將 model loading 混入 per-article inference。
- 沒有固定 device、CPU thread、batch size、warm-up、precision 或 max sequence distribution。
- 既有表把純 meta-heuristic selector 與包含 PLM 的 full fusion 混成一個 efficiency 敘事。
- active Multi-News full run 的記錄時間約為：
  - Base 9,408.8 秒。
  - PLM 5,811.0 秒。
  - Graph 77.4 秒。
  - Union 1.0 秒。
  - Stage 2 17,019.4 秒。
  - 總計約 32,317.5 秒，即 5.75 秒/篇，而非單一 selector 的毫秒級敘事。

必做修正：

- 模型只載入一次；分開報 cold-start 與 warmed inference。
- 對每個系統使用相同硬體、batch policy、資料順序與計時範圍。
- 同時報 end-to-end latency、throughput、peak RAM、peak VRAM；可行時加 energy。
- 純 statistical variant 與 full PLM variant 分開定位。
- runtime 至少重複 5 次；排除第一次 warm-up，報 median、mean、standard deviation、P95。
- quality-latency Pareto 圖必須以完整 pipeline 的成本作圖。

驗收條件：

- 計時腳本能從空 process 一鍵執行。
- 每個 latency 數字都附硬體與 software stack。
- 不再用 component time 代替 end-to-end time。

### P0-10. Data integrity 與 preprocessing 未被控制

已觀察到：

- Multi-News test 有 1 筆零句文件。
- test_4241 有 3,295 句、約 145,099 whitespace tokens，是極端 outlier，需和原始資料核對。
- CNN/DailyMail 舊 validation.full 平均約 21.46 句、最大 25，顯示輸入曾被截斷為前 25 句，但論文未揭露。
- simple regex、NLTK Punkt 與資料集原始句界混用。
- token 在程式中其實是 whitespace-delimited word，卻被稱為 token。

必做修正：

- 新增 dataset validation report：row count、unique ID、空文件、句數/字數分布、reference count、checksum。
- 明確定義 sentence segmentation；對每個資料集固定版本與設定。
- max_sentences 截斷必須是顯式 experimental factor，不可隱藏在 preprocessing。
- 將 whitespace count 命名為 words，真正 token budget 則綁定明確 tokenizer。
- 將 `max_words`、`max_sentences`、`max_model_tokens`、`candidate_budget` 與 `compute_budget` 分成不同欄位，不得再以單一 `max_tokens` 混用。
- 對空文件與 outlier 先訂規則，再看 test 結果；不得事後挑選排除。
- Multi-News 應同時檢查 2024 年提出的 Multi-News+ 清理問題，至少做 contamination/irrelevant-document sensitivity。

驗收條件：

- 每個 split 都有 machine-readable data card 與 checksum。
- 正式 run 遇到不符合 schema 的 row 必須 fail 或依預先定義規則標記，不可靜默跳過。

## 3. 研究定位：不能再用「三個舊模組相加」當主要新穎性

Reviewer #4 的判斷基本正確：NSGA-II、centroid PLM ranking、thresholded TextRank 本身都不是新方法。若只把三者串起來，IEEE Access 仍可能認為技術貢獻不足。

### 3.1 建議的新核心問題

推薦將研究問題改為：

「在不做 task-specific training 的條件下，如何以可重現、可解釋的候選 provenance 與多目標選擇，取得 extractive summarization 的 quality-cost Pareto 改善？」

這個定位的必要條件：

- 不再主張勝過所有 PLM 或 LLM summarizers。
- 主比較群是相同 no-task-training regime。
- supervised extractive 與 generative LLM 系統是 reference groups。
- 方法貢獻必須超過固定 K 的 union。

### 3.2 推薦的真正方法擴充

優先研究下列設計，通過 validation 後再決定是否成為主方法：

1. Provenance-preserving fusion  
   每個候選保留 statistical、graph、sentence-encoder 的 rank 與 calibrated score，Stage 2 直接利用來源資訊，而不是只看 union membership。

2. Budget-aware adaptive routing  
   依文件長度、語意重複度與 graph density 決定是否啟用昂貴 PLM route，以及各 route 分配多少 candidate budget。這才有機會把 efficiency 變成方法的一部分，而不是事後計時。

3. Pareto output policy  
   明確定義從 Pareto front 選解的方法。若使用 scalarization，權重只能在 validation 決定；也可比較 knee-point、hypervolume contribution 與 reference-point selection。

4. Exact constraint semantics  
   句數、word budget、model-token budget 必須是不同 constraint，不再混稱 L_max。

5. Objective normalization  
   importance、coverage、redundancy 的尺度需先校準，避免因 sum 與 mean 的量級不同，使 sentence count 或某一 objective 隱性支配結果。

### 3.3 三個可接受的研究路線

路線 A：方法型，推薦

- 加入 provenance-aware fusion 與 budget-aware routing。
- 主貢獻是 adaptive quality-cost optimization。
- 需要完整重構與新實驗，但最有 IEEE Access 說服力。

路線 B：嚴謹實證型，風險較高

- 不宣稱新演算法，改為系統性比較 statistical、graph、PLM 與 meta-heuristic 的效益與成本。
- 需要非常強的 reproducibility、統計分析與跨資料集結果。
- 若沒有清楚的新發現，仍可能因 novelty 不足被拒。

路線 C：刪除弱模組

- 若 corrected ablation 顯示 PLM 或 Graph 的 95% CI 包含 0，將該模組降為 optional 或刪除。
- 不可為了保留原標題而保留沒有實證貢獻的元件。

### 3.4 總體可救性判斷

目前稿件與程式不能靠修正文字、增加兩三個 baseline 或微調權重救回。原因不只是分數低，而是目前分數無法回答「哪個元件有效、為什麼有效、在什麼條件有效」。若維持固定三路 union、舊 BERT centroid、thresholded TextRank 與預設 NSGA-II，只修 evaluator 後投稿，仍屬 No-Go。

但研究問題仍有可發表空間。應保留的核心不是現有實作，而是：

- no-task-training／offline extractive summarization。
- 明確的 information coverage、redundancy 與 cost trade-off。
- 可追溯的 candidate provenance。
- 在長文件或多文件上依輸入特徵動態分配計算資源。

必須放棄的預設立場：

- 三個 route 一定互補。
- PLM route 一定應該每篇啟用。
- NSGA-II 一定優於 greedy、MMR 或 deterministic submodular selection。
- 同一套固定 K 與 objective 可以合理涵蓋單句、三句與 245-word 摘要。

判斷原則是「保留研究假設、重寫實作；由 validation 實驗決定模組生死」，而不是維護既有架構外觀。

### 3.5 資料集與方法的適配性

現有 ICT Express 稿件明確使用三個資料集，不是兩個：CNN/DailyMail、SciTLDR-AIC、Multi-News。三者不應被包裝成等價的泛化驗證。

| Dataset | 與核心多目標方法的適配 | 建議角色 | 原因 |
|---|---|---|---|
| Multi-News（原版） | 高 | 主要 benchmark | 多文件、多句、長 budget，coverage、cross-document redundancy、route cost 都有實際意義；保留原版才能與既有文獻直接比較，但必須重建 document boundaries。 |
| Multi-News frozen U+FFFD clean | 高 | v1 必跑 paired sensitivity | 在共同 5,549 rows 比較 main／clean，只回答 replacement-character rows 的影響；不得取代 5,621-row main。 |
| Multi-News bad-retrieval-removed / Multi-News+ | 高 | reserve，v1 不跑 | 可檢查 irrelevant-document contamination，但與 frozen U+FFFD clean 是不同問題；不得混成同一 sensitivity。 |
| GovReport | 高 | 建議新增的第二主要 benchmark | 長單文件、summary 也長，重要資訊分散，適合測 adaptive routing、全局 coverage 與長度擴展性。 |
| CNN/DailyMail | 中低 | Gate 3 後 optional sanity | lead bias 強、文件相對短；不阻塞核心 Gate，不用於方法選擇。 |
| SciTLDR-AIC | 低 | v1 排除 | 官方主設定只抽一個 source sentence，redundancy objective 幾乎恆為零，多目標搜尋退化；與核心多句方法不匹配。 |
| Multi-XScience | 中 | 科學多文件的可選外部驗證 | 題目與 graph/cross-document 關係相符，但資料與 reference 偏 abstractive，不能取代 extractive-aligned 主 benchmark。 |

資源有限時的推薦配置：

1. 兩個 primary datasets：GovReport（長單文件）與原版 Multi-News（多文件）；frozen U+FFFD clean 作必跑 paired sensitivity。bad-retrieval-removed／Multi-News+ 不屬 v1。
2. CNN/DailyMail 不列入核心 Gate；只有兩個 primary 已通過 validation Gate 3 且計算資源允許，才以 frozen method 跑官方 test 11,490 作次表／supplement sanity。
3. SciTLDR 不列入 v1 執行矩陣，不產生新結果表。若投稿前要恢復，必須在看結果前修改 `ACTION_PLAN.md` §2.0，並先通過 official evaluator conformance。

若不想新增資料集，最低可行配置是 Multi-News 為主、CNN/DailyMail 為次、SciTLDR 降為 stress test；但對 IEEE Access 的說服力低於加入一個真正的長文件 benchmark。

資料集角色、primary endpoint 與保留／排除規則必須在 test 前寫入 manifest，不能看完 test 才把輸的資料集降成 appendix。

### 3.6 可救的新架構

完整 schema、模組介面、task-profile objective matrix、route 刪除條件與 freeze gate，以 `ARCHITECTURE.md` 的 Target Architecture v1 為技術規格。該規格在兩個 primary validation pilots 通過前仍是候選架構，不得在論文中寫成已證實貢獻。

建議把「three-stage fusion」改成「cost-aware provenance-preserving multi-objective extraction」：

1. Independent candidate routes  
   lexical、semantic sentence encoder、sparse graph/structure 各自在同一完整輸入上輸出 ranking，不可先被共同 candidate pool 截斷。Position、document 與 section 只能作可消融的 strata coverage guard，不是假裝獨立語意 route。

2. Structure-aware graph  
   由無界 dense graph 改為稀疏 kNN／可承受的 block graph；edge 可使用 lexical/semantic similarity 與已存在的 document/section metadata。Entity/coreference 只有在確實實作並通過 ablation 後才能列為方法，不能先寫進貢獻。

3. Calibrated provenance fusion  
   每句保留 route rank、raw score、校準後 score、route agreement 與成本。融合器使用這些證據，而非只知道某句是否出現在 union。

4. Budget-aware adaptive router  
   依句數、文件數、section 數、廉價 lexical redundancy／topic dispersion、cheap lexical graph density 與預估 route cost 決定各 route 的 K，並允許在容易文件跳過 PLM。不可用必須先執行昂貴 route 才能得到的 route agreement 作事前決策；routing policy 只能在 validation 設計與凍結。

5. Selector competition  
   在完全相同 candidate、feature、budget 下比較 greedy、MMR、GRASP、NSGA-II；小實例另以 exhaustive/ILP 解驗證 optimality gap。2026-08-06 matched pilot 與五 seed extension 已觸發去留規則：MMR 為 main、Greedy 為 reference、NSGA-II 降為比較組；GRASP 是 optional comparator，不再阻塞主線。

   Phase 1e 已完成工程前提：Greedy、GRASP、MMR、NSGA-II 共用 constraints 與 final evaluator；Greedy／NSGA-II 另嚴格共用同一 scalar objective，MMR 的 sequential rule 則明確分開標示。200-row paired selector comparison 與 NSGA 五 seed stability 已完成，證據不支持 NSGA-II；exact small-instance gap 只在仍需解釋 stochastic comparator 時補做，不再是 main architecture blocker。

   新 shared objective 的 3-row correctness smoke 也揭露：若 multi-sentence mean-salience 只有 max budget、沒有 lower length bound，deterministic greedy 三筆皆退化為單句（41–88 words）。逐列 effective minimum 只能在完整來源於相同 upper constraints 下本身不可達時調降，且 requested/effective/source capacity/reason 全部保存。全文可達但 candidate pool 不可達時必須報錯，不能以第二次 clamp 掩蓋候選設計缺陷。2026-08-08 的 frozen A1 最終依兩資料集各自的一次 dev-test 與預註冊 paired-bootstrap/Holm 規則，將 Multi-News 凍結為 200–250 words（cross-method macro 0.310382），GovReport 凍結為 500–650（0.393415）；兩者相對三案的 adjusted p 均為 0.000600。Multi-News Greedy 仍有 3/1,686 floor shortfalls，必須原樣報告。GovReport 的 no-floor Greedy 只輸出約 181–202 words；floor-bound Greedy 雖到 637.1 words，macro 0.375048 仍低於 Lead 0.394815 與 Random 0.410382，證明長度修正不能替代 salience 改進。這個結果再次說明「修正公式」不等於「效果自然會變好」。

6. Pareto output policy（comparator only）
   不再把 Pareto front 最後任意 scalarize。若最終稿仍報 NSGA-II comparator，須預先固定 knee point、reference point 或使用者偏好，並報 front 的穩定性；不得再將它升格為主方法。

   現行程式的 `weighted_sum_on_shared_objectives` 僅是 provisional engineering policy；必須在 validation 改為並凍結預先指定的 policy，否則不可作 test 或論文主結果。

7. Regime-aware objectives  
   單句任務不得保留無意義的 redundancy objective；多文件加入 document coverage，長文件加入 section/topic coverage。objective 啟用條件是方法定義的一部分，不是 dataset-specific 偷調權重。

這個架構要驗證的潛在優勢不是絕對 ROUGE SOTA，而是：在不微調摘要模型的前提下，相對同 regime 的強方法提供較佳或相當品質，同時具有較低／可控成本、較長輸入支援、跨文件去重與可審計的選句 provenance。任何一項都必須由 validation/test、消融或成本量測支持後才能成為論文 claim。

### 3.7 先做可行性 pilot，再決定是否投入完整重構

不得直接跑 test。先在兩個 primary validation splits 做小型 pilot：

- 正確 evaluator、Lead、LexRank/TextRank、PACSUM、sentence-encoder+MMR。
- independent routes 與 candidate recall@K。
- greedy/MMR 對 NSGA-II 的同條件比較。
- No-Graph、No-PLM、No-Router ablation。
- end-to-end latency、RAM/VRAM。

Pilot 通過條件：

- full method 在至少一個主要資料集明顯優於強 no-task-training baseline，另一個至少不劣或形成清楚的 cost Pareto 優勢。
- graph 與 semantic route 至少有一個產生非零且可重現的 unique oracle-candidate recall；否則刪除無效 route。
- adaptive router 相對 always-on full method 降低至少一項完整成本，且 quality loss 在預先定義的 non-inferiority margin 內。
- [已觸發] NSGA-II 相對 deterministic selector 沒有穩定增益，已從標題與主貢獻
  移除 meta-heuristic；後續只作 comparator。

若 pilot 全數失敗，停止以「新摘要方法」投稿。可改寫為嚴謹的 negative-result／empirical study，但必須有具普遍性的分析發現，不能只是報告本系統失敗。

## 4. 程式重構藍圖

### 4.1 Legacy freeze

- 目前 commit 與 runs 標記為 legacy_ict_express。
- 既有 metrics 不可被新稿腳本自動讀取。
- 不刪除舊結果，但在 README 明確標示 invalid for publication。

### 4.2 建議模組邊界

- data：下載、split、schema validation、sentence segmentation、fingerprint。
- candidates：lead、statistical、graph、PLM、NSGA Pareto-frequency 等 route。
- fusion：score calibration、provenance union、route budget allocation。
- selection：greedy、GRASP、NSGA-II，以及 Pareto solution policy。
- evaluation：ROUGE、BERTScore、length、redundancy、human-eval export、paired statistics。
- experiments：validated configs、seed matrix、manifests、tables、plots。
- artifact：environment lock、one-command reproduction、expected outputs。

frontend、backend demo 與研究 benchmark 應分離，避免產品 fallback 或 API convenience 影響論文結果。

### 4.3 強制 schema

Document record：

- dataset、split、document_id。
- source_documents 與 sentence boundaries。
- references 為 list，不是單一任意串接字串。
- preprocessing metadata。

實作狀態（2026-07-28）：已建立 canonical schema 與 pinned Multi-News preprocessor，保存 source order、raw document/sentence hash、字元 span 與 cleaned-to-original mapping。完整 validation 已由固定 revision 生成：5,622 raw rows 中 1 列空來源被排除，得到 5,621 canonical rows；其中 72 列共含 1,042 個 U+FFFD。資料政策已在任何新 validation quality result 產生前凍結為 `multinews-validation-v1`：主分析保留全部 5,621 個 structurally valid official rows、禁止文字修復；paired clean sensitivity 僅排除 tracked manifest 中固定的 72 列，得到 5,549 rows。兩套 artifact 的 row count、revision、canonical fingerprint、file SHA、U+FFFD count，以及 replacement/exclusion manifests SHA 均由 runner 在建立 run directory 前強制核對並寫入 `dataset_preflight.json`。舊 flat JSONL 與任意 `--allow_replacement_character` 輸出仍不可作正式結果；train/test 與其他資料集仍待生成。

同日已將 provenance 接入 production selector：candidate artifact 保存各 route 的 raw score、完整 rank、percentile、top-K proposals、route agreement、normalized RRF fusion、reservation/inclusion reason、model revision、截斷統計與 deterministic cost facts；MVP selector 實際使用 normalized RRF salience，prediction 亦保存每個 selected sentence 的 selection evidence。正式 semantic route 現已在完整輸入上 batch encode 後排名，graph route 預設改為有界 sparse TF-IDF kNN。`route_top_k`、`min_per_route` 與 total cap 分開；route-exclusive evidence/guards 優先保留，RRF 不得從 proposal union/guards 外補句。短文件不再因句數少於 configured reservation 而中止：設定層仍驗證 `min_per_route <= route_top_k`，逐列則保存 requested/effective reservation 與 shortfall。這些是 correctness contract 已完成，不代表 route utility 已被實驗證實；正式 cost pilot、unique recall 與 quality ablation 仍是 freeze 前必要條件。

2026-07-26 correctness smoke：pinned `all-MiniLM-L6-v2@c9745ed...` 已在 CPU 直接 encode 並正確回報刻意截斷；其後以 canonical Multi-News validation 3-row debug subset 端到端處理 399 句。修正 membership-only 缺陷後，三列 final pool 為 55／60／60（第一列 proposal union 僅 55，誠實 underfill 5），union/guard 外候選為 0，selector salience 與 normalized RRF mismatch 為 0，且三列選句都相對舊 smoke 改變。此 run 只證明 wiring/artifact contract，不是 validation quality pilot；3-row ROUGE 不得寫入論文結果。

Phase 1e shared-objective smoke 先以 `min_words=0` 重跑，三列皆退化為單句（51／88／41 words），證明 mean-salience 不能只搭配 max budget。改用暫定 200–250 word validation band 後，三列長度落在 246／238／248 words。後續 strict audit 又發現 facility coverage 曾只覆蓋 candidate pool；拆成 `full source × candidates` coverage 與 `candidates × candidates` redundancy matrix 後，coverage universe 正確為 80／227／92 句，最終選取 5／4／7 句、246／240／248 words，全部 feasibility 為 true，candidate union/guard 越界與 RRF mismatch 均為 0。這只驗證 contract 與長度退化已受控；200 仍是待 validation freeze 的 provisional requested 下限，三列品質不能當效果證據。

⚠️ **注意兩個「72 列」是不同批，不可混談**：U+FFFD 有 72 列（frozen policy 的 replacement manifest），全文不足 `min_words=200` 也有 72 列，但兩者**交集為 0、聯集為 144 列（佔 5,621 的 2.56%）**。`clean_sensitivity` 排除的是前者，後者在兩種分析裡都保留並套用 `effective_min_words` 調降。

完整 canonical validation 的 length-feasibility audit 顯示：72/5,621 列（1.28%）全文不足 requested `min_words=200`；在句子不可切割與 `max_words=250` 下做 exact attainable-capacity 計算後仍恰為同 72 列，沒有額外 fragmentation case，且這 72 列最多只有 17 句，不受目前 candidate cap 40/60 截斷。正式 contract 因此保留這些官方 rows，僅將其 effective minimum 調為 source capacity。另有 100 句（44 列）單句超過 250 words，包含 `validation_5156:d001:s000002` 的 2,638-word 極端值；它們保留於 canonical data 與 coverage universe，但在 route top-K 前標為 selection-ineligible，不能浪費候選 quota，也不能未經 policy freeze 擅自重切句。

Candidate record：

- document_id、original_sentence_index、text hash。
- route、rank、raw_score、calibrated_score。
- model/checkpoint、representation、route cost。

Prediction record：

- selected original indices、summary sentences、summary text。
- requested method、actual method。
- length statistics。
- effective config hash、seed、commit、data fingerprint。
- per-stage timing與任何 warning。

### 4.4 Config 與失敗策略

- 使用 Pydantic 或 dataclass schema 驗證 config。
- 未知 key、缺少 key、互斥設定、錯誤 unit 一律 fail。
- 不再以多層 get 加 default 隱藏缺失。
- 所有 dependency failure 都 fail-fast。
- config_used 必須由 effective runtime object 序列化，不是原 YAML 原樣複製。

### 4.5 測試最低門檻

初始稽核時，tests 主要只檢查「有輸出」與 index range，requirements 也沒有 pytest；以下缺口必須逐項以獨立證據關閉。

必補：

- TF-ISF、length、position 的 hand-calculated golden tests。
- TF-IDF/TF-ISF similarity parity tests。
- Graph diagonal、threshold、dangling node、mass conservation tests。
- candidate top-K rank preservation tests。
- Stage-2 provenance score fusion tests。
- sentence/word/token budget boundary tests。
- exact-search（若實作）／metric-specific greedy-reference tests。
- multi-reference evaluation tests。
- rougeL 與 rougeLsum golden tests。
- NSGA-II parameter propagation、seed determinism、no-fallback tests。
- full toy pipeline snapshot test。
- dataset schema/fingerprint tests。

2026-07-27 狀態：TF-ISF v1/v2、length、position v1/v2 與內部 ROUGE protocol 的 hand-calculated golden 已完成；golden audit 同時促使 TF-ISF v2 從可能產生負 ubiquitous-term 權重的 legacy smoothing，修正為非負 `log((N+1)/(sf+1))`。`rougeL`／`rougeLsum`、對稱分句及 max-ROUGE-1 same-reference policy 已釘住；10-document full toy snapshot、短文件 reservation、similarity fail-loud 與 runtime validation/test-lock regression 亦已完成。TF-IDF/TF-ISF similarity parity 與 official `files2rouge` 數值 conformance（僅保留 SciTLDR 時）仍未完成。

品質 gate：

- unit tests 全過。
- research core branch coverage至少 90%。
- lint、type check、config validation 全過。
- 一個 10-document smoke experiment 能重現固定 hash。

## 5. 資料集與切分協定

### 5.1 CNN/DailyMail（條件式 optional sanity，非 v1 核心）

- 本節只有在 `ACTION_PLAN.md` §2.0 正式把 CNN/DailyMail 納入後才執行；目前不阻塞 Gate 2／3。
- 若納入，使用本地完整官方 split：train 287,113、validation 13,368、test 11,490。
- 本研究無 task-specific training 時，train 可不用；validation 仍只能用來選 K、權重與 route policy。
- test 只在 configuration freeze 後執行。
- 主設定採標準三句輸出，並回報平均 words 與 model tokens。
- Lead-3 必須是主表第一個 sanity baseline。
- 不得隱藏 first-25-sentence truncation；若研究長文件截斷，另做 sensitivity。

### 5.2 SciTLDR-AIC（v1 排除；以下是重新納入時的先決條件）

目前不排程、不跑新結果、不放比較表。以下條目不是現行待辦；只有在 test 前修改已凍結的實驗矩陣、說明重新納入理由後才啟用。

- 先依官方 paper、dataset card 與 evaluation code重建 split。
- 不再把多個 target 串成一個 reference。
- 一句摘要任務使用 exact same one-source-sentence constraint，不能用多句但相同 token budget 假裝同設定。
- 主協定以每篇對多個 gold TLDR 的 max ROUGE 聚合；mean aggregation 只作補充分析。
- 報 R1、R2、RL，以及 exact single-sentence extractive oracle、Lead-1、PACSUM、official labels baseline。
- 若保留 SciTLDR stress test，先重現官方 AIC oracle R1 52.4 與至少一個 released baseline，作為 evaluator conformance test；若不保留則不產生 SciTLDR 比較表。
- 需明確討論 extractive output 對 abstractive TLDR reference 的 mismatch。

### 5.3 Multi-News

- validation 做所有 K、weight、threshold、centrality 與 routing 選擇。
- test 一次性正式評估。
- 使用 245-word 或官方對應長度協定時，名稱必須是 words，不是 tokens。
- 本地重跑 Lead、TextRank、LexRank 與公平的 unsupervised baselines。
- 做 data-quality audit，包括零句、極長 cluster 與 unrelated documents。
- 增加文件長度分層結果，避免平均值被少數超長 cluster 主導。

## 6. Baseline 設計

Baseline 必須按 training regime 分組，不能把不同資源條件混成一張勝負表。

### 6.1 Sanity baselines

- Lead-3：CNN/DailyMail（僅在 optional sanity 被正式納入時）。
- Lead under same word budget：Multi-News。
- Lead-1：SciTLDR（v1 不跑；只有重新納入 official stress protocol 時）。
- Random with fixed seeds。
- Exact 或 greedy extractive oracle。

若 proposed method 連 Lead 都無法穩定超過，停止投稿並重新設計。

### 6.2 公平的 no-task-training baselines

- TextRank、LexRank。
- PacSum。
- Sentence-BERT centroid 加 MMR。
- GUSUM。
- 可重現時加入 Bi-GAE／DASG 或同 regime 的近期 graph baseline。
- Greedy、GRASP、NSGA-II 使用完全相同 feature 與 candidate input，隔離 optimizer effect。

### 6.3 Supervised extractive reference

- BERTSumExt。
- MatchSum。
- DiffuSum 或較近期可重現 supervised extractive system。

這一組用來顯示 absolute performance ceiling，不得宣稱在資源條件不同時公平勝負。

### 6.4 LLM baselines

Reviewer 已要求 modern LLM comparison。至少加入：

- 一個 zero-shot extractive sentence selector：只允許回傳 source sentence IDs，確保 output regime 相同。
- 一個 generative zero-shot summarizer：獨立列組，因其不是 extractive。
- 若預算允許，一個固定版本 closed model 與一個 open-weight model。

要求：

- 固定 model version、prompt、temperature、max output、日期、重試規則。
- prompt 與完整輸出附 supplement。
- 報 API cost、latency 與失敗率。
- 不得把會隨時間變動的未版本化 API 結果當永久 baseline。

## 7. Evaluation protocol

### 7.1 自動指標

Primary：

- ROUGE-1 F1。
- ROUGE-2 F1。
- CNN/DailyMail 與 Multi-News 使用 ROUGE-Lsum F1，保留句界。
- SciTLDR 使用官方 files2rouge／ROUGE-L，並由最大 R1 選定同一 reference；單句輸出下不可把名稱或 aggregation 與多句資料集混用。

Secondary：

- BERTScore F1，固定 checkpoint 與版本。
- source coverage。
- intra-summary redundancy。
- summary length compliance。
- candidate recall against extractive oracle。

Feasibility／denominator policy（F-17）：

- 每個 input 必須恰有一列 prediction；lower-bound 不可行不得讓 atomic batch 缺列。
- primary metric 計分 all rows，並另報 feasible count／rate 與 reason-code 分布。
- `feasible-only` 只作單 run 診斷；跨方法 sensitivity 必須取共同 feasible ID
  intersection，保存每篇 score 後才作 paired bootstrap。
- post-F-17 artifact 少任何 gold ID、duplicate ID、混合 schema 或非 boolean
  `feasible` 都是 artifact failure；pre-F-17 缺列 artifact 只能在顯式未驗證假設下作 diagnostic。

對 extractive system 不需把 factuality 當主要賣點，因句子源自原文；但多文件拼接仍可能造成指涉與時間順序問題，可在 human evaluation 評 coherence。

### 7.2 統計

- stochastic optimizer 至少 5 seeds，建議 10 seeds。
- 報 mean ± standard deviation。
- system comparison 用 paired bootstrap，至少 10,000 resamples，報 95% CI。
- 多個 pairwise tests 使用 Holm correction。
- 同時報 absolute delta 與相對改善，不只 p-value。
- primary endpoint 與 model-selection rule 在 test 前固定。

### 7.3 Human evaluation

建議 50 至 100 篇，至少 3 位評分者，blind、randomized、系統名稱隱藏。

評估面向：

- Informativeness。
- Non-redundancy。
- Coherence。
- Overall preference。

要求：

- 先做 pilot 與 power/variance 檢查。
- 報 inter-annotator agreement，例如 Krippendorff alpha。
- 提供 annotation guideline、樣例、排除規則與倫理/報酬資訊。

## 8. 必跑實驗矩陣

### E0. Correctness smoke tests

- 10 至 50 篇。
- 驗證 route rank、union provenance、length、metric 與 manifest。
- 不用來選研究結論。

### E1. Main results

- 兩個 primary datasets（GovReport、原版 Multi-News）的完整 official test。
- 分組報 sanity、no-task-training、supervised reference、LLM。
- 所有本地 baseline 使用同一 evaluator。
- 所有方法的 primary table 使用完整 official split 的同一 denominator；另列
  infeasibility rate，不能各自刪除失敗文件後比較 ROUGE。
- Multi-News main 是 primary 結果；frozen clean sensitivity 只在共同 rows 報 paired 差異。
- CNN/DailyMail 只有在 Gate 3 後預先納入才追加 frozen official-test sanity；SciTLDR-AIC v1 不跑。

### E2. Optimizer isolation

- 固定 features、candidates、budget。
- 只替換 Greedy、GRASP、NSGA-II。
- 報 quality、runtime、seed variance、hypervolume。

### E3. Module ablation

- Full。
- No statistical。
- No graph。
- No PLM。
- No provenance score，只保留 membership。
- No adaptive routing，固定 K。
- No NSGA-II，使用 deterministic selector。

每個 ablation 必須只移除一個因素，不能同時改 union size、score 與 optimizer。

### E4. Sensitivity

- K 或 total candidate budget。
- Graph threshold。
- PageRank、degree、eigenvector、betweenness。
- Fusion weights。
- Population、generation、mutation per variable。
- Document truncation。
- Summary length。

不得用 test sensitivity 曲線選參數。

### E5. Quality-cost Pareto

- x 軸 end-to-end latency 或 energy。
- y 軸 primary quality metric。
- 點包含 pure statistical、graph、PLM、full、LLM baselines。
- 另報 memory。
- 只有真正 non-dominated 的 variant 可稱 Pareto-efficient。

### E6. Candidate analysis

- 每一路對 validated exact-oracle candidate／明確標示的 greedy-reference candidate 的 recall@K。
- route 間 overlap/Jaccard。
- union marginal recall。
- PLM/Graph 對 statistical route 的 unique contribution。
- 依 document length 與 domain 分層。

這組實驗直接回答「三路到底是否互補」。

### E7. Qualitative error analysis

- 成功案例、失敗案例各至少 3 個。
- 顯示原文句號、route provenance、各分數、最終選擇。
- 分析 lead bias、duplicate facts、跨文件矛盾、指涉斷裂與 domain mismatch。
- 案例選擇規則需預先定義，不能只挑漂亮樣本。

## 9. Go / No-Go 標準

在寫主文前先做以下決策。

### Go

至少同時符合：

- 在至少兩個 dataset 上，相對強 no-task-training baseline 有一致正向效果。
- primary metric 的 paired 95% CI 不跨 0，或雖 quality 相當但在完整成本上形成清楚 Pareto 優勢。
- PLM 與 Graph 若被列為核心，各自必須有可測的 marginal contribution。
- 結果能由乾淨環境一鍵重現。
- 方法與 code traceability audit 全數通過。

### No-Go / Redesign

任一成立即停止投稿：

- 只勝過 TextRank/LexRank，卻輸給 Lead-3、PacSum 或合理 sentence-encoder baseline。
- 在完全對齊的 SciTLDR-AIC no-task-training 協定下仍輸給本地重跑的 PACSUM，卻把 SciTLDR 當成跨領域有效性的主要證據。
- 無法重現 SciTLDR-AIC one-sentence oracle 約 52.4，或仍需串接 references／只報 R1 才能維持結論。
- Full 對 No-PLM 的差異仍接近 0，卻保留 PLM 核心主張。
- 所有效果只在 tuned test 或單一 seed 出現。
- quality gain 小於不確定區間，且 runtime 顯著更差。
- 需要未揭露 truncation、挑樣本或跨論文 metric 才能宣稱優勢。

## 10. 論文主張重寫規則

禁止：

- meta-heuristics outperform LM-based methods。
- converge to the global optimum。
- substantially improves coherence，除非有人評或 coherence metric 支持。
- captures long-range dependencies，若只有 TF-IDF graph。
- primary innovation，若 graph 只是 thresholded TextRank。
- semantic precision，若沒有定義與測量。
- 3×–170× speedup 概括 full pipeline。

可接受：

- under a fixed no-task-training protocol。
- provides a statistically supported quality-cost trade-off。
- graph centrality acts as a structural complementary signal。
- PLM contribution is conditional on domain/document length，若分層實驗支持。
- NSGA-II explores trade-offs among explicitly defined objectives；不保證 global optimum。

## 11. 建議的新論文骨架

1. Introduction  
   問題是 no-task-training 條件下的 quality-cost trade-off；清楚說明 ICACT extension 與真正新增內容。

2. Related Work  
   Extractive summarization、unsupervised/frozen encoders、graph methods、meta-heuristics、multi-document、LLM-based selection。加入比較表，但只放可核對的屬性。

3. Problem Definition  
   定義 document、candidate route、budget、objectives、training regime 與 cost。

4. Proposed Method  
   Provenance-aware candidates、adaptive routing、objective normalization、Pareto selection policy。每個 equation 對應 code。

5. Experimental Protocol  
   Dataset splits、data audit、baselines、metrics、statistics、hardware、effective hyperparameters。

6. Main Results  
   依 regime 分組，不混淆 cross-paper reference。

7. Analysis  
   Optimizer isolation、ablation、sensitivity、candidate recall、quality-cost Pareto、scaling。

8. Human and Qualitative Evaluation

9. Discussion and Limitations  
   Extractive-abstractive reference mismatch、lead bias、domain dependence、runtime、LLM baseline drift。

10. Conclusion  
    只重述被數據支持的發現。

11. Reproducibility / Code Availability

Supplement：

- 全部 configs。
- prompts。
- per-seed tables。
- 額外案例。
- response matrix。

## 12. Reviewer 意見對照

Reviewer #1：

- 三 paradigms 不清：Introduction 與 problem definition 重寫。
- NSGA-II 缺直覺說明：Method 加 non-dominated sorting、crowding distance、front selection。
- related work 太薄：更新並批判式比較；必須實質討論 reviewer 指名的 G-SEEK 與 G-SEEK-2。這兩篇是低資源 extract-then-abstract／heterogeneous-graph 系統，不一定是公平的主 baseline，但必須清楚解釋本研究在輸出形式、graph 結構、training regime 與 multi-document setting 上的差異。
- 無 code：release artifact。
- extractive/abstractive mismatch：oracle、semantic metric、limitations。
- 只有 ROUGE：BERTScore、human evaluation。
- 缺 LLM：加入 extractive-ID LLM 與 generative reference。
- 無案例：E7。

Reviewer #2：

- baseline 太弱：按 regime 重建 baseline matrix。
- reproducibility 缺失：manifest、effective config、data fingerprint、artifact。
- graph 分析不足：threshold、centrality、candidate contribution、qualitative。

Reviewer #3：

- grammar：最後才做專業英文校對。
- recent work：systematic literature update。

Reviewer #4：

- novelty overstated：改為具體、可驗證差異。
- strawman PLM：Sentence-BERT/PacSum/GUSUM，加 supervised references。
- oracle 矛盾：重新定義。
- efficiency 矛盾：完整 pipeline Pareto。
- PLM 幾乎無貢獻：corrected ablation 後決定保留或刪除。
- 超參數不完整：validated effective config。
- Stage 1/2 不一致：重構為 canonical specification。
- entity relationship signal 不存在：刪除，除非真的實作並評估。
- graph primary innovation 不成立：降級或做真正方法擴充。

## 13. ICACT 擴充與 IEEE Access 投稿合規

ICACT 已獲獎且屬 prior publication。IEEE Access 接受作者自己 conference paper 的擴充版，但目前規則要求：

- 新稿必須引用 ICACT paper。
- 主文與 cover letter 清楚說明關聯與新增內容。
- IEEE Access 官方頁面目前寫明 similarity 必須低於 35%。
- 不可只改寫文字；新增方法、完整實驗與分析必須是實質內容。

本地文字 n-gram 診斷顯示 ICT Express 稿與 ICACT 主文仍有明顯重疊；這不是 iThenticate 結果，不能當正式比例，但代表新稿不可直接沿用段落。

建議準備 conference-extension table：

- ICACT contribution。
- IEEE Access 新增方法。
- 新增 datasets/baselines。
- 新增 statistical/human analysis。
- 新增 reproducibility artifact。
- 新增 conclusions。

其他目前官方要求：

- IEEE Access 建議主文控制在 20 頁內，較長需先詢問 EIC。
- experiments、statistics、analysis 必須達 high technical standard 且結論受資料支持。
- AI-generated text 必須依 IEEE Access 現行規定在 acknowledgements 揭露，並指出使用的系統、章節與使用程度。
- source file 與 PDF 內容必須完全一致。
- 所有作者需 biography；投稿作者需公開且完整的 ORCID。

## 14. Reproducibility artifact

最低交付：

- LICENSE、CITATION.cff、Code Availability。
- 固定 Python 與套件版本的 lockfile。
- 可選 Docker/CodeOcean capsule。
- 一鍵下載與驗證資料腳本。
- 一鍵 smoke test。
- 一鍵重製每張表與圖。
- expected outputs 與容許誤差。
- hardware/software manifest。見 `docs/research/COMPUTE_ENVIRONMENT.md`（目前涵蓋 sumy word-only tokenizer 契約、dependency pin、baseline 成本與 timing 限制，仍非完整硬體清單）。
- seeds、configs、per-example predictions、per-example metrics。
- 不包含受授權限制的 dataset 或 model weights；提供合法下載方式與 checksum。

IEEE Access 的 reproducibility guidance 特別要求 artifact dependencies、installation、execution time、完整 workflow 與 expected results；本專案應以「Code Available」甚至「Code Reviewed」等級準備，而不是只放 source files。

## 15. 執行順序與里程碑

### Phase −1：研究決策與 legacy 凍結，1 至 2 天

- 接受 F-0 的正確範圍：legacy Multi-News 未穩定勝過同協定 Lead；CNN/DailyMail 尚無公平勝負。
- 決定路線 A、B 或 C；預設選 A。
- 選定 primary benchmarks：GovReport + 原版 Multi-News；frozen U+FFFD clean 作必跑 paired sensitivity，external retrieval-cleaned variants 不屬 v1。
- 固定 primary metrics 與 Go/No-Go。
- 以明確 commit `1b9fe6f` 標記 legacy，而不是標記含未提交修改的工作目錄。
- 寫 canonical method specification。

完成 gate：研究路線、benchmark、method specification 與 Go/No-Go 均有書面凍結版本。

### Phase 0：專案治理與可重現性整理，約 1 天

- 依 `REPO_CLEANUP.md` 盤點 configs、runs、scripts 與疑似 dead code。
- 未證實的 dead code 先移入 archive，不直接刪除；任何大量移動或刪除另行確認。
- 分離 runtime 與 development dependencies，補上 pytest 與 lock intent。
- 建立 `runs/README.md`，標記 legacy、test-tuned 與可引用狀態。

完成 gate：新實驗入口、legacy 區、依賴與 artifact 狀態可由第三人辨識。

### Phase 1：correctness refactor，1 至 2 週

- 重構 data、candidate、fusion、selection、evaluation。
- 完成 schema、manifest、no-fallback、effective config。
- 補核心 tests。

完成 gate：所有 hand-calculated 與 integration tests 通過。

### Phase 2：data/baseline validation，1 週

- 重建兩個 primary datasets；CNN/DailyMail 僅為 Gate 3 後 optional sanity，SciTLDR v1 不跑。
- 跑 Lead、TextRank、LexRank、PacSum、Sentence-BERT centroid。
- 驗證 official split；多句資料採明確保留句界的 ROUGE-Lsum；只有保留 SciTLDR 時，才用官方 files2rouge、單句與 max-R1-reference 協定重現官方 oracle。

完成 gate：baseline 在合理範圍，差異有可解釋來源。

### Phase 3：方法與 validation experiments，1 至 2 週

- route provenance、adaptive routing、NSGA policy。
- K、threshold、weights、population、generation sensitivity。
- 只使用 validation。

完成 gate：configuration freeze。

### Phase 4：locked test，約 1 週計算時間

- 兩個 frozen primary datasets、全 seeds；只追加已在 test 前預先納入的 optional sanity。
- paired statistics。
- runtime/memory。
- 生成 immutable artifacts。

完成 gate：Go/No-Go。

### Phase 5：分析與寫作，1 至 2 週

- ablation、candidate analysis、qualitative/human evaluation。
- 重寫圖表、方法、discussion。
- conference extension 與 reviewer response matrix。

### Phase 6：投稿前稽核，2 至 3 天

- equation-code-config-result traceability。
- 引文與 retraction 檢查。
- grammar、IEEE template、AI disclosure、similarity。
- 從乾淨環境重製一次。

## 16. 最終投稿檢查表

- [ ] 所有 P0 關閉。
- [ ] test 從未用於調參。
- [ ] **條件式**：若 Gate 3 後正式納入 CNN/DailyMail optional sanity，使用 11,490 筆 official test；未納入則從新稿結果表完全移除。
- [x] SciTLDR 已從 v1 實驗矩陣移除；若日後重新納入，須先重開 official split／files2rouge／單句／max-R1-reference conformance gate。
- [x] Multi-News validation data-quality 規則已在新結果前固定為 versioned main + paired clean sensitivity policy；train/test 仍須各自受版本化 policy／manifest 管理。
- [x] 新 canonical pipeline 的 Stage 1 真正輸出各 route top-K ranked candidates，並保存 provenance／reservation／guard。
- [x] 新 canonical pipeline 的 Stage 2 使用真實、可追溯的 PLM/graph/statistical route scores；route 效果仍待 validation ablation。
- [x] 新 canonical pipeline 的 NSGA effective parameters、seed、objective 與 Pareto artifact 可追溯；最終 output policy 仍待 validation freeze。
- [x] 新 canonical pipeline 的 candidate／feature／optimizer failure 無 silent fallback；legacy 路徑不得產生新稿結果。
- [ ] GovReport／Multi-News 的 ROUGE-Lsum、共用 Punkt sentence boundaries 與 published-protocol parity 全部驗證；SciTLDR 不屬 v1。
- [ ] 強 baseline 在同一 evaluator 下重跑。
- [ ] 5 至 10 seeds、paired 95% CI、multiple-comparison correction。
- [ ] full pipeline runtime、memory、hardware 完整。
- [ ] Full、No-PLM、No-Graph 的結論符合 CI。
- [ ] 論文沒有 global optimum、coherence、speedup 等 unsupported claim。
- [ ] ICACT 被引用，extension 與 similarity 合規。
- [ ] code/data artifact 可由第三人重現。
- [ ] AI-assisted text 依 IEEE 規定揭露。
- [ ] 主文、supplement、code、tables 的數字逐項一致。

## 17. 重要來源

IEEE Access：

- About / current journal statistics: https://ieeeaccess.ieee.org/about/
- Submission Guidelines: https://ieeeaccess.ieee.org/authors/submission-guidelines/
- Preparing Your Article: https://ieeeaccess.ieee.org/authors/preparing-your-article/
- Reproducibility: https://ieeeaccess.ieee.org/authors/reproducibility/
- Reviewer Guidelines: https://ieeeaccess.ieee.org/reviewers/reviewer-guidelines/

資料集與方法：

- CNN/DailyMail BERTSumExt: https://aclanthology.org/D19-1387/
- MatchSum: https://aclanthology.org/2020.acl-main.552/
- PacSum: https://aclanthology.org/P19-1628/
- GUSUM: https://aclanthology.org/2022.textgraphs-1.5/
- Bi-GAE: https://aclanthology.org/2023.findings-emnlp.328/
- DiffuSum: https://aclanthology.org/2023.findings-acl.828/
- SciTLDR paper: https://aclanthology.org/2020.findings-emnlp.428/
- SciTLDR official evaluator: https://github.com/allenai/scitldr/blob/master/scripts/cal-rouge.py
- Multi-News: https://aclanthology.org/P19-1102/
- Multi-News+: https://aclanthology.org/2024.emnlp-main.2/
- GovReport: https://aclanthology.org/2021.naacl-main.112/
- Multi-XScience: https://aclanthology.org/2020.emnlp-main.648/
- SummEval: https://aclanthology.org/2021.tacl-1.24/
- G-SEEK, ECAI 2023: https://doi.org/10.3233/FAIA230460
- G-SEEK-2, IEEE/ACM TASLP 2025: https://doi.org/10.1109/TASLP.2024.3490375

## 18. 最重要的原則

下一版不能以「把 reviewer 要求都補進去」為目標；真正目標是讓每個結論都能由正確 split、正確程式、正確 metric、正確統計與完整 artifact 支持。

若重算後結果變差，應如實改定位或停止投稿，而不是尋找另一個 metric、subset 或表述方式掩蓋。這是把專案提升到 IEEE Access 水準最重要的一步。
