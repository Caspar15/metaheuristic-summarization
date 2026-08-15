# Selector comparison protocol（completed development evidence）

初始凍結：2026-08-06 ｜ 狀態覆核：2026-08-15
狀態：**pilot、full-dev D2 與 paired analysis 已完成；正式 test 尚未解鎖**

> 本文件前半保留 200-row pilot 的原始設計與當時判斷；§9 的 full-dev 結果已取代
> 「MMR 普遍成為 main selector」的外推。現行 frozen GovReport candidate 使用
> TF-IDF MMR λ=0.7；NSGA-II 是成本高且沒有品質優勢的 comparator。selector search
> 已結束，接下來只有 E3 中事前指定的 matched ablation，不能新增 selector grid。

## 1. 要回答的問題

本階段只更換 selection module，回答：在相同來源、候選池、表示、輸出
budget 與最終評分函數下，deterministic Greedy、SBERT-MMR heuristic、
NSGA-II 哪一個值得進入最終架構。

這不是「哪個完整摘要系統最好」的唯一實驗。完整論文還必須另跑
full-source strong baselines；否則把 60-sentence candidate pool 內的 MMR
稱為標準 SBERT+MMR baseline，會不公平地替 proposed candidate router 擋掉
它本來可看的來源句。

## 2. 兩個不可混稱的比較

### A. Candidate-matched selector swap（本階段主實驗）

三個方法固定下列輸入：

- 同一 canonical row 與 selection-eligible source sentences；
- 同一 lexical + semantic route proposal、reservation、coverage guard 與
  final candidate indices；
- 同一 pinned `sentence-transformers/all-MiniLM-L6-v2` revision；
- 同一逐句 L2-normalized SBERT embeddings；
- 同一 candidate salience、candidate×candidate cosine matrix、
  full-source×candidate coverage matrix；
- 同一 `min_words`／`max_words`／`max_sentences`／non-empty constraints；
- 同一 shared evaluator，並保存 candidate/input fingerprints。

只交換：

1. **Greedy**：逐步加入使 shared scalar utility 邊際值最大的句子。
2. **Candidate-matched SBERT-MMR**：逐步最大化
   `lambda * relevance(i) - (1-lambda) * max similarity(i, selected)`；
   main run 預先固定 `lambda=0.7`，同分取較小 candidate-relative index。
3. **NSGA-II**：對 salience、facility coverage、redundancy 搜尋 Pareto front，
   再以預先固定 policy 選一個解。

MMR 的逐步 rule 本身與 shared scalar utility 不完全相同。因此：

- Greedy vs NSGA-II 才是嚴格的「相同 objective、不同 search」比較；
- 三者整體是「相同 selector interface/input/evaluation、不同 selection
  algorithm」比較；
- 不得寫成三者在數學上最佳化完全相同的函數。

### B. Full-source strong baselines（Gate 2 必跑，不能被 A 取代）

- **SBERT centroid-only**：對所有 eligible source sentences 依
  sentence-to-centroid cosine 排名，在相同 upper budget 下 rank-and-fill。
- **SBERT+MMR**：對所有 eligible source sentences 使用相同 embedding、
  centroid relevance 與 MMR rule，不經 proposed candidate router。

這兩個方法要使用 baseline contract，artifact 必須標成 baseline；不得把
candidate-matched MMR 的結果拿來宣稱已勝過 full-source SBERT+MMR。

## 3. SBERT 表示契約

Checkpoint：`sentence-transformers/all-MiniLM-L6-v2`  
Revision：`c9745ed1d9f207416be6d2e6f8de32d1f16199bf`  
Pooling：attention-mask-aware mean pooling  
Sentence embedding：pooling 後逐句 L2 normalization  
Similarity：normalized embeddings 的 dot product（等價 cosine）  
Maximum model tokens：256  
Tie-break：較小的 original/candidate-relative index

現行舊程式雖使用同一 checkpoint 與 mean pooling，但先以**未正規化**句向量
計算 centroid，再個別正規化做 cosine。2026-08-06 四句 smoke 對照官方
SentenceTransformer pipeline 時，逐句 normalized embedding 最大絕對差約
`6.7e-8`，但 centroid score 最大差為 `0.00867`。因此新實驗必須改用上述
模型原生 Normalize contract；舊 semantic-route artifact 不可冒充本協定結果。

Sentence Transformers 官方文件說明該類模型產生 fixed-size embeddings，
並以 cosine similarity 比較；本地 pinned model 的 `modules.json` 明確包含
Transformer → mean Pooling → Normalize。SBERT 論文本身沒有定義本文的
centroid+MMR 摘要 baseline，因此稿件應分別引用 SBERT 與 MMR，不得寫成
「SBERT 論文提出 SBERT+MMR 摘要法」。

主要來源：

- [Sentence-BERT, EMNLP-IJCNLP 2019](https://aclanthology.org/D19-1410/)
- [Sentence Transformers: semantic textual similarity](https://sbert.net/docs/sentence_transformer/usage/semantic_textual_similarity.html)
- [Carbonell and Goldstein, MMR, 1998](https://kilthub.cmu.edu/articles/journal_contribution/The_Use_of_MMR_and_Diversity-Based_Reranking_in_Document_Reranking_and_Summarization/6610814)
- [RL-MMR for multi-document summarization, EMNLP 2020](https://aclanthology.org/2020.emnlp-main.136/)

## 4. 凍結的實驗順序

1. golden/unit tests：公式、tie、budget、empty/oversized、determinism。
2. canonical 3-row smoke：確認三者 candidate/input fingerprints 相同。
3. 看 ROUGE 前先產生固定 pilot manifest 與 SHA-256。已於 2026-08-06 凍結
   `configs/pilot_manifests/multinews_selector_pilot_v1.json`：從 5,621-row
   canonical validation 以 `sha256(salt\0row_id)` 最小的 200 個 ID 取樣，
   manifest SHA-256 為
   `b0562eb4fe33b59f460c47e7b2db0bf0a3d2778687695422027bf33ed1c31b2e`；
   它只作方向／成本 diagnostic，不取代 full validation。
4. Multi-News validation main：先跑 deterministic Greedy/MMR；NSGA-II 先做
   cost preflight，再決定可承受的 population/generations，但參數須在看比較
   ROUGE 前寫入 config。
5. NSGA-II 至少 5 個預先列出的 seeds；報 median、range、失敗率與時間，
   不得只挑最好 seed。
6. 對相同 row IDs 做 paired bootstrap（至少 10,000 resamples）、95% CI；
   多個主要比較使用 Holm correction。
7. Multi-News validation 結論形成後，再於 GovReport validation 重複；兩個
   primary 都完成前不解鎖 test。

## 5. NSGA-II 去留規則

NSGA-II 只有同時滿足下列條件，才保留在標題與主要貢獻：

- 相對 deterministic Greedy，在至少一個 primary benchmark 的預先指定主要
  ROUGE 指標有 corrected-significant、且至少 `+0.003` absolute 的改善；
- 其他主要 ROUGE 指標沒有 corrected-significant regression；
- 第二個 primary benchmark 至少 non-inferior；
- 多 seed 結果穩定，而非單一 lucky seed；
- Pareto front 提供實際會使用、可解釋且優於單一 weighted-sum search 的
  trade-off；
- 額外時間／記憶體成本有報告，且品質成本比可辯護。

若不滿足，NSGA-II 降為 ablation/comparator；最終方法改以 deterministic
selector 為主。這不等於「重寫另一篇無關論文」，而是 ICT Express 拒稿後以
實證移除無效複雜度。ICACT 已發表版本仍須在新稿中引用並明確說明擴充與差異。

## 6. Artifact 最低要求

每一 row 除既有 prediction contract 外，必須保存：

- selector method 與所有 selector parameters；
- model name、resolved revision、pooling、normalization、max length、device；
- candidate original indices 的 SHA-256；
- selector salience、similarity、coverage inputs 的 SHA-256；
- selection order（輸出仍按來源順序重建）；
- shared-objective final evaluation；
- NSGA-II 的 seed、population、generations、Pareto front 與 chosen-row policy；
- 每列保存 deterministic operation counts；另以獨立 run-level cost artifact 報
  cold model load、warm representation、candidate construction 與 selector 時間。
  不把 nondeterministic wall-clock 寫進需要 byte-identical 的 prediction rows。

沒有 matched fingerprints，就不能把差異歸因給 selector。

## 7. 長度解讀規則

三者共用 `min_words=200`、`max_words=250`，但停止規則不同：MMR 是
rank-and-fill；Greedy 在下限已滿足且沒有正的 shared-utility 邊際增益時停止；
NSGA-II 從可行 Pareto front 依固定 weighted-sum policy 選解。因此每個品質表
必須同報 mean／median selected words 與 sentences。若某方法的 ROUGE 優勢伴隨
明顯較長輸出，須再作 length-matched sensitivity，不能直接歸因於 selector 品質。

## 8. Frozen 200-row pilot 結果（diagnostic）

程式 commit：`0fc2be16d4c504e252c6cdcb0c1e30f5eb1a1a10`；三法皆
200/200 feasible，逐列 selector-input fingerprints 完全相同。完整機器可讀 evidence：
`docs/research/evidence/selector_comparison_pilot_v1_summary.json`（SHA-256
`14c3fe7221f9a2096008d2c4df72bbd8fa6f78b135a6aa2ee5a482283a0f20c3`）。

| selector | R-1 | R-2 | R-Lsum | words | sentences | total seconds |
|---|---:|---:|---:|---:|---:|---:|
| Greedy | 0.40612 | 0.11284 | 0.36854 | 242.84 | 12.96 | 90.0 |
| SBERT-MMR | **0.42100** | **0.12756** | **0.37624** | 245.91 | 9.91 | 74.6 |
| NSGA-II seed 2024 | 0.40525 | 0.11054 | 0.36987 | 238.33 | 14.45 | 411.7 |

相對 Greedy，MMR 的 paired R-1 差為 `+0.01488`（95% CI
`[+0.00697,+0.02272]`，Holm `p=0.0024`），R-2 為 `+0.01472`
（`[+0.00523,+0.02371]`，Holm `p=0.0100`）；R-Lsum 為 `+0.00770`
但 CI 輕微跨 0、Holm `p=0.2080`。NSGA-II 三指標皆未顯著優於 Greedy。

僅看第一輪 seed 2024 時，架構判斷是 **MMR 升為 provisional main selector，
Greedy 作 deterministic objective-search reference，NSGA-II 暫降 comparator**；
當時尚不得以單 seed 寫成定案。下方五 seed stability extension 已完成這個
pilot 層級的去留判斷。
此外，本實驗只證明 candidate pool 內 MMR 較好，尚未證明 proposed candidate
router 優於 full-source SBERT+MMR；後者仍是下一個必要 baseline。

Pilot 看過 seed 2024 後，為避免以單 seed 解讀 stochastic NSGA-II，另凍結
`configs/pilot_manifests/selector_nsga_seed_extension_v1.json`，要求完整報告
seeds `[7, 42, 2024, 2025, 3407]`、不得選最好 seed。因 seed 2024 已先被觀察，
這只能稱 stability extension，不能冒充正式 preregistered multi-seed study。

Evidence 的 `git.dirty=true` 來自工作區既有、未追蹤的 `oracle_canonical.py`；
pilot 前後 `git diff` 均無 tracked source 變更，selection code 綁定上述 commit。
該未追蹤檔不是 runner import dependency，但仍保留 dirty 標記而不竄改 provenance。

### 五 seed stability extension 結果

完整 evidence：`docs/research/evidence/selector_comparison_nsga5_stability.json`
（SHA-256 `9820826d17a4dfadfb80a3372b94bc8ff074edb3b74d8700e62bc5becfd42cad`）。

| metric | NSGA-II 5-seed mean | SD | range | Greedy | MMR |
|---|---:|---:|---:|---:|---:|
| R-1 | 0.40376 | 0.00203 | 0.40125–0.40586 | 0.40612 | **0.42100** |
| R-2 | 0.10970 | 0.00212 | 0.10670–0.11229 | 0.11284 | **0.12756** |
| R-Lsum | 0.36820 | 0.00203 | 0.36606–0.37047 | 0.36854 | **0.37624** |

五個 seed 的 mean pairwise selected-sentence Jaccard 為 `0.639`；只有
`19/200 = 9.5%` 文件在五 seed 得到完全相同集合，平均每篇有 `4.425/5`
種不同集合。沒有任何 seed 的 corpus R-1 或 R-2 超過 Greedy；15 個
seed×metric vs Greedy comparisons 經 Holm 校正後均不顯著優於 Greedy。

因此 200-row pilot 當時的 selector gate 決策是：**MMR 是 provisional main selector；Greedy 是相同
shared-objective 的 deterministic search reference；NSGA-II 是被否證的
stochastic comparator，不再構成方法名稱或主貢獻。** 這個結論只處理 selector
選擇；整個 proposed system 是否成立，仍取決於 MMR 對 full-source SBERT+MMR、
PacSum 與兩個 primary datasets 的結果。此決策已由第 9 節 full-dev 結果更正。

## 9. Frozen full-dev D2 結果（Multi-News；取代 pilot 外推）

`d2-selector-full-dev-v1` 在任何新 selector score 前凍結 14 candidates／dataset；
Multi-News 只讀 dev manifest 的 3,935 rows，study summary 明記
`dev_test_accessed=false`、`test_split_accessed=false`。固定 S02b routes、candidate
pool、RRF salience、200–250 words，只替換 selector 與 similarity representation。

| selector | similarity | macro | R-1 | R-2 | R-Lsum | selection seconds |
|---|---|---:|---:|---:|---:|---:|
| Greedy anchor | TF-IDF | **0.328077** | **0.440759** | 0.139803 | **0.403670** | reused |
| NSGA-II 64×80 | TF-IDF | 0.322615 | 0.433008 | 0.134434 | 0.400403 | 4690.7 |
| MMR λ=0.3 | TF-IDF | 0.321744 | 0.438392 | 0.138005 | 0.388837 | 149.1 |
| MMR λ=0.5 | TF-IDF | 0.321462 | 0.436943 | **0.139598** | 0.387846 | 145.0 |
| MMR λ=0.7 | SBERT | 0.316612 | 0.430702 | 0.136888 | 0.382246 | 146.2 |
| NSGA-II 64×80 | SBERT | 0.308827 | 0.417747 | 0.124633 | 0.384101 | 5519.0 |

表中只列主要候選；完整 14-candidate evidence 在
`runs_v2/d2_selector_full_dev_v1/multinews/dev/study_summary.json`。NSGA-II+TF-IDF
比 Greedy 低 `0.005462` macro，且約為最佳 TF-IDF-MMR selection time 的 `31.5×`；
兩個 NSGA 候選均未達「距最佳 deterministic ≤0.002 或勝任一 metric」的多 seed
門檻。故本節**明確取代**第 8 節把 200-row MMR 優勢外推為 main-selector 決策的做法：
Multi-News 暫採 Greedy anchor，MMR 與 NSGA-II 都不升格。GovReport 同規格其後完成：
TF-IDF-MMR λ=0.7 macro `0.446154`，對 Greedy macro `+0.028293`，95% CI
`[+0.024725,+0.031910]`；104-endpoint Holm `p=0.020798`、228-opportunity correction
`p=0.045595`，四個 endpoints 全通過。然而它仍低 full-source SBERT+MMR adversarial
baseline `0.006614`；Multi-News Greedy 也仍低 P08 `0.003663`。

沒有任何 shared deterministic selector 在兩資料集都距各自 winner ≤`0.001`。依事前
規則，下一輪採顯式 task-profile policy，而不是 dataset-name switch：

- `multi_document + multi_sentence` → Greedy-TFIDF；
- `single_document + multi_sentence` → TF-IDF-MMR λ=0.7。

`promotion_eligible=false`，所以不讀 dev-test。第 8 節保留為研究歷史，不刪除；
paired evidence 為 `runs_v2/d2_selector_full_dev_v1/analysis/paired_summary.json`。
