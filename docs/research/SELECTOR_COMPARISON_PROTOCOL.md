# Selector comparison protocol（validation freeze draft）

更新日期：2026-08-06  
狀態：**實作與 validation pilot 用；正式 test 尚未解鎖**

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
