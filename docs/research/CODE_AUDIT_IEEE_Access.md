# 程式碼稽核報告 — 投稿 IEEE Access 前的實證檢查
**Code Audit for: Combining Meta-Heuristic Optimization, Graph Centrality, and PLM Semantics for High-Quality Extractive Summarization**

> 稽核日期：2026-07-26
> 稽核對象：repo 根目錄（commit `1b9fe6f`）
> 新 pipeline 狀態覆核：2026-08-02（master 已含 PR #10）
> 與 `paper_revision_plan_IEEE_Access.md` 的關係：**本文件是 evidence ledger，不取代研究主計畫**。
> 研究主計畫定義「論文該怎麼改」；本文件記錄「legacy 程式碼與 artifact 實際上做了什麼」。
> 本文件是 legacy snapshot 的 evidence ledger，不是目前 working tree 的驗收證書。
> **2026-07-26 更新**：原本只存在暫存目錄的稽核腳本，其中四支已版本化至
> `scripts/audit/`（F-0 Lead 比較、選句位置／重疊診斷、
> 資料集 headroom、PLM 計時分解），並由該處重跑確認數字一致 —— 見附錄 B 與
> `scripts/audit/README.md`。
> **但版本化 ≠ 可作論文結果**：這些仍跑在 test-tuned legacy artifact 上、使用內部 Lsum 協定、
> 部分為抽樣、且未做 paired significance test，標籤維持 diagnostic。
> 少數分析（如 370/500 換行雜訊統計）尚未版本化，仍須重做。
> 後續修正狀態以 §0.0、`ACTION_PLAN.md` 與目前測試結果為準；下方 legacy 敘述不會隨 working tree 改寫。
> **2026-08-11 v2 freeze-package 覆核**：作者端已核准 GovReport-centered 定位；
> GovReport 是唯一 primary，Multi-News 是 boundary evidence。F-72 登錄官方 evaluator
> mismatch risk；457 tests、freeze verifier 與 provenance audit 已通過，protected splits 仍鎖定。
> **2026-08-06 selector evidence**：200-row reference-blind matched pilot 與五 seed
> NSGA-II stability extension 已完成。MMR 對 Greedy 的 R-1/R-2 paired gain 為
> `+0.01488/+0.01472` 且 Holm-significant；NSGA-II 五 seed mean 均低於
> Greedy、selection Jaccard 僅 `0.639`。selector 因此採 MMR，NSGA-II 降為
> comparator；這仍未回答 full-source MMR／PacSum／兩 primary 的 system gate。
> **2026-08-08 development-split governance**：先前所有 validation pilot 都直接使用
> full validation，沒有可供「反覆搜尋」與「單次確認」分離的 manifest。F-20 已為
> Multi-News 與 GovReport 都已補上 reference-blind dev/dev-test 凍結與 runtime
> enforcement；GovReport 資料層詳見 F-22。兩者均未讀取 test split。
> **2026-08-08 D1 lexical evidence**：兩 primary 的 12-config lexical/objective dev
> family 已完成。Multi-News 全文候選仍輸 Lead；GovReport 全文候選 macro
> `0.415585`，dev point estimate 高於 Lead／Random。後續 strong baseline／greedy reference／
> paired finalists 已完成並顯示 S02b 對兩 primary adversarial winner 均顯著落後；見 F-62。
> F-30/F-31 另記 exact Greedy 效能修正與錯誤 runtime 外推。

---

## 0.0 ⚠️ 各項發現的現況（讀本文件前先看這張表）

> **本文件的 F-1 ~ F-13 描述的是 commit `1b9fe6f`（legacy）的行為，以現在式書寫。**
> 其中多項在 Phase 1 重構後**已於新 pipeline 修好**，但 legacy 程式路徑仍保持原狀以便重現舊 artifact。
> 沒有這張表，讀者會把已修好的缺陷當成待辦事項。
>
> 「新 pipeline」= `configs/phase1_mvp_multinews.yaml` 走的 canonical 路徑。
> 「legacy」= `configs/1_*.yaml` / `2_*.yaml` 與 `fast_fused.py`，**刻意不修**。

| # | 發現 | legacy | 新 pipeline | 修在哪 / 為何未修 |
|---|---|---|---|---|
| **F-0** | 系統未贏 Lead | 🔴 成立 | 🔴 **Gate 2 dev 失敗** | route matched evidence 有正訊號，但正式 paired finalists 顯示 S02b 對 Multi-News P08 與 GovReport SBERT+MMR 的 macro CI 均全負；selection-aware wins 0。不得進 dev-test，見 F-62 |
| F-1 | 論文 "oracle" 不是 oracle | 🔴 成立 | ✅ 已可正確計算 | `src/eval/oracle.py`；canonical 與三個 metric target 已修，詳見 F-21。舊稿 0.136 須撤回 |
| F-2 | ROUGE-L 應為 Lsum | 🔴 成立 | ✅ 已修 | `src/eval/rouge.py`；published-protocol parity 仍待驗證 |
| F-3 | Stage 2 沒有 PLM | 🔴 成立 | ✅ **已修** | 新增 semantic route + `selector.salience_source: rrf_fusion`。`fast_fused.py` 保持原狀 |
| F-4 | PLM 每篇重載模型 | 🔴 成立 | ✅ 程式已修 | `load_encoder()` 快取。**但正式計時數字仍須依鎖定 protocol 重測** |
| F-5 | 相似度矩陣就地竄改 | 🔴 成立 | ✅ 已修 + regression test | `src/features/graph.py` |
| F-6 | `pop_size`/`n_gen`/`seed` 未接線 | 🔴 成立 | ✅ 已修 | `optimizer_dispatch.py` |
| F-7 | salience 用總和 → 基數偏誤 | 🔴 成立 | 🟡 部分禁止（僅限有 `task_profile` 的 profiled multi_sentence；例外詳見 F-14） | `objectives/factory.py` 拒絕 profiled multi_sentence config 用 raw sum；legacy_unprofiled（無 `task_profile`）與 legacy 保留 sum |
| F-8 | SciTLDR 多重 reference 被串接 | 🔴 成立 | ✅ 已修 | `preprocess_scitldr.py` 改存 `references: list` |
| **F-9** | **repo 無任何 baseline 實作** | 🔴 成立 | ✅ **工程解除／品質失敗** | 兩 primary frozen-dev non-PLM 各 23/23、PLM 各 27/27、greedy reference 6/6 與 paired finalists 已完成；S02b 對兩 adversarial winners 均顯著落後，見 F-62 |
| F-10 | 圖模組 τ 套用不一致 | 🔴 成立 | ✅ 已修 | τ 已傳入 `feature_builder.py` 與 graph route |
| **F-11** | `centrality` 與 `novelty` 完全反相關 | 🔴 成立 | 🔴 **仍然成立** | **未修**。新 MVP 兩者權重皆 0 所以不觸發,但退化仍存在 |
| F-12 | 分句用純正則 | 🔴 成立 | 🟡 部分修 | Multi-News／GovReport canonical 已改 NLTK Punkt；**legacy `preprocess.py` 未動，CNN-DM 僅在 Gate 3 後納入時處理** |
| F-13 | 其他（requirements 重複、pytest 缺、靜默 fallback…） | 🔴 成立 | ✅ canonical 主路徑已修 | requirements 去重 + pytest、fallback 移除；Greedy／GRASP／NSGA-II 共用同一 evaluator。legacy `fast_*` 路徑仍只供舊 artifact 重現 |

### 目前真正還開著的（不要被上面的 ✅ 誤導）

1. 🟡 **v1 Gate 2／D3b 雙-primary gate 失敗；v2 evidence 待補** —— 後續 D3b GovReport
   已對 strongest baseline 顯著勝出，但 Multi-News 仍低且 R-2 顯著較差；作者端已核准
   GovReport-centered 縮窄，不得再搜尋或進 protected split
2. ✅ **F-9 baseline 工程矩陣已完整** —— non-PLM、PLM、greedy reference 與 paired finalists 均完成；完成矩陣不等於方法有效，反而提供 redesign 的否證基準
3. 🔴 **F-11 centrality/novelty 退化** —— 若日後啟用這兩個特徵會出問題
4. 🟡 **F-12 legacy 分句與條件式 CNN-DM 分句規則**
5. 🔴 **F-72 GovReport official evaluator parity**；🟡 **F-4 正式 cold/warm timing、memory、scaling**
6. 🟡 **F-14 的 legacy_unprofiled raw-sum 例外**與 **F-15 的 legacy unmatched ablation**；兩者不得被誤當成新 canonical pipeline 的 matched evidence

---

## 0. 執行摘要（先看這裡）

> ⚠️ 本節與以下所有 F 條目描述 **legacy** 行為。現況請對照上方 §0.0 狀態表。

我把四位審稿人的技術指控逐條拿去對程式碼驗證。結論分成三類：

| 審稿人的指控 | 稽核結果 |
|---|---|
| R4 #3：oracle 邏輯矛盾 | ✅ **0.136 的錯誤來源已找到**；0.514 是非官方三句 greedy diagnostic，不是真 exact oracle，不能拿來重現官方 52.4 |
| R1-2：ROUGE-L 異常低 | ✅ 已確認舊碼算的是單序列 ROUGE-L；改按目前多句內部 Lsum 協定後由 0.201 → **0.388**，published-protocol parity 仍待驗證 |
| R4 #5：PLM 貢獻幾乎為零 | ✅ **證實，但原因不是「PLM 沒用」，而是 Stage 2 根本沒有 PLM** |
| R4 #4：BERT/RoBERTa 執行時間差 | ⚠️ 程式確實每篇重載模型。**純推論比值 ≈1.0 為穩定結論**（1.04× / 1.02×）；載入佔比不穩定（78% / 93%），不可引用特定數字。腳本 `scripts/audit/plm_timing.py`，須依鎖定 protocol 重測 |
| R2/R4：超參數缺失 | ✅ **比審稿人想的更嚴重**：部分超參數寫在 config 裡但程式從未讀取 |
| R2/R4：baseline 太弱、選擇性報告 | 🔴 **比審稿人想的嚴重得多 —— 見下方 F-0，這是最致命的一條** |

### 最重要的四句話

1. 🔴 **最壞的消息（F-0）**：本地 Lead prefix 在**全部 5622 篇** Multi-News 測試集上與論文 Table 7 的當家配置做 ID 對齊比較。結果：**論文的系統在 ROUGE-2 與 ROUGE-Lsum 上較低**，ROUGE-1 只高 0.0021，而且平均多約 13 個 whitespace words。這足以否定舊稿「across every metric」的主張；但兩者都是 legacy/test-tuned diagnostic，不可升格為新稿結果。

2. **可修正的事實錯誤**：R4 指出的矛盾確實存在於稿件，但來源已定位：0.136 不是 oracle。這不是 reviewer 的假警報；是稿件命名錯誤。修正 evaluator 後，legacy ExpB 的 Multi-News 指標由 R-L 0.2019 變成 R-Lsum 0.3880。

3. **方法與實作不符**：論文 Section 3.4 描述的 Stage 2 融合機制（Eq. 7–8 用 PLM embedding 相似度、`w_plm` 加權 PLM 語意分數）**與程式碼行為不符**。程式碼裡那個叫 `w_bert` 的參數，加權的是 TF-IDF，不是 BERT（F-3）。

4. **必須誠實面對**：程式確實每篇重載 encoder，因此舊 3×–170× 加速宣稱無法成立。載入佔比在重跑時由 78% 變成 93%（**不穩定，不可引用**）；正式幅度必須在修正後依完整 pipeline protocol 重測（F-4）。

---

## 🔴 F-0. 系統在 Multi-News 上並沒有贏過 Lead baseline —— 最致命的發現

> 這一條是我在稽核尾聲自己補跑 baseline 才發現的，**四位審稿人都還沒抓到**（他們手上沒有你的程式碼與資料）。
> 但 IEEE Access 的審稿人只要自己跑一次 Lead 就會發現。**必須主動處理。**

### 論文的說法

Section 4.4.1 與 Table 7：
> "The proposed method **significantly outperforms all extractive baselines across every metric**."

| Table 7（論文原文） | ROUGE-1 | ROUGE-2 | ROUGE-L |
|---|---|---|---|
| Lead | 0.4124 | 0.1291 | 0.1884 |
| LexRank | 0.4124 | 0.1269 | 0.1884 |
| TextRank | 0.4151 | 0.1315 | 0.1901 |
| NSGA-II + BERT + Graph | **0.4352** | **0.1405** | **0.2019** |

**Table 7 的圖說自己寫明了問題所在**：
> "The results of Lead, LexRank, and TextRank are **adopted from [16]**, while the remaining results are obtained from our proposed framework."

也就是說 —— **baseline 數字是從別的論文抄來的，不是在你自己的 pipeline 上跑的**（與稽核當時的 F-9 一致：commit `1b9fe6f` 中確實沒有任何 baseline 實作；目前 master 已有 Lead，但不能回溯挽救舊表）。

### 實測結果 ✅

我自己實作 Lead，用**完全相同的預處理、相同的資料、相同的 ROUGE 設定**，在**全部 5622 篇**上做 ID 對齊比較。系統端用的是產生論文 Table 7 數字的那一次 run（`runs/tuning_experiments/ExpB_K20_Max_Coverage`，其 `metrics.csv` = 0.435226 / 0.140524 / 0.201948，與論文完全吻合）：

| 系統 | ROUGE-1 | ROUGE-2 | ROUGE-Lsum | 平均長度 |
|---|---|---|---|---|
| **論文當家配置（NSGA-II+BERT+Graph, K=20）** | **0.4352** | 0.1405 | 0.3880 | 241.3 |
| Lead，245-whitespace-word 上限（與 legacy 系統設定同名預算） | 0.4331 | **0.1453** | **0.3901** | 228.0 |
| Lead，逐篇對齊到系統輸出長度 | 0.4325 | **0.1449** | **0.3895** | 225.8 |
| *（論文抄來的 Lead 數字）* | *0.4124* | *0.1291* | *0.1884* | *?* |

- ROUGE-1：系統高 **+0.0021**，差距很小；尚未做 paired CI，不能稱顯著勝出
- ROUGE-2：系統**輸 −0.0048**
- ROUGE-Lsum：系統**輸 −0.0021**
- 系統平均多用了約 13 個 whitespace words

> ⛔ **ROUGE-Lsum 欄位已過期**（2026-07-30, PR #9）。evaluator 的分句器從手寫 regex
> `(?<=[.!?。！？])\s+` 換成 `src/data/sentence_split.py` 的共用 Punkt tokenizer
> ——舊 regex 在每個縮寫句點後都切一刀，是真 bug。實測（800 篇真實 validation、
> Lead-style extract、245-word budget）：**R-Lsum +0.0032；R-1 與 R-2 皆 +0.0000**。
>
> 因此上表中 `0.3880 / 0.3901 / 0.3895` 三個值與 `−0.0021` 的差距**都必須重跑
> `scripts.audit.lead_vs_system` 才能再引用**。
> **R-1 的 `+0.0021` 與 R-2 的 `−0.0048` 不受影響，仍然有效**——F-0 的結論
> （舊稿「across every metric」不成立）單靠 R-2 落後就已經站得住。

**論文引用的跨論文 Lead 不能代表本地同協定 Lead**：R-2 相差 0.016，R-L 的大差距主要是 ROUGE-L/Lsum 度量不同（見 F-2）。混用來源與 evaluator 是「系統看起來大勝 baseline」的來源。

### 為什麼會這樣（這不是你們的錯，但必須面對）

Multi-News 的 **lead bias 極強**是文獻上 well-known 的現象 —— 新聞把最重要的資訊放在開頭，而參考摘要平均 217 字、系統輸出 241 字，長度已接近「把開頭抄下來」。在這種設定下 Lead **不是** trivial baseline，而是很強的對手。

### CNN/DailyMail 目前不能判定勝負

論文在 CNN/DM 報告 R-1 = 0.351。文獻中 **Lead-3 在 CNN/DM 是 0.4042**（Liu & Lapata 2019）。
但本地系統結果是 13,368 筆 validation，文獻數字是 11,490 筆 test，evaluator 也未對齊；因此不能宣稱系統低 0.05。這只能當作必須在官方 test、相同 preprocessing、budget 與 evaluator 下重跑 Lead-3 的高風險訊號。

**→ 目前只證實 legacy Multi-News 在 R-2 與 R-Lsum 未贏 Lead；CNN/DM 尚無有效勝負。**

### 該怎麼辦（按推薦度排序）

**(A) 徹底改變論文定位 —— 我強烈推薦這條**

不要再主張「品質勝過 baseline」。改成：

> **在 zero-training、無標註資料、CPU 可行的限制下，本文探討 meta-heuristic 多目標最佳化能否以極低成本達到與強 baseline 相當的品質，並系統性分析三種訊號的互補性與 quality–cost trade-off。**

配套：
- 誠實報告 legacy Multi-News 與 Lead 的診斷，並在正式重跑後再討論各資料集的 lead bias
- 主打 **quality–latency Pareto 圖**（F-4 修正後）
- 將版本化、同協定的 exact／greedy reference gap 作診斷；現有 0.591 抽樣值不可直接引用
- 把 negative findings（PLM 在 zero-training 下貢獻有限、Lead 在新聞領域極強）寫成**貢獻**，而不是藏起來

> IEEE Access 明確接受「完整、誠實、可重現」的研究，不要求絕對 SOTA。這條路可行。

**(B) 想辦法真的贏過 Lead**
- 先修好 F-3（真的接上 Sentence-BERT）再重跑，看是否足以拉開差距
- 換一個 lead bias 較弱的資料集當主場（SciTLDR、arXiv、PubMed、BillSum、Reddit-TIFU）
- **風險**：不保證做得到，Multi-News/CNN-DM 的 lead bias 是結構性的
- **建議**：先花 1–2 週試 (B)；不論成敗，最終論文都以 (A) 的框架呈現

**(C) 絕對不要做的事**
- ❌ 繼續沿用抄來的 baseline 數字
- ❌ 只報告系統略高 0.0021 的 ROUGE-1，而略過較低的 R-2 / R-Lsum
- ❌ 在 CNN/DM 繼續省略同 split、同 evaluator 的 Lead-3

這三件事任何一件被抓到，都會比目前的拒稿嚴重得多 —— 那會變成**研究誠信問題**，不只是技術問題。

---

## Part 1 — 已驗證的程式碼缺陷

嚴重度：🔴 致命（影響論文主張正確性） / 🟠 重大（影響可重現性與結論） / 🟡 應修（品質問題）

---

### 🔴 F-1. 論文的 "oracle" 不是 oracle —— 完整化解 R4 #3

**論文主張**：Section 4.3，SciTLDR-AIC 的 dataset oracle ROUGE-1 = 0.136，並據此論證「系統拿到 0.234 是很強的」。
**審稿人反擊**：extractive oracle 是理論上界，系統不可能超過它 → 判定為 factual error。

**稽核結果 ✅ 已定位確切來源**

那個 0.136 來自 SciTLDR 資料集自帶的 `rouge_scores` 欄位。我直接重現：

```
資料集 rouge_scores 欄位對「所有文件的所有句子」取平均 = 0.13758
論文報告值                                              = 0.136
```

`rouge_scores` 是 SciTLDR 提供的**每一個原文句子單獨對 target 的 ROUGE 分數**，用途是產生 extractive 訓練標籤（`source_labels`）。
**對它取平均 = 「隨機抽一句話的期望分數」，這在數學上是下界性質的統計量，不是上界。**

以下數字定位了 0.136 的來源，但除官方 one-sentence oracle 之外都不是 SciTLDR 投稿級 protocol：

| 指標 | ROUGE-1 | ROUGE-2 | ROUGE-Lsum |
|---|---|---|---|
| 論文誤稱的 "oracle"（`rouge_scores` 全句平均） | 0.1376 | — | — |
| 每篇取 legacy `rouge_scores` 最大值（非官方 diagnostic） | 0.4311 | — | — |
| **Legacy greedy reference（3 句、串接 references，非官方）** | **0.5136** | **0.1931** | **0.4146** |
| 系統實際表現 | 0.2338–0.2391 | — | — |

這足以證明 0.136 不是 oracle，但不能用 0.514 取代它。正式 evaluator 必須依官方程式：單句輸出、files2rouge，先以最大 ROUGE-1 選定同一個 reference，再從該 reference 報 R1/R2/RL，並重現官方 oracle R1 52.4。

> 📌 這一條證明舊稿的 0.136 必須撤回。IEEE Access 是新投稿，不應假設有對 ICT Express reviewer 的正式 response letter；可在內部 response matrix 與新稿 evaluation protocol 中說清楚，並只報符合該資料集官方／預註冊協定的 oracle reference。

**修正動作**
- 刪除 Section 4.3 現有的 oracle 論證
- 先重現官方 SciTLDR one-sentence oracle；其他資料集若另算 greedy reference，必須明確寫出貪婪法、句數/word 上限與 ROUGE 設定
- 三個資料集都補 oracle reference；exact 才能稱 upper bound，greedy 必須明確標成 greedy reference
- 順帶說明「SciTLDR 的 abstractive reference 與原文重疊度偏低」是資料集特性，寫進 Limitations

**相關檔案**：`src/data/preprocess_scitldr.py:19`

---

### 🔴 F-2. ROUGE-L 用錯了 —— 應為 ROUGE-Lsum，修正後分數大幅上升

**程式碼**：`src/eval/rouge.py:10`
```python
scorer = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)
```
**程式碼**：`src/pipeline/select_sentences.py:129`
```python
summary = " ".join([sentences[i] for i in selected])   # 空白串接，無換行
```

`rouge_score` 套件的 `rougeL` 是把整段摘要當成一個 token 序列做 LCS；`rougeLsum` 是多句摘要可採的 summary-level variant。它與 Perl ROUGE／pyrouge published numbers 不保證完全相同，因此只能作新的統一本地 protocol，所有主 baseline 必須同 evaluator 重跑。

**稽核結果 ✅ 在你自己的 5622 篇 Multi-News 完整測試集上實測**

| 設定 | ROUGE-1 | ROUGE-2 | ROUGE-L |
|---|---|---|---|
| 論文目前報告（`rougeL`，空白串接） | 0.4337 | 0.1391 | **0.2014** |
| 目前多句內部 `rougeLsum`（正規化空白後分句） | 0.4337 | 0.1391 | **0.3857** |

Full benchmark artifact 的 ROUGE-L 從 0.2014 變成 ROUGE-Lsum 0.3857；ExpB artifact 則是 0.2019 → 0.3880。這是 metric definition 改變，不是模型品質「提升 91%」。

⛔ **這兩個 Lsum 值是在舊的 regex 分句器下量的**（2026-07-30, PR #9 已換成共用 Punkt
tokenizer，實測位移 +0.0032）。`full_benchmark_result` 的 `predictions.jsonl` 已不存在，
**該數字永遠無法在新協定下重算**，只能當歷史紀錄，不得寫進論文。

這一改只讓 metric 名稱與多句內部協定一致，不能推論模型變得有競爭力；同協定 Lead 的 R2 與 R-Lsum 仍較高，且 published-protocol parity 尚未完成。

> ⚠️ **分句必須兩邊一致**。探索性抽樣曾發現 370/500 predictions 含來源雜訊換行、references 沒有，且不對稱換行可能改變約 0.023；原分析腳本未版本化，須重做。`src/eval/rouge.py::_as_lsum` 目前採「先正規化空白、再依標點分句」，仍須 golden tests 與 published evaluator parity，不能先稱唯一正確做法。

> ⚠️ 注意：ROUGE-1 / ROUGE-2 **不受影響**（它們與句子切分無關）。所以這是純粹的加分，不會動搖其他結論。

**修正動作**
- `src/eval/rouge.py` 改用 `rougeLsum`，並在 pred / ref 兩邊都以 `\n` 分句
- **三個資料集全部重算**，論文所有表格的 R-L 欄位更新
- 在 Experimental Setup 明確寫出：`rouge-score` (Google 官方實作)、版本號、`use_stemmer=True`、ROUGE-Lsum 的分句方式

---

### 🔴 F-3. Stage 2 融合完全沒有用到 PLM —— 這才是 ablation 失效的真正原因

**這是本次稽核最嚴重的發現。**

**論文主張**（Section 3.4，R4 detailed 也引述）：Stage 2 以 `w_base` / `w_plm` 融合統計分數與 **PLM 語意分數**；Eq. 7–8 的目標函數建立在 **PLM embedding 相似度**上。

**程式碼實際行為**：

`configs/2_Fusion_Final.yaml` 指定 `optimizer.method: fast_nsga2`，走到 `src/models/extractive/fast_fused.py:113-116`：

```python
def fast_nsga2_select(sentences, base_scores, ..., w_base=0.5, w_sem=0.5, ...):
    sem_scores, sim = _tfidf_scores_and_sim(sentences)   # ← TF-IDF，不是 PLM
    base_n = _minmax_norm(list(base_scores))
    sem_n  = _minmax_norm(list(sem_scores))
    importance = [w_base * base_n[i] + w_sem * sem_n[i] for i in ...]
```

而 `src/pipeline/optimizer_dispatch.py:139-140` 把 config 裡的 `fusion.w_bert` 餵給 `w_sem`：

```python
w_sem = float(fcfg.get("w_bert", 0.5))
```

**也就是說：那個叫 `w_bert` 的參數，加權的是 TF-IDF centroid 相似度，與 BERT 無關。**
**Stage 2 的相似度矩陣 `sim` 也來自 `tfidf_scores_and_sim`，是 TF-IDF，不是 PLM embedding。**

你自己的 `README.md` 其實已經寫明了這件事：
> 「Stage2 僅使用 fast 系列（**TF‑IDF 語義** + MMR/GRASP/NSGA2）」

**PLM 在整條 pipeline 中唯一的作用**：Stage 1b 用 BERT 挑出 top-K 句子的**索引**，併入 Stage 2 的候選聯集（`scripts/utils_fusion.py`）。它的 embedding 分數在 Stage 2 被完全丟棄。

**這解釋了 R4 #5**：Table 8「移除 BERT 只掉 0.0005 ROUGE-1、ROUGE-L 甚至上升」—— 因為移除 BERT 只是讓候選池少了幾個索引，PLM 的語意訊號從頭到尾就沒有進入評分函數。**這不是「PLM 沒有幫助」的科學發現，而是「PLM 沒有被接上」的實作缺陷。**

> 🚨 **誠信風險**：論文 Eq. 7–8 描述的方法與程式碼不符。IEEE Access 要求公開程式碼（P2-3），審稿人比對後會發現。**這一項必須修，不能靠改寫文字繞過。**

**修正動作**

- **先實作真的 PLM route，並把是否保留交由 validation ablation 決定**
  - 讓 Stage 2 的 `sem_scores` 與 `sim` 真的用 sentence encoder 計算
  - **優先 pilot Sentence-BERT / SimCSE**（`all-MiniLM-L6-v2` 已在本機 HF cache），與 raw BERT mean-pooling 公平 ablate；最終保留哪一種由 validation 決定 —— 對應研究主計畫 P0-4、P1-1
  - 重跑 ablation。**這次的數字才是真的可以拿來論證的**
- **若正確實作後仍無 marginal contribution，再誠實降級或刪除**
  - 若跑完 (A) 後 PLM 仍無顯著貢獻，才改寫標題與 Abstract，把 PLM 降級為 optional module，並把它寫成 negative finding
  - **但順序不能顛倒** —— 先修好再下結論

---

### 🟠 F-4. PLM 每篇重複載入；載入佔比不穩定，只有「推論比值 ≈1.0」可採信

**論文主張**：Abstract 宣稱 3×–170× speedup；Table 2 報告 BERT 950.6 ms/article、RoBERTa 634.7 ms/article。
**審稿人質疑**（R4 #4）：兩者 encoder 架構幾乎相同，1.5× 的差距說不通。

**根本原因**：`src/models/extractive/encoder_rank.py:29-42`

```python
def _sentence_embeddings(sentences, model_name=..., ...):
    tokenizer = AutoTokenizer.from_pretrained(model_name, ...)   # ← 每次呼叫都重新載入
    model = AutoModel.from_pretrained(model_name, ...)           # ← 每次呼叫都重新載入
```

`_sentence_embeddings` 被 `encoder_select` 呼叫，而 `encoder_select` 在 `summarize_one()` 中**逐篇文件**被呼叫。
**→ 每處理一篇文章，就從磁碟重建一次 tokenizer 和 model。**

**實測**（腳本 `scripts/audit/plm_timing.py`，40 句/篇。**兩次執行結果差異很大，見下方警告**）

第一次（CPU，20 threads）：

| 模型 | 載入 (ms) | 推論 (ms) | 合計 (ms) | 載入佔比 |
|---|---|---|---|---|
| bert-base-uncased | 1225.6 | 343.8 | 1569.4 | 78.1% |
| roberta-base | 262.2 | 331.7 | 593.9 | 44.1% |
| xlnet-base-cased | 763.1 | 478.5 | 1241.6 | 61.5% |

第二次（同機器，CPU，14 threads，磁碟快取狀態不同）：

| 模型 | 載入 (ms) | 推論 (ms) | 合計 (ms) | 載入佔比 |
|---|---|---|---|---|
| bert-base-uncased | 4090.0 | 323.6 | 4413.6 | 92.7% |
| roberta-base | 1027.4 | 318.0 | 1345.4 | 76.4% |
| xlnet-base-cased | 2418.9 | 409.5 | 2828.5 | 85.5% |

| 比較方式 | 第一次 | 第二次 |
|---|---|---|
| **只算推論**（BERT/RoBERTa） | **1.04×** | **1.02×** |
| 載入 + 推論（BERT/RoBERTa） | 2.64× | 3.28× |

> ⚠️ **載入時間在兩次執行間差了 3 倍以上（1225ms → 4090ms），載入佔比 78% → 93%。**
> **不可引用任何特定的載入佔比數字。**

**可確認的結論**（兩次都成立）：

1. **純推論的 BERT/RoBERTa 比值 ≈ 1.0**（1.04× 與 1.02×）—— 兩個架構等價的 encoder 本來就該如此。
   **這直接回答 R4 的疑問：舊稿的 1.5× 差距不可能來自 encoder 架構或 tokenizer。**
2. **載入時間遠大於單篇推論時間**，因此舊碼把重複建構模型納入 per-article 時間，
   舊的 3×–170× 加速宣稱不可沿用。

**不可確認的**：具體載入占比、修正後的加速倍數 —— 必須用鎖定的 runtime protocol
（固定硬體／thread／batch、排除 warm-up、≥5 次重複、報 median/mean/std/P95）重測。

**修正動作**
- 重構：模型只載入一次，在文件間重複使用（見 Part 3 的 R-2）
- **重新量測全部計時數字**
- 明確區分並分別報告：模型載入（一次性）/ 每篇推論 / 每篇選句
- 報告 mean ± std over ≥5 runs，排除 warm-up
- 補完整硬體規格：CPU 型號與核數、GPU 型號與 VRAM、batch size、max_length、fp16 與否、thread 數

> ⚠️ 修好後 PLM baseline 會變快，但幅度未驗證；不得預填 3–5 倍。
> **建議策略**：不要再使用舊「170× 加速」數字；依研究主計畫 P0-2，在修正後完整 pipeline 上畫 **quality–latency Pareto 圖**，同時呈現純 meta-heuristic 與完整 fusion 變體。

---

### 🟠 F-5. 相似度矩陣被就地竄改（silent corruption）

**程式碼**：`src/features/graph.py:36`
```python
if threshold > 0:
    similarity_matrix[similarity_matrix < threshold] = 0.0   # ← 就地修改呼叫端的陣列
```

**稽核結果 ✅ 實測確認**：呼叫 `compute_textrank_scores(sim, threshold=0.2)` 後，呼叫端持有的 `sim` 有 8/64 個元素被永久歸零。

**影響路徑**（`src/pipeline/select_sentences.py`）：
1. L46：算出 `sim`
2. L68：`build_candidate_union(..., sim_matrix=sim, threshold=0.2)` → **`sim` 被就地截斷**
3. L90：`sub_sim = sim[np.ix_(cand_idx, cand_idx)]` → 取用**已被汙染的**矩陣
4. L117：`sub_sim` 傳給 NSGA-II，用於 coverage 與 redundancy 目標函數

**後果**：只要 `candidates.sources` 含 `graph`/`textrank`，NSGA-II 的 coverage 與 redundancy 就是在一個「所有 <0.2 的相似度都被歸零」的矩陣上計算的 —— 這是非預期的副作用，論文從未描述，且會隨 config 不同而靜默改變實驗語意。

**修正**：`similarity_matrix = similarity_matrix.copy()` 後再做 thresholding（一行修正）。

---

### 🟠 F-6. config 裡的 NSGA-II 超參數從未被程式讀取

**稽核結果 ✅**：`pop_size` 與 `n_gen` 這兩個字串在整個 `src/` 中**只出現在 `nsga2.py` 的函式簽章與內部**，`optimizer_dispatch.py` 從未從 config 讀取或傳遞它們。

```
src/models/extractive/nsga2.py:109:    pop_size: int = 100,
src/models/extractive/nsga2.py:110:    n_gen: int = 100,
（其他地方：無）
```

**後果**：
- `configs/1_Base_NSGA2.yaml` 寫 `pop_size: 40, n_gen: 50` → **完全被忽略**，實際跑的是 100/100
- 論文若報告了 population size 40 / 50 generations，**那是錯的**
- `seed` 也沒有傳給 `nsga2_select()`（`minimize(seed=None)`）

**關於可重現性的好消息 ✅**：我實測過，`select_sentences.main()` 開頭的 `set_global_seed(cfg["seed"])` 會讓 pymoo 內部抽取的隨機種子變成確定的，因此**整份腳本重跑的結果是可重現的**（三次獨立執行結果完全相同）。但這是靠全域狀態的巧合，非常脆弱 —— 應該把 seed 顯式傳進去。

**修正**：把 `pop_size` / `n_gen` / `seed` 從 config 正確接線，並在論文中報告**實際使用的值**。

---

### 🟠 F-7. NSGA-II 的第一個目標是「總和」→ 產生基數偏誤（cardinality bias）

**程式碼**：`src/models/extractive/nsga2.py:78`
```python
imp = np.sum(self.importance[idx])      # 未正規化的總和
```

三個目標中：
- `-imp`（重要性總和）：**加入任何句子都必然變好**（importance 非負）
- `-cov`（coverage）：**加入任何句子都不會變差**（max 運算，單調遞增）
- `red`（冗餘度）：唯一會懲罰多選的項

**後果**：三個目標裡有兩個對集合大小單調遞增，搜尋被系統性推向**約束邊界的最大可行子集**。NSGA-II 實際上退化成「在長度上限內盡量塞滿，再由冗餘度做微調」，這削弱了 multi-objective 論述的說服力 —— 而 multi-objective formulation 正是論文宣稱的核心貢獻之一。

**修正建議**
- 改用**平均重要性**（`imp / |S|`），或明確加入 cardinality 作為第四個目標
- 補一張 **Pareto front 視覺化**與所選解的位置（對應研究主計畫的 Pareto output policy）
- 做敏感度分析：對比 sum vs. mean 兩種 formulation

---

### 🟠 F-8. SciTLDR 多重參考被串接，而非依官方規則選 reference

**程式碼**：`src/data/preprocess_scitldr.py:19`
```python
"highlights": " ".join(ex["target"]),   # 註解寫 "Join target sentences"
```

**問題**：SciTLDR-AIC 的 `target` 是**多個「替代版本」的 TLDR**（作者版 + 標註者版），不是同一篇摘要的多個句子。官方 `cal-rouge.py` 對各 reference 計分後，以最高 ROUGE-1 選定一個 reference，R1/R2/RL 都取自該同一 reference；不是串接，也不是每個 metric 各自取最大值。

實例（第一篇）：
> "FearNet is a memory efficient neural-network, inspired by memory formation in the mammalian brain, that is capable of incremental class learning without catastrophic forgetting. **This paper presents a novel solution to an incremental classification problem based on a dual memory system.**"

這明顯是**兩份獨立的 TLDR**被黏在一起。

**稽核結果 ✅**：串接後的 reference 平均 66.4 words，而單一 TLDR 約 20–25 words → reference 被膨脹了約 3 倍，壓低了 F-measure 的 precision 項。

**額外發現**：618 篇中有 **139 篇**的 `len(sentences) != len(rouge_scores)`（例：84 句 vs 83 個分數）。若有任何程式碼把這兩者按索引對齊，就會產生錯位。

**修正**：保留 reference list，使用官方 files2rouge 與 max-R1-reference aggregation，並在論文中明確說明協定。

---

### 🟠 F-9. 稽核當時 repo 完全沒有 baseline；目前矩陣仍不完整

**稽核結果 ✅**：以 `lead`, `LexRank`, `PacSum`, `BERTScore`, `bert_score` 等關鍵字全域搜尋 `src/`、`scripts/`、`tests/` —— **零命中**（唯一命中是 `position.py` 裡的變數名）。

**目前狀態（2026-08-09）**：Lead、Random、TextRank／LexRank 與 full-source SBERT
centroid／MMR 程式皆已接線，
所以「目前 repo 零 baseline」已不再成立。Lead／Random 共用 canonical data-policy
preflight 與 output upper-bound contract；TextRank／LexRank 包裝 pinned sumy。PR #15
已移除 offline `punkt_tab` regression，Linux CI 綠燈。Multi-News frozen-dev non-PLM
23/23 已完成，P08 是該 family 最強；Multi-News PLM 27/27 亦完成，PacSum-SBERT P03
仍低 P08 `0.000282`；GovReport non-PLM 已完成且 LexRank 勝出；PLM 27/27 亦完成，
full-source MMR λ=0.9 只高 LexRank `0.001148`。greedy reference 與完整 paired
matrix 仍未完成。因此 F-9 仍只能列「部分解除」，
Gate 2 不能標成完成。

**後果**：舊論文 Table 6 在 Multi-News 上報告的 Lead / TextRank / LexRank 數字，
**仍無法由本 repo 回溯重現，也不得沿用**。現在已有同一 preprocessing、budget 與
evaluator 下的新 replacement artifacts（Lead 與 final TextRank／LexRank），所以新版論文
應換成新數字與 provenance，而不是宣稱重現舊表。舊表若繼續出現，仍會落在 R4 點名的
「selective reporting」風險。

**修正**：所有主 baseline 都必須在本地同一 preprocessing、budget 與 evaluator 下重跑。可優先使用官方／可信實作並鎖定版本，不要求為了「自己寫」而重造演算法。清單見 Part 4。

---

### 🟡 F-10. 圖模組的 τ 套用不一致

- `src/pipeline/feature_builder.py:85`：`compute_textrank_scores(similarity_matrix)` —— **沒有傳 threshold**，用預設 0.0，即**完全不剪枝**
- `src/pipeline/candidate_builder.py:31`：`compute_textrank_scores(sim_matrix, threshold=threshold)` —— **有**剪枝

**後果**：`graph_params.threshold: 0.2` 只影響候選池挑選，不影響 graph 特徵分數本身。論文若把 τ 描述成「圖模組的邊剪枝閾值」，與實作不符。做研究主計畫要求的 τ 敏感度實驗前**必須先統一語義**，否則曲線沒有可解釋性。

---

### 🟡 F-11. `centrality` 與 `novelty` 是同一個特徵（完全反相關）

`src/features/semantic.py`：
- `centrality` = 相似度矩陣的**列平均**（且**含對角線自身相似度 1.0**，本身也是個 bug）
- `novelty` = `1 - (列和 - 1)/(n-1)`

兩者都只是「列和」的單調函數，一個遞增一個遞減。min-max 正規化後，數學上恰好滿足 `centrality_norm = 1 - novelty_norm`。

**後果**：同時給這兩個特徵獨立權重是退化的 —— `w_c·x + w_n·(1-x) = (w_c - w_n)·x + w_n`，實際自由度只有一個。若論文把它們列為兩個獨立特徵，是誤述。順帶：centrality 應排除對角線。

---

### 🟡 F-12. 分句採用純正規表示式，產生大量壞句子

`src/data/preprocess.py:11`：`(?<=[.!?。！？])\s*` —— 沒有處理縮寫（`U.S.`、`Dr.`、`Inc.`）、小數（`3.5`）、引號結尾。

**稽核結果 ✅**（Multi-News 前 500 篇，37,349 個「句子」）：
- **358 個**「句子」超過 80 words（明顯是壞切分），最長的達 **855 words**
- **10/500** 篇有新聞版面雜訊被黏成句子，例如：
  `"GOP Eyes Gains As Voters In 11 States Pick Governors Enlarge this image toggle caption Jim Cole/AP Jim Cole/AP Voters in 11 states will..."`
  （標題 + 圖說 + 內文被合併成單一「句子」）
- 2/500 篇有 U+FFFD 編碼損毀字元（輕微）

**後果**：一個 855-word 的「句子」會超過程式所稱的 245-token 預算；該預算實際是 whitespace-word count。同時 `length_scores` 偏好長句、v1 的 `sentence_tf_isf_scores` 分數也隨句長遞增 → **系統對這些壞句子有正向偏好**。

**修正**：改用 NLTK `punkt` 或 spaCy 分句；Multi-News 需正確處理 `|||||` 文件分隔符與換行；論文中明確報告分句方法（R2 已點名）。

---

### 🟡 F-13. 其他

| # | 問題 | 位置 |
|---|---|---|
| a | GRASP 的建構階段用 `α·score − (1−α)·max_sim`，局部搜尋卻用 `score − (1−α)·Σ pairwise` —— 兩個階段最佳化的目標不同，且後者隨集合大小二次成長 | `grasp.py:8-19` vs `:41` |
| b | v1 `sentence_tf_isf_scores` 對 token 加總不做長度正規化 → 實質是句長的代理變數（v2 有除以 `√len`，但 v1 是預設值） | `tf_isf.py:48-72` |
| c | `requirements.txt` 中 `pymoo` 與 `scikit-learn` 各被宣告兩次且版本規格衝突（`pymoo==0.6.1.1` vs `pymoo>=0.6.0`） | `requirements.txt` |
| d | `pytest` 未安裝，`tests/` 無法執行 | — |
| e | `_minmax_norm`（fast_fused）在常數輸入時回傳 0.0，`_minmax_normalize`（compose）回傳 0.5 —— 行為不一致 | `fast_fused.py:8` vs `compose.py:4` |
| f | `optimizer_dispatch.py:87` 用 `except (ImportError, Exception)` 吞掉所有例外並靜默退回 greedy —— 實驗可能在你不知情的情況下跑了 greedy 而非 NSGA-II | `optimizer_dispatch.py:87` |

> ⚠️ **(f) 值得特別注意**：若某次實驗中 NSGA-II 因故拋出例外，程式只會印一行 warning 就改跑 greedy，而 `metrics.csv` 不會留下任何記號。**建議在重跑所有實驗前先把這個 except 拿掉**，確認沒有實驗其實是 greedy 跑出來的。

---

### 🟡 F-14. raw-sum 禁令只涵蓋「有 task_profile 的 multi_sentence」，legacy_unprofiled 路徑可繞過

**與 F-7 的關係**：F-7（§0.0 狀態表第 38 行）的「新 pipeline」欄已改為限定範圍的說法，並註明「legacy_unprofiled（無 `task_profile`）與 legacy 保留 sum」——本條目就是這個已知例外的具體化：明確給出讓 raw sum 在 profiled 路徑之外實際生效的觸發條件、證據行號與測試缺口。

**發現日期**：2026-07-27（investigation agent 覆核，未執行任何 git 操作）。

**觸發條件**（需三者同時成立）：
1. 傳入 `summarize_one()` 的 `doc` **沒有 `task_profile` 欄位**（例如舊格式 flat JSONL：`{"id": ..., "sentences": [...], "highlights": ...}`）
2. `optimizer.method` 設為 `nsga2`（或 `fast_nsga2`，走同一個 `objective_spec.importance_aggregation` 轉發路徑）
3. config 的 `objectives.importance_aggregation` **未被顯式設定**

**證據**：

- `src/objectives/factory.py:44` 的 `if not task_profile:` 分支（legacy_unprofiled）在 `:52-54` 直接取
  ```python
  "importance_aggregation": str(
      objective_cfg.get("importance_aggregation", "sum")
  ),
  ```
  沒有任何值域限制。
- 唯一擋 raw sum 的檢查在 `:87-92`：
  ```python
  aggregation = str(objective_cfg.get("importance_aggregation", "mean")).lower()
  if aggregation not in {"mean", "length_normalized"}:
      raise ValueError(...)
  ```
  這段程式碼**只存在於「有 `task_profile` 且 `output_mode == "multi_sentence"`」的分支**（`:59-64` 的 mode 檢查之後）。`output_mode == "single_sentence"` 的分支（`:75-85`）把值覆寫成 `"single_item"`，也不受影響。
- `src/pipeline/select_sentences.py:78-116` 的 `summarize_one()` 對傳入的 `doc` **沒有呼叫** `src/data/schemas.py` 的 `validate_document_example()`（該函式在 `:157-159` 才會強制檢查 `task_profile` 必須存在），而是直接呼叫 `flatten_sentence_records(doc)`（`:79`）與 `doc.get("task_profile")`（`:116`）。因此沒有 `task_profile` 的 dict 目前仍能被完整跑過整條 selector pipeline。
- `tests/test_pipeline_integration.py:241-244`（`test_single_sentence`）目前仍在用這種無 `task_profile` 的 dict 呼叫 `summarize_one()`，證明這條輸入路徑**不是理論可能性，而是測試套件現在就在用的真實輸入形態**（該測試本身不涉及 nsga2，只是證明該路徑活著）。
- `optimizer_dispatch.py:96-98` 把 `(objective_spec or {}).get("importance_aggregation", "sum")` 原樣轉給 `nsga2_select(...)`，沒有第二層檢查。

**影響範圍**：**潛在缺陷，尚未確認實際發生。** 沒有證據顯示現有 `runs/` 底下任何一個 run 是用「無 `task_profile` 的資料 + method=nsga2 + 未設 importance_aggregation」這個組合跑出來的——但也沒有任何機制阻止未來這樣跑，也沒有測試會在這個組合發生時發出警告。CLAUDE.md 與 `docs/research/ACTION_PLAN.md` 1e 目前的敘述提到「canonical multi-sentence 已禁止 raw sum」，沒有標明這個例外，容易讓人誤以為 raw sum 已經被全面擋下。

**測試覆蓋**：`tests/test_pipeline_integration.py:275-298` 只測了「有 `task_profile`」的兩種情形（`mean` 預設、`sum` 被拒）。**沒有任何測試**針對「無 `task_profile` + method=nsga2 + 未設定 importance_aggregation」這個組合去驗證 "sum" 是否真的流到 `nsga2_select`。

**建議修法（未動手，待你決定方向）**：
- 選項 A：讓 `factory.py:44` 的 legacy_unprofiled 分支也套用跟 profiled 分支相同的 mean/length_normalized 限制；風險是可能讓依賴 legacy flat JSONL 重現的既有腳本失敗，需先確認是否還有人在跑這條路徑。
- 選項 B：在 `select_sentences.py` 入口強制呼叫 `validate_document_example()`，讓沒有 `task_profile` 的資料直接被擋在門口；風險是會移除目前對舊格式的相容性，需先確認這個相容性是否還要保留。
- 兩種選項都需要新增覆蓋「無 task_profile + method=nsga2 + 未設定 importance_aggregation」組合的 regression test。

**目前狀態**：🔴 未修。本條目只記錄發現，不代表已有任何程式碼變更。

---

### 🟠 F-15. fusion+NSGA-II 與其宣稱的對照組不是 matched condition

**發現日期**：2026-07-27（investigation agent 覆核，未執行任何 git 操作）。

**背景**：`2_Fusion_ExpA/B/C/Final.yaml`（`optimizer.method: fast_nsga2`）與 `2_Fusion_NoNsga2.yaml`（`optimizer.method: fast_fused`，檔案註解自稱「Ablation: Without NSGA-II」）常被拿來當作「有沒有用 NSGA-II」的對照組。逐行追蹤兩條路徑實際執行的程式碼後，兩者的差異遠不只「optimizer 有沒有換掉」。

**呼叫鏈**：

- `2_Fusion_ExpA/B/C/Final.yaml` → `optimizer_dispatch.py:141-163`（`fast_nsga2` 分支）→ `fast_nsga2_select`（`src/models/extractive/fast_fused.py:93-129`）→ `nsga2_select`（`src/models/extractive/nsga2.py:105-173`，呼叫點在 `fast_fused.py:107,117`）
- `2_Fusion_NoNsga2.yaml` → `optimizer_dispatch.py:116-125`（`fast`/`fast_fused`/`tfidf_fused` 分支）→ `fast_fused_select`（`fast_fused.py:27-54`）→ `greedy_select`（`src/models/extractive/greedy.py:7-50`，呼叫點在 `fast_fused.py:37,45`）

**逐項對照**：

| 項目 | `fast_nsga2`（→ `nsga2.py`） | `fast_fused`（NoNsga2，→ `greedy.py`） |
|---|---|---|
| 目標函數項數 | **三個獨立目標**：`out["F"] = [-imp, -cov, red]`（`nsga2.py:94`） | **單一純量分數**：`score = alpha*base_scores[i] - (1-alpha)*max_sim`（`greedy.py:32`） |
| Coverage 項 | **有**：`_compute_coverage(sim_mat, idx, coverage_method)`（`nsga2.py:86`），對全部句子算「與已選集合最大相似度的平均」（`_coverage_max`，`nsga2.py:16-19`） | **完全沒有**。`greedy.py` 全文（1-50 行）沒有任何 coverage 計算 |
| Redundancy 公式 | `red = mean(已選集合內部所有配對相似度的上三角)`（`nsga2.py:88-92`），在 `_evaluate` 對整個候選子集一次算完 | `max_sim = max(候選句 i 與已選集合中任一句的相似度)`（`greedy.py:29`）——不是平均、不是配對加總，是 MMR 型的 max-similarity |
| Redundancy 作用時機 | 事後從 Pareto front 挑解時的線性權重（`nsga2.py:162`），**不影響** NSGA-II 搜尋過程本身（`_evaluate` 未使用 `lambda_*`） | 每一步貪婪決策當下就直接使用，驅動選句過程本身 |
| 搜尋方式 | population-based 多目標 Pareto 搜尋（NSGA-II，`nsga2.py:132-146`） | 單輪貪婪、一次通過（`greedy.py:20-47`），無族群、無世代、無 Pareto front |
| 參數來源 | `objectives.lambda_redundancy`（`optimizer_dispatch.py:159`） | `redundancy.lambda`（`optimizer_dispatch.py:120`）——**不同的頂層 key** |
| 實際生效的冗餘權重 | **`1.2`**（`2_Fusion_ExpB.yaml` 明確宣告 `objectives.lambda_redundancy: 1.2`） | **`0.7`**（硬編碼預設值，來源 `optimizer_dispatch.py:120`；`2_Fusion_NoNsga2.yaml` 沒有 `redundancy:` 區塊，該檔宣告的 `objectives.lambda_redundancy: 1.2` 在這條路徑完全是死值，`optimizer_dispatch.py:116-125` 從未讀取 `cfg.get("objectives")`） |

兩個「實際生效的冗餘權重」不只數值不同（1.2 vs 0.7），角色也不同：一個是事後 Pareto-front 選解時乘在「原始 mean-pairwise 相似度」上的權重（不影響搜尋過程），另一個是每一步貪婪決策當下直接使用、同時決定 importance 與 redundancy 相對權重的係數（`alpha` 與 `1-alpha`）。兩者連量綱都不可比。

**影響**：共同作者已確認 `2_Fusion_NoNsga2.yaml` 的設計意圖就是 NSGA-II 消融組（不是另一條獨立 baseline）。因此這是**方法學缺陷，不是命名或論文呈現問題**：此消融比較為 **confounded**，至少同時改變了三個變因：(1) coverage 目標的有無、(2) redundancy 的計算公式與作用時機、(3) 實際生效的冗餘權重數值（1.2 vs 0.7，且不是同一個 cfg key、不是同一個數學角色）。任何基於「`fast_nsga2` vs `fast_fused`（NoNsga2）」這組對照所做的「NSGA-II 帶來多少貢獻」的結論，都無法把觀察到的指標差異單獨歸因於「NSGA-II 這個搜尋演算法本身」。

**Hypothesis（⚠️ 未實測，純方向性推論，不是結論）**：對照組（NoNsga2）同時失去了 coverage 目標，而不只是失去 NSGA-II 的搜尋機制。若 coverage（對整篇文件的代表性）原本是系統表現的重要來源之一，那麼目前的消融設計可能會讓「移除 NSGA-II」看起來造成更大的指標掉幅——也就是**可能高估 NSGA-II 本身的貢獻**，因為掉幅裡混了「失去 coverage 目標」的效果。這只是一個待驗證的方向；究竟是高估、低估、或影響方向不定，必須做一次真正 matched 的消融（同樣有 coverage 項、同一個 redundancy 公式與權重來源，唯一差異是搜尋演算法）才能確認，目前完全沒有實測數字支持或反駁這個 hypothesis。

**附帶發現**：`fast_nsga2` 分支（`optimizer_dispatch.py:141-163`）從不轉發 `objectives.coverage_method` 給 `nsga2_select`——即使 config 想調整 coverage 計算方式也調不到，`coverage_method` 永遠吃 `src/models/extractive/nsga2.py:118` 的函式預設值 `"max"`。相較之下，plain `nsga2` 分支（`optimizer_dispatch.py:92`）會讀取並轉發這個鍵。

**目前狀態**：🔴 未修。設計意圖已由共同作者確認為消融組，故此為方法學缺陷；上方 hypothesis 尚未實測。本條目只記錄程式碼行為與其造成的比較混淆，不代表已有任何程式碼或 config 變更。

---

### 🟡 F-16. Lead baseline 的 `min_words` 全量分布，及與 F-1e 72 列的交叉驗證

**發現日期**：2026-07-31（PR #10 review，`src/baselines/lead.py` / `src/baselines/contract.py`）。

**背景**：`src/baselines/lead.py` 的 `document_order` Lead 現在一律以 `apply_min_words=False` 呼叫 `summarize_one_baseline`（`SelectionConstraints.min_words` 恆為 0），理由見該檔案 `LEAD_MIN_WORDS_NOT_APPLIED_REASON`：(1) 技術理由 —— `resolve_effective_min_words` 的容量由 `maximum_feasible_words` 以**任意子集**的 bitset subset-sum 計算，Lead 只能取前綴，落點是離散且遠比任意子集稀疏的集合，`[min_words, max_words]` 窗口對前綴型方法可能結構性無解；(2) 方法理由 —— `ACTION_PLAN.md` 1e 記載 `min_words=200` 是為了防止 mean-salience 目標退化成單句，Lead 沒有選句目標函數，不存在該退化，這個 guard 對它不適用。

**實測**（`min_words=0`，5,621 篇 Multi-News validation，`document_order` Lead，`max_words=250`）：

- 零失敗：5,621/5,621 篇皆產生 feasible 摘要。
- 選取字數分布：mean 233.6、median 238、max 250、min 22。
- `<200` words 共 **212 筆（3.77%）**，可再拆解為兩個不重疊的子群：
  - **72 筆**是來源本身容量不足（`source_capacity_words < 200`，即使不受 Lead 前綴限制、改用任意子集也拿不到 200 字）—— 這與 `paper_revision_plan_IEEE_Access.md`（588 行）記載的、`1e` 的 length-feasibility audit 找到的 **72/5,621 列全文不足 requested `min_words=200`** 完全吻合。兩邊是**獨立算出**的同一組列（那邊用 `maximum_feasible_words` 做 exact attainable-capacity 計算；這邊是 Lead 前綴路徑 + `source_capacity_words` 診斷欄位），構成交叉驗證，而不是同一次計算的重複引用。
  - **140 筆（2.49%）**是前綴落點無解：來源本身可以任意子集湊到 ≥200 字（`source_capacity_words >= 200`），但 `document_order` 的嚴格前綴走法在某一句放不下後就停止，實際選到的字數落在 200 之下。這是**所有前綴型方法共有的性質**（先到先得、不回頭補洞），不是這次實作的缺陷——`fabbri_first_k`、任何「取前 N 句/前 N 字」的 baseline 都會有同一種落差。

⚠️ **範圍聲明——不要宣稱這影響任何既有結果**：本條目只描述 `min_words` 對 Lead 的（不）適用性與其字數分布，**不涉及、也不改變**任何 ROUGE 數字。CLAUDE.md 第 2 節已記載的舊 Lead 分數（例如 `0.4331 / 0.1453 / 0.3901`）與其 R-Lsum 分量，其陳舊狀態由 `c23d1a9`（分句器換成共用 Punkt tokenizer 後，全域標記所有既有 ROUGE-Lsum 數字過期）決定，與本條目無關；本條目不構成、也不應被引用為那些數字的重新驗證。

**目前狀態**：✅ 已驗證，非缺陷。`min_words_applied: false` 與 `min_words_not_applied_reason` 已進 `output_budget` artifact（見 `tests/test_baselines_lead.py`），使這個例外可被逐篇稽核。

---

### ✅ F-17. `min_words` 下界對著「沒有任何 selector 保證達得到」的容量定義，已採 option 1

**發現日期**：2026-08-03（第一次 validation pilot；`src/models/extractive/greedy.py:87`）。

**症狀**：以 `configs/phase1_mvp_multinews.yaml` 跑全量 validation，在**第 4,066 篇**中止：

```
Summarizing: 4066it [09:16]
  greedy.py:87  evaluator.assert_feasible(selected)
ValueError: selector returned an infeasible summary: {'min_words': 15.0}
```

因為 `write_jsonl_atomic` 消費 generator，**整整 9 分 16 秒的計算全部作廢，`predictions.jsonl` 不會產生**。全量 5,621 篇只有 **1 篇（0.02%）** 觸發。

**逐項診斷**（`validation_4066`）：

- 該文件只有 **15 句**，候選池上限是 60 —— **完全沒有發生池子縮減**，池子就是全文。因此**不是 candidate-induced**。
- `source_capacity_words = 250`、`effective_min_words = 200`、`min_words_relaxed = False` —— 放寬邏輯正確地判定「來源達得到 200 字」，因此**也不是 relaxation bug**。
- 15 句的長度是 `[13, 12, 3, 4, 22, 4, 15, 35, 40, 2, 140, 9, 18, 6, 2]`（總和 325）。greedy 依效用先吃掉 14 個短句 = **185 字**，只剩 140 字那句，`185 + 140 = 325 > 250` 上限 → `can_add` 為 False → 迴圈以 `if not ranked: break` 結束，停在 185 < 200。
- **可行解確實存在**：窮舉後落在 `[200, 250]` 的子集有 **428 組以上**（例如 `{22, 40, 140} = 202` 字）。greedy 拿不到，是因為**它沒有回溯**——早期為效用 commit 到短句，之後長句再也塞不進去。

**根因（這一條是重點，四個看似獨立的失敗是同一件事）**：

`resolve_effective_min_words` 的放寬目標是 `maximum_feasible_words`，那是**任意子集的精確 bitset subset-sum 最佳解**。但實際的 selector 沒有一個是最佳裝箱器：

| selector | 為什麼不是最佳 | 實測失敗率 |
|---|---|---|
| Lead（`document_order`） | 閱讀順序嚴格前綴，不回頭補洞 | 140/5,621（F-16） |
| Random（`_select_random`） | 隨機排列 first-fit | 2–4/5,621（PR #11） |
| **greedy** | **短視效用最大化，無回溯** | **1/5,621（本條）** |
| Random（樸素 stop 變體） | 同 Lead 的停止規則 | 130–146/5,621（PR #11） |

**下界是對著一個沒有任何 selector 保證達得到的容量定義的。** PR #10 第一版的 Lead、PR #11 的 Random、以及本條的 greedy，是同一個根因的三種表現。GRASP 與 NSGA-II 尚未逐一驗證（NSGA-II 在本次全量 run 中為零失敗，見 F-18）。

**修正決策（2026-08-04，PR #12 修正版）**：採用 option 1，**逐篇記錄不可行並繼續**；不替 selector backfill、不放寬 candidate-induced shortfall，也不把 NSGA-II／GRASP 靜默換成 greedy。這保留「某 optimizer 找不到可行解」作為研究結果，而不是把它修掉。

實作 contract 不只捕捉 `greedy.assert_feasible`：candidate capacity shortfall、Greedy／GRASP／NSGA-II 的 least-violating attempted solution、空來源與無 eligible sentence 都會產生一列完整 artifact，包含 `feasible=false`、machine-readable `infeasible_code`、文字 reason、violations 與 selection evaluation。只有 `max_length`／`max_sentences` 上界違規、schema/config 錯誤與 route/model failure 仍中止整批，因為那些不是合理的文件層結果。

評估政策同時修正：**正式 primary 預設計分 all rows 並另報 infeasibility rate**；`--feasible-only` 是單 run 診斷，跨方法則只能用共同 feasible ID intersection 作 paired sensitivity。不得讓各方法排除各自失敗列後，把不同 denominator 的 ROUGE 放在同一表直接比較。legacy 缺列 artifact 必須顯式 `--assume-legacy-feasible`，且只保留 diagnostic 身分。

**驗收結果（2026-08-04）**：單元／整合／negative tests 已增至 **261 passed**。完整 governed Multi-News validation 成功產生 **5,621/5,621 rows**，其中 5,620 feasible、1 recorded infeasible；唯一一列仍是 `validation_4066`，保留 185-word attempted summary、`min_words` shortfall 15，而非中止。primary all-rows R1/R2/Lsum 為 `0.423018 / 0.129178 / 0.372800`；5,620-row feasible-only sensitivity 為 `0.423007 / 0.129171 / 0.372792`，證明本案例排除與否只影響約 `1e-5`，但正式 denominator 仍固定用 all rows。selection time 為 2,146.53 秒（本機 CPU；成本數字不可脫離 hardware 環境引用）。

可追蹤的 dataset identity、commit、artifact SHA-256、metrics 與 timing 摘要：`docs/research/evidence/f17_pr12_validation_regression.json`。495 MB predictions 保持本機 bulk artifact，不進 Git；該 SHA-256 只能驗證同環境下的完全一致；跨機器重現請比對逐篇 selected_indices，見下方判讀規則。

> **跨機器可重現性的判讀規則**：先分開比較 dataset/config identity、逐篇
> `selected_indices`、逐篇 metrics 與序列化檔案。即使方法沒有 PLM、optimizer
> 或隨機性，dependency、排序 tie-break、文字編碼與換行仍可能改變 artifact；
> 只有完整鎖定環境與 serialization contract 時才能要求 byte identity。
> `5e-6` 是本次兩個 evaluator 環境的**觀察值**，不是所有 PLM／optimizer run
> 的通用容忍度。任何差異都須先定位到哪一層，不能先宣布是浮點噪音。

**重現**：`data/processed/multi_news_validation_canonical.jsonl` 第 4,066 列（`validation_4066`），config `configs/phase1_mvp_multinews.yaml`。

---

### 🟠 F-18. 第一次 validation pilot：`mean` 節流、長度括弧、與 §7.3 初步結果

**量測日期**：2026-08-03。**全部是 diagnostic，不是 Gate 2 結果**（見末尾適用範圍）。

Lead 的 governed baseline artifact 已保存於 `runs_v2/gate2_lead_document_order_validation/`
（5,621 篇、14.9 秒、`0.433204 / 0.146768 / 0.394039`）。以下系統端量測因 F-17 而**跳過不可行文件後繼續**，故各 run 的文件數略有差異（5,613 / 5,620 / 5,621）；Lead 在三組上分別為 `0.4332 / 0.4333 / 0.4332`，交叉比較安全。

#### (a) `importance_aggregation: mean` 在節流輸出

| config | R-1 | R-2 | R-Lsum | 句/篇 | 字/篇 | 不可行 |
|---|---|---|---|---|---|---|
| greedy + `mean` | 0.4230 | 0.1292 | 0.3728 | **6.05** | 227.0 | 1 |
| greedy + `length_normalized` | **0.4347** | **0.1354** | **0.3960** | **13.47** | 244.0 | 8 |

用 `mean` 時，一旦 `min_words` 滿足，再加入任何低於當前平均的句子都會**降低**目標值，greedy 因此停手 —— 句數只有 Lead 的一半多。改成 `length_normalized`（`factory.py` 允許的另一個值；raw `sum` 因 F-14 被禁）後三項全面上升，**R-Lsum +0.0232**。

> ⚠️ 副作用：不可行文件由 1 篇增為 8 篇（`length_normalized` 偏好高分密度短句，更容易湊不到下界）。

#### (b) 長度括弧：沒有任何配置贏過 Lead

以 `scripts/audit/length_matched_lead.py` 對 `greedy + length_normalized` 產生上下界（句子粒度使精確等長不可能，故必須兩側都報）：

| | R-1 | R-2 | R-Lsum | 字/篇 |
|---|---|---|---|---|
| Lead，對齊系統長度（**不足**） | 0.4324 | 0.1460 | 0.3931 | 229.4 |
| Lead，固定 250 字預算 | 0.4333 | 0.1468 | 0.3941 | 233.6 |
| **系統（greedy + `length_normalized`）** | **0.4347** | **0.1354** | **0.3960** | **244.0** |
| Lead，對齊系統長度（**超過**） | **0.4354** | **0.1495** | **0.3965** | 258.8 |

按字數排序，R-1 與 R-Lsum **單調遞增**（0.4324 → 0.4333 → 0.4347 → 0.4354；0.3931 → 0.3941 → 0.3960 → 0.3965），系統的位置剛好對應它的字數。

**結論：系統看似領先的 R-1 (+0.0014) 與 R-Lsum (+0.0019) 完全由多用的 10.4 個字解釋。給 Lead 同等字數，Lead 三項全勝。** R-2 更直接 —— Lead 在三種長度下都是 0.146–0.149，系統 0.1354，**在任何長度下都輸 0.011–0.014**。

> ⚠️ **機制修正，結論不變**（2026-08-05，加入 TextRank／LexRank baseline 時發現）：
> 上面「系統多用 10.4 個字」的原始寫法容易讀成「系統這個方法本身」的性質。
> 實測顯示這其實是**填充規則**的性質，不是任何單一方法的性質——
>
> | baseline | 字/篇（全量 5,621 篇） | 填充規則 |
> |---|---|---|
> | Lead（`document_order`） | **233.6** | **stop-tolerant**：碰到第一個放不進的句子就停 |
> | Random（skip-tolerant fill） | 246.83 | skip-tolerant：放不進就跳過，繼續嘗試後面的句子 |
> | 系統（greedy + `length_normalized`） | 244.0 | 邊際效用搜尋，非嚴格 stop-at-first-miss |
> | TextRank（`select_by_score`） | 247.35 | skip-tolerant（見 `src/baselines/contract.py`） |
>
> 四個 baseline 裡，**Lead 是唯一的異常值，也是唯一用 stop-tolerant 規則的
> 那一個**；其餘三個不論方法本身是什麼（隨機、邊際效用搜尋、centrality
> 排序），只要填充規則是 skip-tolerant，字數都貼近 250 上界、彼此相差
> 不到 3 個字。也就是說：**只要用 skip-tolerant 填充規則的方法，都會
> 系統性地比 Lead 多用 13–14 個字**，這是規則造成的字數差，不是
> 系統這個方法特有的優勢。原始的「10.4 個字」結論本身沒有錯（那次比較
> 的兩個對象確實差 10.4 字），但把它寫成「系統的字數性質」而非「
> skip-tolerant 填充規則的通性」會讓 reviewer 誤以為系統的 R-1/R-Lsum
> 領先有某種方法特有的理由——沒有，純粹是填充規則。
>
> **Lead 用 stop-tolerant 是刻意選擇，不是缺陷**：標準 Lead-3／First-k
> 定義就是照閱讀順序取前 k 句，跳句就不再是「Lead」這個基準線本來的定義
> （見 `src/baselines/lead.py` 模組 docstring）。但這個刻意選擇有一個
> 系統性代價——讓 Lead 在同一個字數上界下，比任何 skip-tolerant 方法
> 少用約 13 個字——**論文必須明寫這一點**，否則 reviewer 會問「為什麼
> Lead 用不到預算」，而正確答案是「這是 stop-tolerant 定義的必然結果，
> 不是 Lead 這個基準線太弱」。
>
> **長度括弧分析升級為通用 follow-up，不是逐一為每個新 baseline 各做一次**：
> 本節與 (b) 目前只對 `greedy + length_normalized` 做過括弧。既然差距的
> 機制是通用的（填充規則，不是方法），正確的做法是**做一次通用的
> length-matched bracket 基礎設施，套用到所有 skip-tolerant baseline／
> 系統 vs. Lead 的比較**，而不是每加一個新方法（TextRank、LexRank、未來
> 的 SBERT centroid 等）就重做一次 `scripts/audit/length_matched_lead.py`
> 那樣的一次性分析。這是一項待排入 follow-up 的通用基礎設施工作，本輪
> 只記錄機制，不實作。

#### (c) §7.3 NSGA-II 生存 gate 初步結果

同一 objective（`mean`）、同一候選池、同一預算，僅更換 selector：

| | greedy | NSGA-II | Δ |
|---|---|---|---|
| R-1 | 0.4230 | 0.4242 | +0.0012 |
| R-2 | 0.1292 | 0.1299 | +0.0007 |
| R-Lsum | 0.3728 | 0.3767 | **+0.0039** |
| 字/篇 | 227.0 | 226.8 | −0.2 |
| 不可行文件 | 1 | **0** | −1 |
| 選句時間 | ~10 分 | **322 分** | **32×** |

三項均正、且字數幾乎相同（故非長度效應），§7.3 **條件 1 名目成立**。NSGA-II 另有一項獨立優點：**零不可行文件**（族群搜尋找得到 greedy 因無回溯而錯過的可行解，見 F-17）。

但 quality-cost 很難講：

| 改動 | ΔR-Lsum | 成本 |
|---|---|---|
| `mean` → `length_normalized`（改一行 config） | **+0.0232** | 10 分 |
| greedy → NSGA-II（5.4 小時搜尋） | **+0.0039** | 322 分 |

**目標函數的選擇比最佳化演算法重要約 6 倍。** 尚未量測的 §7.3 條件 2（等品質下的 coverage/redundancy Pareto 優勢）與條件 3（跨 budget 的穩定 operating points）仍可能成立。

> 附帶驗證：本次 322 分鐘與 `docs/research/` 先前估計的 legacy NSGA-II「平均 5.0 小時」一致，即 §9 ablation 矩陣（8 配置 × 2 資料集 × 5 seeds ≈ 399 小時）的估計**成立**，排程時必須計入。

#### (d) 🔴 用 `mean` 時系統低於 Random baseline

PR #11 的 Random baseline（seed 0、5,621 篇）：`0.416164 / 0.121989 / 0.378817`。

| | R-1 | R-2 | R-Lsum |
|---|---|---|---|
| Lead | 0.4332 | 0.1468 | 0.3940 |
| **Random** | 0.4162 | 0.1220 | **0.3788** |
| 系統 greedy + `mean` | 0.4230 | 0.1292 | **0.3728** ❌ |
| 系統 NSGA-II + `mean` | 0.4242 | 0.1299 | **0.3767** ❌ |
| 系統 greedy + `length_normalized` | 0.4347 | 0.1354 | 0.3960 ✅ |

**`mean` 配置下，greedy 與 NSGA-II 的 ROUGE-Lsum 都低於隨機抽樣。** 這正是 Random baseline 存在的理由（「打不贏隨機抽樣的方法，問題在調參之前就出了」），且沒有 PR #11 就看不到這個訊號。改用 `length_normalized` 後三項均超過 Random。

#### (e) 選句與 Lead 的重疊率

`scripts/audit/selection_overlap.py`，全量、以 `sentence_id` 比對（不受排序影響）：

| config | 重疊率（平均） | 中位數 | Jaccard | 與 Lead 完全相同 |
|---|---|---|---|---|
| greedy + `mean` | 27.5% | 25.0% | 14.7% | 124（2.2%） |
| greedy + `length_normalized` | 24.3% | 18.8% | 17.9% | 131（2.3%） |
| NSGA-II + `mean` | 27.6% | 22.2% | 15.5% | 125（2.2%） |

**候選池漏斗確實打開了** —— 系統不再是「昂貴版的 Lead」。但如 (b) 所示，**不再像 Lead 並未轉化為品質**：診斷正確、修法照做、結果仍不如 Lead。

> ⚠️ 舊 diagnostic 的 **61.7%** 來自不同 split、200 篇抽樣、test-tuned artifact，**方向可比、數值不可相減**。

#### 適用範圍（引用前必讀）

- ✅ **已依新 pipeline 以 primary all-rows 協議重算**（2026-08-05）。
  ⚠️ **但本節仍全部是 diagnostic，不是 Gate 2 結果**——重算只換掉了計分
  協議，沒有改變下面三條限制中的任何一條。三格同分母 5,621 篇：

  | config | R-1 | R-2 | R-Lsum | 不可行 |
  |---|---|---|---|---|
  | Lead (document_order) | 0.433204 | 0.146768 | 0.394039 | 0 |
  | greedy + mean | 0.423018 | 0.129178 | 0.372800 | 1 |
  | greedy + length_normalized | 0.434669 | 0.135345 | 0.395945 | 8 |

  ⚠️ `length_normalized` 在 R-1／R-Lsum 上高於 Lead，但那是長度效應：
  該配置平均 244.0 字、Lead 233.6 字。以 §F-18 的長度括弧對照，Lead 在
  244 字附近的 R-Lsum 約 0.395，與本表持平。R-2 則在所有長度下都輸
  0.011–0.014，且 selector 換成 NSGA-II 只給 +0.0007。詳見 (b)。

  provenance 必須分開標註，不要混記成同一次量測：
  - Lead：`runs_v2/gate2_lead_document_order_validation/`，commit `6abd4e9`。
    另一台機器曾回報六位小數相同；因兩端完整 predictions 未一併 version，
    這只支持 aggregate metric replication，不等於跨環境逐位元組確定。
  - greedy + `mean`：commit `6abd4e9`，見
    `docs/research/evidence/f17_pr12_validation_regression.json`。
  - greedy + `length_normalized`：原表的 5,621-row 數字來源沒有獨立
    versioned manifest／prediction hash，且「與 pre-fix artifact 逐篇相同」只
    存在敘述、沒有可重跑的 comparison artifact；因此已在本修正分支以 commit
    `d5346c0` 的 pipeline 與獨立 tracked config 全量重跑。最終產生 5,621 個
    unique rows、5,613 feasible／8 recorded，all-rows 為
    `0.434669 / 0.135345 / 0.395945`。完整 config、dependency、artifact hashes、
    8 個 ID 與 integrity checks 見
    `docs/research/evidence/f18_length_normalized_final_pipeline.json`；這份 manifest
    取代舊的 `0.434678 / 0.135353 / 0.395951`。

  final all-rows 下，mean → length-normalized 的 R-Lsum 差為 **+0.023145**；
  目前兩份 full predictions 的共同 5,613 feasible IDs 重算則為 **+0.023136**
  （mean `0.372865` → length-normalized `0.396001`）。共同 ID、兩組逐篇 ROUGE、
  metrics 與 report 已版本化於
  `docs/research/evidence/f18_paired_intersection/`。因此「objective 影響遠大於
  當時 greedy → NSGA-II 的 +0.0039」在 all-rows 與 paired-feasible 兩種視角
  方向一致；但尚未跑 paired significance test。舊的「跨集合 +0.0232」仍沒有
  可稽核 artifact，不再列作第三個已驗證協議。

  不可行率對 primary 數字的影響只能在各 run **直接量測**：mean 的 all-rows
  與 feasible-only R-Lsum 差 8e-6（1 篇）；本次 final length-normalized rerun
  實測差 5.6e-5（8 篇）。不能把「每篇平均影響」
  線性外推到 GovReport：不可行列的 ROUGE、資料列數、摘要長度與失敗機制都
  不同。GovReport 必須在 frozen split 上同時報 all-rows、infeasibility rate 與
  paired feasible sensitivity，不能沿用 Multi-News 的比例估計。
- ⚠️ **單一 seed、未做 paired bootstrap** —— 上表所有差距（含 +0.0039 與 −0.0174）**都尚未驗證顯著性**。
- ⚠️ **MVP config only**：`enabled_routes: [lexical, semantic]`，**沒有 graph 軌**；`position` 與 `length` 特徵權重皆為 0。因此 (b) 不是「完整架構打不贏 Lead」的結論，(c) 也不是 §7.3 的最終裁決。
- ⚠️ **尚未跑過的關鍵組合**：NSGA-II + `length_normalized`（目前最佳 objective 配最佳 selector）、以及開啟 graph 軌的任何配置（§5.4 刪除條件）。
- 重現腳本：`scripts/audit/length_matched_lead.py`、`scripts/audit/selection_overlap.py`。

---

### 🟠 F-19. TextRank／LexRank baseline：最終實作的 Multi-News full-split 結果，與 TextRank 的長句偏好證據

**量測日期**：2026-08-05。下表的 TextRank／LexRank 已由 PR #15 merge commit
`c38dfae7cfd022d9bd1b90961d14fa1c59626578` 的最終 offline tokenizer 實作重跑，
不是先前 commit `b9b7fb8` 的 historical diagnostic。兩個 baseline 均單程序跑完整
frozen Multi-News validation，保存 5,621-row predictions、selected indices、metrics、
config、preflight、feasibility 與 SHA-256；完整 manifest 見
`docs/research/evidence/f19_centrality_final_pipeline.json`。**但未做 paired significance、
未跑 GovReport，因此這仍只是 Gate 2 的一部分，不是 Gate 2 完成。**

以新 all-rows 預設協議（PR #12 之後 `evaluate` 的預設）、`protocol multisentence_lsum`、分母皆為 **5,621**：

| | R-1 | R-2 | R-Lsum | 字/篇 |
|---|---|---|---|---|
| Lead | 0.4332 | 0.1468 | 0.3940 | 233.6 |
| Random (seed 0) | 0.4162 | 0.1220 | 0.3788 | 246.8 |
| 系統 greedy + `mean` | 0.4230 | 0.1292 | 0.3728 | 227.0 |
| 系統 greedy + `length_normalized` | 0.4347 | 0.1354 | 0.3960 | 244.0 |
| **TextRank** | **0.413845** | **0.128837** | **0.368487** | 247.35 |
| **LexRank** | **0.430671** | **0.135995** | **0.389532** | 246.99 |

**TextRank 是六個方法裡 R-1 與 R-Lsum 最低的**（R-1 甚至低於 Random；R-2 與 Random、系統 `mean` 相近）。**LexRank 全面優於 TextRank**。目前系統 `length_normalized` 相對 LexRank 是 R-1 +0.003998、R-2 −0.000650、R-Lsum +0.006413；相對 Lead 則是 R-1 +0.001465、R-2 −0.011423、R-Lsum +0.001906。這是混合結果，不可寫成「全面勝過 baseline」，且尚未做 paired significance。

#### 機制證據：TextRank 分數與句長有強關聯，LexRank 較弱

> 本節的 full-split 選句長度分布與最終 rerun 一致；但 300 篇 score-correlation
> 稽核是在 pre-hotfix scorer 上量測。因 word-token stream 已知有 1,550 句不同，相關係數
> 必須用最終 scorer 重算後才能放進論文；目前只保留為機制假說的 historical diagnostic。

觀察：TextRank 平均 8.08 句/篇、34.01 字/句；LexRank 11.02 句/篇、24.07 字/句；Random 13.11 句/篇、18.8 字/句。TextRank 是六者中唯一明顯偏長句的，34.01 字/句接近 F-18(a) `mean` 病理的 ~37.5 字/句（227.0/6.05）——但**這是表面症狀相似，不是同一個機制**，見下方判讀。

**驗證假說**：sumy TextRank 的邊權重是「共同詞數 ÷ (log(句1長)+log(句2長))」（`TextRankSummarizer._rate_sentences_edge`），假設 log 正規化對長句不足以抵銷詞彙重疊機會的增加。全量 5,621 篇的選中句 vs. 候選池（全文件句子）字數分布：

| | 候選池 mean | 候選池 median | 選中 mean | 選中 median | 選中−池 mean |
|---|---|---|---|---|---|
| TextRank | 21.55 | 19 | **30.62** | **28** | **+9.07** |
| LexRank | 21.55 | 19 | 22.42 | 20 | +0.87 |

300 篇樣本、句子分數 vs. 字數的相關係數（同一文件內，避免跨文件分數尺度不同造成的混淆）：

| | pooled correlation | 文件內平均 correlation |
|---|---|---|
| TextRank | 0.2234 | **0.6601** |
| LexRank | 0.1498 | 0.2225 |

**證據支持此假說，但不是單獨的因果證明**：TextRank 文件內分數與句長的平均相關係數為 0.66，選中句平均比候選池長 9.07 字（約 42%）；LexRank（TF-IDF cosine，沒有相同的 log-length 邊權重）則為 0.22 與 +0.87 字。這與 sumy 對 Mihalcea TextRank 邊權重公式的實作機制一致；可在論文中寫成已量測的 baseline 特性，但不可寫成已排除主題、位置、詞彙密度等混淆因素的因果結論。

#### ROUGE 判讀：效應類別相同，機制不同，不可混為一談

TextRank 的 R-Lsum（0.3685）低於 LexRank（0.3894），同時伴隨較少選句（8.08 vs 11.02）與較長句子（34.01 vs 24.07 字）。這個型態**與** F-18(a) 的 `mean` 病理一致：句數少、句子長可能減少 ROUGE-Lsum 逐句 LCS 比對的獨立匹配機會；但目前沒有受控介入或 paired causal analysis，因此只能寫「consistent with」，不能寫「完全由此造成」。

但**根因不同，不能寫成同一個機制**：
- F-18(a) 的 `mean` 病理：**搜尋型目標函數的聚合規則**造成的提早停止——加入任何低於目前平均分數的句子會拉低 `mean`，greedy 因此主動停手。這是系統選句過程中的動態（贏了就停）。
- F-19 的 TextRank 偏誤：**base scorer 本身**對長句的系統性偏誤（分數與句長相關係數 0.66）——`select_by_score` 是單純 rank-then-fill，完全沒有聚合公式或提早停止的概念，句數變少純粹是因為「長句先被排到前面、把預算填滿得比較快」。

兩者都導致「句少字長 → R-Lsum 受損」這個**下游效應**，但上游成因一個是搜尋動態、一個是 scorer 本身的公式性質，論文若把兩者寫成同一件事會誤導審稿人。

#### 與文獻 [16] 的方向性對照（不是驗證，兩邊 pipeline 不同）

Table 6 的 Lead/LexRank/TextRank 數字「adopted from [16]」（見本文件 §0 附近的引文），前處理、分句、ROUGE 設定都與本專案不同，**不能當公平對照**——僅供方向參考：

| | 文獻 [16] R-1 | 本專案 R-1 | 差距 |
|---|---|---|---|
| TextRank | 0.4151 | 0.4139 | −0.0012（0.3%） |
| LexRank | 0.4124 | 0.4306 | +0.0182（4.4%） |

TextRank 與文獻數字意外地接近（0.3% 差距），LexRank 則明顯高於文獻數字（4.4%）——但因為兩邊 pipeline 不同（分句、預處理、可能連 split 都不同），**這個接近或差距本身不能解讀為「重現成功」或「重現失敗」**，只是記錄下來備查。

#### 適用範圍（引用前必讀）

- ⚠️ **final-implementation full split 不等於完整 Gate 2**：目前仍是單一 deterministic run，未做 paired bootstrap；上表所有差距皆未驗證顯著性，也尚未在 GovReport 重跑。
- ✅ **最終實作的本機 artifact 已完整產生**：兩份 predictions 均為 5,621 unique IDs、全數 feasible、0 selected-index／summary mismatch；LexRank 的兩個 scorer-degenerate rows 被顯式標記。bulk predictions 約 143 MB，仍只在本機 `runs_v2/` 且未進 Git；投稿／外部 artifact review 前必須上傳 immutable store 並依 manifest hashes 驗證，不得聲稱跨機器逐位元組相同。
- ⚠️ **TextRank/LexRank 皆為 MVP baseline 設定**（`length_gate=True`、`apply_min_words=False`），與系統兩列的候選路線／objective 設定不對稱，不是同一個 pipeline 的兩端。
- ⚠️ 300 篇相關係數來自 pre-hotfix scorer，且不是全量；方向可作假說，數值不得在最終 scorer 重算前引用。
- 最終實作 full-split provenance（dataset／artifact SHA-256、依賴、指令、runtime、integrity）：
  `docs/research/evidence/f19_centrality_final_pipeline.json`；pre-hotfix historical provenance
  仍保留於 `docs/research/evidence/f19_textrank_lexrank_baselines.json`，不可再當現行結果。
- offline word-only adapter 的全量 token 差異稽核（456,942 句、1,550 mismatch）：
  `docs/research/evidence/f19_word_tokenizer_parity.json`。
- 重現：`python -m src.pipeline.evaluate --pred runs_v2/gate2_<textrank|lexrank>_multinews_validation_offline_v1/predictions.jsonl --gold data/processed/multi_news_validation_canonical.jsonl --out <out>.csv --protocol multisentence_lsum`。

---

### ✅ F-20. validation 沒有 development holdout，反覆 pilot 會把 validation 變成另一個 tuning set

**發現**：2026-08-08 前的正式 policy 只區分 upstream `validation` 與 `test`；selector
pilot、objective diagnostic 及 baseline reality check 都可反覆看完整 validation，沒有凍結的
dev/dev-test membership。這不等同 P0-01 的 test contamination，但若繼續用同一批 5,621 rows
挑長度、route、objective、selector 與權重，最終 validation 數字也會因研究者反覆決策而偏樂觀。

**修正**：新增 `scripts/audit/freeze_validation_partitions.py`，只使用 row ID 與
`split=validation`，依 `sha256(seed\0row_id)` 建立 reference-blind 70/30 membership。
Multi-News 以 seed 3407 凍結為 dev 3,935、dev-test 1,686；manifest file SHA-256
`e61405482cda203c0bd50dda3e958986b11a51b48124129e68617f97ce9e42ee`。兩條 runner
均先對完整輸入執行既有 frozen data-policy preflight，再由 `src/data/partitions.py` 過濾；
因此不需也沒有修改既有 `multinews-validation-v1` policy。

**fail-loud 條件**：manifest file SHA、input file SHA、dataset name、row count、selected-ID
SHA 任一不符即停止；selected ID 遺失或重複也停止。run 另存不含大型 ID 清單的
`partition_preflight.json`。dev 可重複搜尋，dev-test 每個 configuration hash 只看一次。

**GovReport 後續狀態（F-22）**：canonical validation 建立後，已在任何該資料集方法分數
前以同一 seed/rule 凍結為 dev 681、dev-test 292；manifest SHA-256
`7a15ffbb87abe690fe4e72a1e0daf27bf34b3a3293371983ae8e362d06e2717e`。過去已看過的
Multi-News full-validation 結果仍保留並標為 historical diagnostic，不刪除、不重新包裝成
partitioned evidence。

**重現**：

```bash
python -m scripts.audit.freeze_validation_partitions \
  --input data/processed/multi_news_validation_canonical.jsonl \
  --output configs/validation_partitions/multinews_validation_dev_v1.json \
  --dataset Multi-News --seed 3407 --dev_fraction 0.70
```

---

### ✅ F-21. greedy reference 對 canonical schema 靜默產生 0，且搜尋／輸出順序不一致

**重現（修正前）**：`oracle_scores()` 固定讀 `d.get("sentences", [])`。canonical
Multi-News／GovReport 的來源位於 `documents[].sections[].sentences[]`，所以每列都得到空
source、空 prediction，最後可能正常結束並回報 0.0000，而非報 schema 錯誤。此外搜尋按
「句子被加入的順序」組 summary 計分，return 時才排序 indices；ROUGE-2/Lsum 因此可能
用一個不會被實際輸出的句序挑解。

**修正（2026-08-08）**：canonical 路徑改由 `flatten_sentence_texts()` 完整驗證並展平；
legacy 只接受明確的非空 `sentences: list[str]`，缺 source/reference、錯誤 schema、空 corpus、
非法 metric/budget 全部 fail loud。每一步 candidate summary 都先按 source index 排序，與
最後輸出完全相同。

**metric contract**：CLI 未指定 `--target_metric` 時會分別執行 `rouge1`、`rouge2`、
`rougeLsum` 三次，輸出每個 target 的 selections、三項 scores、平均字數與句數；report
明記 `exact_upper_bound=false`。舊 `greedy_oracle_summary`／`oracle_scores` 僅為 historical
script 相容 wrapper，新文件與程式一律使用 `greedy_reference_*`。

**驗收**：46 個 evaluation/schema/greedy-reference tests 通過，包含 canonical 非零、三
target 分離、schema fail-loud、source-order search regression。此階段是 correctness，尚未
產生任何 primary dev/dev-test greedy-reference 分數；Gate 2 仍待 A1 長度協定凍結後執行。

---

### ✅ F-22. GovReport Primary A 缺資料身分、結構與異常列政策，無法進 Gate 2

**發現**：舊計畫只寫「使用 GovReport」，沒有釘住資料來源。常見 flattened mirror 的
split count 與作者論文不完全一致，且會丟掉 section/paragraph 結構；若直接使用，graph／
structure route 的輸入定義與 denominator 都無法稽核。

**修正（2026-08-08，任何 GovReport 方法分數前）**：使用作者官方 archive，archive
SHA-256 `bedf7a78...cd3c`。只讀取兩個 official validation membership（CRS 362、GAO 612）
及其 payload，不讀 test membership 或 test payload bytes。preprocessor 保留 nested section
path、heading path、paragraph position、原始順序與 raw JSON SHA；GAO top-level `Letter`
paragraphs 依作者 README 排除，但保留其 subsections。

**異常列政策**：官方 974 個 validation IDs 中，CRS `98-228` 的 official `summary=[]`。
不捏造 target，依 raw JSON SHA 釘住後排除，canonical denominator 為 973（CRS 361／GAO
612）。canonical file SHA-256 `db8aa2b7...de678`，dataset fingerprint
`4f2506cb...fad7d`，U+FFFD 為 0。資料政策為
`configs/data_policies/govreport_validation_v1.json`；完整統計與來源／授權證據在
`docs/research/evidence/a3_govreport_validation_data_audit.json`。

**結構與 scaling 證據**：section tree path 與 paragraph position 的 sentence 缺失皆 0；
每列平均 316.7 句、最大 2,889 句，來源平均 8,072 words，reference 平均 570.2 words。
這確認 GovReport 適合測長文件 coverage，也揭露 dense `N×N` 方法在正式 run 前必須通過
memory/cost pilot；不能因跑不動而改 denominator。

**development freeze**：reference-blind seed 3407 manifest 已凍結 dev 681／dev-test 292，
file SHA-256 `7a15ffbb...e2717e`。此段只有 data-layer/reference-blind membership，沒有任何
system score；test split 未存取。

**重現**：

```bash
python -m src.data.preprocess_govreport \
  --archive data/raw/gov-report.tar.gz \
  --output data/processed/govreport_validation_canonical.jsonl \
  --exclusion_manifest data/processed/govreport_validation_exclusions.json
python -m scripts.audit.govreport_validation_audit \
  --input data/processed/govreport_validation_canonical.jsonl \
  --archive data/raw/gov-report.tar.gz \
  --exclusion_manifest data/processed/govreport_validation_exclusions.json \
  --replacement_manifest_out data/processed/govreport_validation_replacement_characters.json \
  --out docs/research/evidence/a3_govreport_validation_data_audit.json
```

---

### 🟡 F-23. `min_words` 不只是輸出協定；它大幅改變 Greedy trajectory，且仍可能在可行時未達 floor

**預註冊 dev 證據（Multi-News 3,935 rows，2026-08-08）**：A1 對四個長度協定
各跑 Lead document-order、Random seed 3407、lexical-only length-normalized Greedy。
同一 `max_words=250` 下，Lead 與 Random 在 `min_words=200`／`0` 的 selected-indices
digest 與分數完全相同，符合 baseline 明確不套用 selector floor 的 frozen contract；
Greedy 則由 no-floor 的平均 200.45 words、macro ROUGE `0.279831`，變為 floor 版本的
242.60 words、`0.310353`。所以 A1 的 dev 暫時由 legacy 200–250 勝出，不支持「只要
把 cap 對齊 reference，floor 病因自然消失」的原推測。

**第二個現象**：floor-bearing Greedy 有 4/3,935 rows 回報
`selector_min_words_shortfall`。四列的 `source_capacity_words=250`、
`candidate_capacity_words=250`、`effective_min_words=200`，但實際只選 193、195、185、
182 words；這不是來源本身不可達，而是 selector trajectory 沒找到已知存在的可行 subset。
F-17 policy 正確保留完整 prediction row 並在 all-row denominator 計分，沒有補句或刪列；
但「誠實記錄 infeasible」不等於 selector 已滿足 length contract。

**一次性 dev-test 決選（1,686 rows，2026-08-08）**：四候選依預註冊各觀察一次；
200–250 的 cross-method macro 為 `0.310382`，相對 max-only250、median220、p75-cap260
的 paired mean difference 分別為 `+0.009185/+0.015510/+0.008022`。10,000 次 paired
bootstrap 的三個 95% CI 全為正，Holm family size 3，adjusted `p=0.000600`，故依原規則
凍結 200–250 words。Greedy 仍有 3/1,686 `selector_min_words_shortfall`，同樣保留完整列
與 all-row score；這確認 floor 是 selector stopping/feasibility mechanism，論文必須報
infeasibility rate，不能只寫成公平的輸出長度上限。

**證據**：dev 與唯一一次 dev-test 分別位於
`runs_v2/a1_length_contract/multinews/dev/`、`.../dev-test/`；每個 partition 都有 12/12
method evidence、同一 input/partition identity、逐篇 selected-indices digest、dependency
versions，search registry 各四筆，`test_split_accessed=false`。dev-test 不得重跑。

**跨資料集補證**：GovReport dev 681 rows 的 no-floor Greedy 只輸出 165.6–184.5
words，macro `0.175990–0.178859`；floor500-cap650 將它推至 638.4 words、
`0.370385`。同一 stopping dependency 因 reference／budget 較長而被放大；所以若 A1
最後保留 floor，必須承認它是方法行為的一部分，而非單純公平比較的外部上限。

---

### 🟡 F-24. GovReport dev 上，floor 修掉早停後的 lexical Greedy 仍輸給 Lead 與 Random

**發現（A1 cheap-method scope）**：在 GovReport frozen dev 681 rows、500–650 words 下，
lexical-only length-normalized Greedy 的 R-1/R-2/R-Lsum 為
`0.497973/0.151940/0.461242`，macro `0.370385`；同長度協定的 Lead 為
`0.514532/0.194032/0.489133`（macro `0.399232`），Random seed 3407 為
`0.538113/0.181603/0.506816`（macro `0.408844`）。681/681 rows 均有輸出且 feasible，
不是 denominator 或失敗列造成的差距。

**解讀邊界**：這個 A1 Greedy 刻意只啟用 lexical route，用來便宜選 length contract；
它不是 final semantic+graph system。因此不能寫成「整個方法已被 Random 擊敗」。但它已
排除「只要換對長度，現行 lexical objective 就自然變強」：floor 修正 stopping 後，選句
品質仍不足。後續 sensitivity scan 必須把 semantic/graph 的增量與 selector choice 分開，
而不是把它們一次堆上後只報總分。

**證據**：`runs_v2/a1_length_contract/govreport/dev/`，12/12 method evidence 完成、
同一 input/partition digest、681 rows、test 未存取。此 finding 的狀態是 open reality
warning；只有後續 route ablation／strong baseline matrix 能解除或升級為重新定位理由。

**一次性 dev-test 補證（292 rows，2026-08-08）**：500–650 仍以 cross-method macro
`0.393415` 勝出；相對三個 no-floor caps 的 paired difference 為 `+0.060642` 至
`+0.067218`，95% CI 全為正，三個 Holm-adjusted `p=0.000600`，292/292 feasible。
它的 dev→dev-test 差只有 `+0.000595`，未見 development overfit。可是同一 frozen
contract 下 lexical Greedy 的 R-1/R-2/R-Lsum 為
`0.503256/0.154923/0.466966`（macro `0.375048`），仍低於 Lead macro `0.394815` 與
Random `0.410382`。因此 F-24 不但未解除，還由一次性 holdout 確認；範圍仍限 A1
lexical-only cheap method，不提前否定 semantic/graph。證據位於
`runs_v2/a1_length_contract/govreport/dev-test/`，12/12 evidence，test 未存取且不得重跑。

---

### ✅ F-25. `features.position` 以攤平後全域 index 計分，Multi-News 文件邊界不會重置

**發現（2026-08-08，D1 分數前）**：`build_base_scores()` 只接收 `sentences`，v1/v2
position 都用 `enumerate(sentences)`。canonical Multi-News 雖保存 `document_id` 與
`document_position`，這條 feature path 完全沒讀；第二篇文件的第一句因此被當成前一篇
之後的中段句，而不是新文件的 lead。若直接把目前為 0 的 position weight 打開，測到的
不是論文宣稱的 document-aware weak prior。

**處理**：D1 preregistration 已在任何新 score 前綁定 correctness precondition：新增
`features.position.scope=document`，必須由 canonical sentence records 計算並在每個
document 重置；請求 document scope 卻沒有 records 時 fail loud。以兩篇 toy document
的 golden test 證明兩個 document starts 都為 1.0 後，才允許執行 position OFAT。

**重現**：在修正前對 document sizes 2+2 呼叫 v1 position，輸出是全域
`[1, 2/3, 1/3, 0]`；正確 document-linear 應為 `[1, 0, 1, 0]`。test split 未存取。

**修正與驗證**：`document_position_scores()` 現在驗證 canonical document ID 與每篇
0..n−1 position，v1/v2 都逐 document 計算；`build_base_scores()` 只有明確
`scope=document` 才使用它，既有 global configs 保持不變。position candidate route 亦
共用此實作。兩文件 v1/v2 golden、缺 provenance／不連續位置 fail-loud、10-document
snapshot 與當時完整 **338 tests passed**（2026-08-08）；加入 D1 governed runner 後為
**343 tests passed**。D1 Multi-News lexical family 完成與 resume/diagnostics regression
後為 **345 tests passed**；該 family 的分數見 F-29。

---

### 🟡 F-26. A1 單一路由使三個 candidate-budget/RRF 宣告值結構上不活躍

**發現**：A1 base 只啟用 lexical route，`route_top_k=40`、`min_per_route=20`、
`total=60`。proposal union 最多 40，因此 total 60 永遠不裁切；單一路由的 reservation
也不改 top-40 membership；`1/(c+rank)` 對 rank 單調，所以 RRF constant 不改排序。
這些值有被程式讀取，卻在該 context 無法改變輸出。

**影響邊界**：A1 問的是 length contract，這不推翻其已凍結結果；但不能用 A1 run
宣稱 candidate budget 已驗證，也不能做「動參數後無差」的假敏感度結論。D1 已預註冊
在 lexical+graph 兩路、40/20/60 base 下分別移動 route top-K、reservation、total 與
RRF constant。完整盤點在 `docs/research/evidence/d1_effective_tunable_inventory.json`。

---

### ✅ F-27. D1 family 只在最後寫 search log，外部 timeout 會留下不可恢復的半套研究

**發現（2026-08-08）**：原 `run_greedy_sensitivity.py` 連續跑完整 family，等所有
candidate 結束後才一次寫 `search_log.jsonl`。Multi-News lexical family 執行到第 12
個 L11 時，外層 job 達 60-minute hard timeout；L00–L10 已有完整 evidence，卻沒有任何
search-log record，L11 只剩 atomic `.partial`，而 runner 因 output root 已存在而拒絕重開。

**風險**：完成結果可能因 orchestrator timeout 在 registry 中「不存在」；人工刪掉
output 後重跑又會消滅失敗成本與形成第二次不受控觀察。這違反「失敗也要記錄」與每階段
可交接要求，雖然本次仍只有 frozen dev，沒有碰 dev-test/test。

**修正**：每個 candidate 完成即以 logical hash 去重寫 log；新增 `--resume`，逐一驗證
resolved config、candidate hash 與 evidence。完整 candidate 只載入，不重跑；不完整
`run` 搬到 `greedy/attempts/attempt_NN_interrupted/`，寫 interruption evidence 與 failed
search-log row，再只重跑缺少者。原 L11 failure 與成功重試都保留，最終是 12 final
success + 1 interruption failure。

**驗收**：partial preservation／evidence tests 與完整 regression **345 passed**；
`c1b2662` 版本化恢復行為。test 與 dev-test 均未存取。

---

### ✅ F-28. `candidates.use=false` 時 diagnostics 把全文 selector pool 誤報成 0

**發現（2026-08-08，分數後、解讀前）**：舊 `_candidate_diagnostics()` 只讀
`candidate_pool.actual_size`。關閉 candidate builder 時 provenance records 按 contract
為空、該欄為 0，但 `selector_inputs.candidate_count` 實際是全文；因此 L10 初始 summary
把最昂貴的全文搜尋誤報為 candidate size 0。ROUGE、selected indices 與 selection
artifact 沒受影響，錯的是 audit 層的成本欄。

**修正**：diagnostics schema v2 以 `selector_inputs.candidate_count` 報 actual selector
mean/p95/max，另保留 `provenance_candidate_size_mean`；無 route records 時 agreement
改為 `null`（not applicable），不再假裝 0 agreement。完成 artifacts 以 `--resume`
重新讀 predictions 刷新 diagnostics，不重跑選句或評分。

**驗收**：L00 actual/provenance mean 都是 `35.9535`；L10 actual mean `81.8513`、p95
`210.3`、max `3,318`，provenance mean 才是 0。新增回歸測試釘住 80-sentence full
source／0 provenance 的案例；完整 **345 tests passed**。

---

### 🟡 F-29. Multi-News top-40 lexical prefilter 是品質瓶頸，但全文搜尋不是可接受解法

**證據（D1 frozen dev 3,935 rows，2026-08-08）**：在同一 200–250-word contract 與
Greedy selector 下，L00 base macro `0.310353`；唯一關掉 candidate prefilter 的 L10
為 `0.320912`，三項 `0.431657/0.135168/0.395911`，相對 base macro `+0.010559`，是
12 個 lexical/objective OFAT 中最大正增益。其次為 coverage weight 加倍
`+0.006216` 與 document-aware position `+0.002568`。

**限制**：同協定 Lead macro 是 `0.326291`，L10 仍低 `0.005379`，特別是 R-2
`0.135168 < 0.148139`。L10 的 selector pool 平均由 `35.95` 增至 `81.85`、最大
`3,318`，wall time `1,138.5 s` vs base `234.3 s`（約 `4.86×`）。這一 family 尚未做
paired bootstrap，也尚未跑 GovReport，不能宣稱顯著或跨資料集成立。

**決策**：保留「候選召回不足」為優先病因，但不採全文 Greedy 作 final architecture。
依預註冊順序繼續 graph／semantic 與兩路 budget family，判斷能否用受控候選池回收品質；
正式 greedy-reference recall@K 完成前不計 headroom capture。完整狀態見
`D1_SENSITIVITY_STATUS.md`。

---

### ✅ F-30. Greedy 對每個候選重算兩次完整 coverage，GovReport 全文變體無法合理擴展

> **數值更正（F-31）**：下段「第二個 row／4.05 CPU hours」是依 Windows 開啟中
> partial size=0 所作的錯誤推論，已作廢；正確是 152-row prefix／約 0.766 CPU hours。
> 保留原敘述是為了讓錯誤推論可稽核，不得引用其數值。

**發現（2026-08-08，GovReport frozen dev、未看分數）**：L10 關閉 candidate
prefilter 後，舊 `greedy_select()` 對每個 remaining candidate 先呼叫 `can_add()`，再呼叫
`evaluate()` 取 utility；兩者都重建 `coverage_matrix[:, selected + candidate]`。在第二個
ordered dev row（2,192 句）已消耗至少 `639.47` CPU seconds，atomic predictions 尚未
flush。以舊實作的句數平方 proxy 對 681-row dev 外推，下限約 `4.05 CPU hours`；這不是
分數，也沒有讀 reference、dev-test 或 test。量測腳本與 evidence 分別是
`scripts/audit/greedy_scaling_projection.py` 與
`docs/research/evidence/f30_greedy_scaling_projection.json`。

**修正**：`SelectionObjective.evaluate_additions()` 現在一次計算 selected subset 的
row-wise coverage maxima，再逐候選精確形成完整 `SelectionEvaluation`；salience、
redundancy、constraints、停止條件與 lower-bound 行為未改。Greedy 改用已計算的 extension
判斷 upper bounds，不再透過 `can_add()` 對同一 extension 重算一次。

**驗收門檻**：單元測試把 batched extension 與原 `evaluate(selected+[candidate])` 在
3 種 importance aggregation、3 種 coverage method、rectangular full-source coverage
及負相似值逐欄精確對照；另以 40 組 deterministic random problems 對照 pre-F-30 完整
Greedy loop 的 selected indices；single-sentence structural guard 另有 pipeline regression。
targeted **43 passed**、完整 **357 passed**。固定的 post-F-30 reference-blind audit
另在真實 GovReport L00 pipeline 重跑全部 681 frozen-dev rows：逐篇 selected indices
**0 差異**，pre/post digest 同為
`8273f16296008019ccd566240ae868e46d841dcb1e04cfa4435b8fc72b4d7982`；沒有評估
references，也沒有讀 dev-test/test。evidence：
`runs_v2/f30_greedy_equivalence/govreport_l00_post_f30/evidence.json`。被中止的 attempt
由 `--resume` 封存，沒有刪除。

---

### ✅ F-31. Windows 開啟中的 atomic partial size=0 不能當作 row progress

**發現與更正（2026-08-08）**：F-30 觀察程序運行中 temp file size=0，誤判只完成
前兩個 ordered rows，並以第二列 2,192 句作平方 anchor。程序關閉後同一 partial 為
`7,485,234` bytes、152 個完整 JSONL rows，IDs 精確等於 frozen dev ordered prefix，
最後一列是 `validation_crs_R44670`。因此原 `4.05 h` 外推無效。

**正確重算**：152-row prefix 占全 dev `sum(N_sentences²)` 的 `0.231827`；以已觀察
`639.47 CPU s` 校準，舊實作全量 proxy 約 `0.766 CPU h`。修正版 L10 實測 selection
`817.84 s`；相對 proxy 約 `3.37×`，但分母不是完整實測 run，只能稱 projected speedup。
`greedy_scaling_projection.py` 現直接讀 archived partial、驗證 ordered prefix，再計算
比例；evidence 同時保留 invalidated prior interpretation。

---

### 🟡 F-32. GovReport top-40 lexical candidate pool 丟失大量可用內容，全文 dev 首次超過便宜 baseline

**證據（D1 frozen dev 681 rows，2026-08-08）**：L00 top-40 macro `0.370385`；L10
全文 macro `0.415585`，增加 `+0.045200`，R-1/R-2/R-Lsum 為
`0.541583/0.190097/0.515075`。同協定 A1 Lead／Random macro 是
`0.399232/0.408844`；L10 point estimate 分別高 `+0.016352/+0.006741`。L10 對
Random 三項都高；對 Lead 仍在 R-2 低 `0.003935`。

**限制與決策**：這是 12-config lexical family 的 dev screen，尚無 paired bootstrap、
多重比較校正或強 baseline。全文 pool mean/p95/max `318.52/698/2,889`，selection
`817.84 s`，不能作 final architecture。證據將「candidate recall」提升為 GovReport
第一優先病因；下一步 graph／semantic 與 candidate-budget family 必須用受控 pool
回收全文增益。未完成 PacSum、SBERT+MMR、greedy reference 與一次性 dev-test 前，
不得宣稱方法勝出。

---

### 🟡 F-33. Multi-News sparse graph 能以受控候選池回收 lexical recall，但仍未補足 R-2

**證據（D1 frozen dev 3,935 rows，2026-08-08）**：純 lexical L00 macro
`0.310353`；lexical+graph G00 `0.323407`（`+0.013054`）；把 route top-K 40 改為 80
的 G02 為 `0.324305`（對 L00 `+0.013952`）。G02 selector pool mean/max 僅
`48.03/60`，卻高於全文 lexical L10 `0.320912`，因此不是靠把全文直接交給 selector
才得到增益。G02 的 R-1/R-2/R-Lsum 是 `0.436696/0.137103/0.399115`；同協定 Lead
是 `0.435033/0.148139/0.395701`，macro 仍低 `0.001986`，差距集中在 R-2。

**架構判斷**：ARCHITECTURE §5.4 的 graph 刪除條件目前**未觸發**；Multi-News dev
已有實質正貢獻，graph 應保留至 GovReport 同 family 與 paired inference 完成。相反，
G06 membership-only 比 G00 低 `0.004023`，證明 route-aware salience 不能退回只決定
candidate membership。G07 soft/full pool + membership-only 的 macro 更低（`0.320997`）、
pool 最大 3,318，品質與成本都不支持保留。這仍是 27-config screen 的其中一格，沒有
dev-test、強 baseline 或顯著性，不得寫成 graph 已勝出。

**證據位置**：
`runs_v2/d1_greedy_sensitivity/multinews/dev/cheap_multiroute/study_summary.json`；
12 個 final runs 均為 3,935 rows，另保留 G00／G01 external interruption attempts；
summary 明示 `dev_test_accessed=false`、`test_split_accessed=false`。

---

### 🟡 F-34. GovReport uncapped section guard 與固定候選總額結構上不相容

**重現（2026-08-08，GovReport frozen dev 第一列）**：G11 同時要求兩 route 各保留
20 個 evidence、每個 section 再保留一個 guard。第一列形成 77 個 mandatory
reservations，但 `candidate_budget.total=60`，因此
`candidate_builder.py` 正確拋出 `candidate total cap 60 cannot fit 77 mandatory
route/guard reservations`。這不是不可行摘要列，不能依 F-17 降低限制或跳列。

**處理**：原 G11 保持 failed，不修改、不覆寫。分數前的 tunable inventory 已明定
`coverage_guard.max_items`「only if guards overflow cap」，所以另立
`d1_govreport_section_guard_followup_v1.json`，固定 `max_items=20`；20 是由 total 60
減去兩 route 最壞情況各 20 個 disjoint reservations 得到，不用 ROUGE 選值。follow-up
只作 feasibility/diagnostic，總比較數改記 28；執行前已版本化 runner，且沒有 split CLI。
prereg commit `3b6813e` 後的實測為 681/681 feasible、candidate pool max 60、macro
`0.404182`（對 G00 `+0.000685`）；dev-test/test 均未讀。這證明 cap-aware guard 可行，
但增益很小，不能由此單獨 promotion。

---

### 🟡 F-35. GovReport hard-pool graph 有增益，但受控候選召回仍不足；全文 soft pool 成本過高

**證據（D1 frozen dev 681 rows，2026-08-08）**：G00 lexical+graph macro `0.403496`，
G02 route top-K 80 為 `0.404282`；相對純 lexical L00 分別
`+0.033111/+0.033897`。G00 也高於 TF-IDF centroid 第二路 G01 `0.003075`，因此 graph
route 目前不符合刪除條件。G02 pool mean/max `59.94/60`，高於 Lead macro
`0.399232`，但低於 Random `0.408844`。

G07 soft/full pool macro `0.414831` 高於便宜 baseline，卻仍低於全文 lexical L10
`0.415585`；selection `804.13 s` 對 G00 `60.53 s` 約 `13.28×`，pool max 2,889。
因此它沒有提供 graph 的全文增量，且不符合最終成本目標。G06 membership-only 在
GovReport 比 G00 低 `0.002624`，與 Multi-News `−0.004023` 同方向，跨資料集支持刪除
membership-only 設計、保留 route-aware salience。仍未做 semantic、強 baseline、
paired inference 或 dev-test，不得宣稱 graph 方法勝出。

---

### 🟡 F-36. 三路各保留 20 個候選與 total 60/coverage guard 結構上衝突

**重現（2026-08-08，Multi-News frozen dev）**：S02 啟用 lexical/semantic/graph，
但沿用 `min_per_route=20`、`total=60` 與 document guard；前三列後遇到 61 個 mandatory
reservations，正確 fail loud。三路最低保留已把 60 全部用完，所以任何不與 route
reservation 重疊的 guard 都會失敗。原 S02 保持 failed。

**處理**：在 GovReport semantic 分數前，以系統時間另立兩-primary preregistration；
S02b 固定 total 80、guard max 20，容量公式為 `3×20+20=80`，不依 ROUGE 選值。
runner 只有 dataset 參數，固定 validation-dev；比較總數改記 29。兩資料集必須各自
在 prereg commit 後執行，原 S02 failure 不得覆寫。

**後續狀態**：Multi-News S02b 已在 prereg commit `73769c5` 後完成 3,935 rows；
3,930 feasible、pool max 80、dev-test/test 未讀。原 S02 failure 不被覆寫。

---

### 🟡 F-37. Multi-News semantic 有 unique candidates，但目前不勝 graph，且 CPU 成本約十倍

**證據（D1 frozen dev 3,935 rows，2026-08-08）**：S00 lexical+semantic macro
`0.322404`，相對純 lexical L00 `+0.012051`，但低 graph G00 `0.001003`、graph G02
`0.001900` 與 Lead `0.003886`。semantic route 平均有 10.48 unique candidates、3.68
unique selected sentences，所以不是完全重複；然而 selection `1,387.23 s`，約 graph
G00 `133.75 s` 的 `10.37×`。S01 直接採 semantic raw salience + SBERT similarity 為
`0.299634`，比 S00 低 `0.022771`，該 selector 接法不保留。

**決策邊界**：S00 尚不能單靠一個資料集觸發整條 semantic route 刪除，因 unique
contribution 存在，且兩-primary S02b 雖為正訊號卻不是 semantic 純 ablation；
但它已排除「直接 semantic selector」並把 semantic 的舉證責任提高為跨資料集 quality
gain 或明確 adaptive-routing 子群增益。
沒有 paired inference、strong baselines 或 dev-test，不得 promotion。

---

### 🟡 F-38. 三路 capacity-correct 配置首次在 Multi-News dev 點估計高於 Lead，但尚不能歸因 semantic

**證據（frozen dev 3,935 rows，2026-08-08）**：預註冊 S02b（lexical + semantic +
graph、total 80、guard cap 20）macro `0.328077`，R-1/R-2/R-Lsum 為
`0.440759/0.139803/0.403670`；相對 graph G02 `+0.003772`、相對 Lead macro
`+0.001786`。候選池 mean/max `52.09/80`，三路 unique selected means 分別為 graph
`2.23`、lexical `1.32`、semantic `1.78`。3,930/3,935 feasible；5 列沿既定政策記錄
infeasible，未放寬 fail-loud。

**限制與決策**：S02b 為修正結構容量的整體 follow-up，同時改變 total cap 與 guard
cap；它不是 capacity-matched 的 semantic-only ablation，不能把全部提升歸因於 semantic。
而且 R-2 仍低 Lead `0.008336`。保留為候選配置，等待 GovReport 複現、strong baseline
與 paired bootstrap；在這些完成前不晉級、不讀 dev-test。

**重現**：`python -m scripts.audit.run_d1_three_route_followup --dataset multinews`；主證據在
`runs_v2/d1_three_route_followup/multinews/dev/S02b_three_route_capacity_80/`，preregistration
SHA-256 `2cac40b58c2d669ab0a2e5088f25afafbaaaa5c332460dad62e7bb9564e92743`，
selected-indices SHA-256 `6ad798fc2b71fd5810e8d6ae1611d3ce668624d4c2c2d1844d940d6f15039d6c`。

---

### 🟡 F-39. GovReport semantic route 有獨立訊號但成本過高；semantic-direct selector 跨資料集刪除

**證據（GovReport frozen dev 681 rows，2026-08-08）**：S00 lexical+semantic macro
`0.407203`，R-1/R-2/R-Lsum `0.534512/0.187499/0.499599`。它相對 graph G02
`+0.002921`，但仍低 Random `0.001641`、低全文 lexical L10 `0.008382`；selection
`917.18 s`，約 graph G00 `60.53 s` 的 `15.15×`。semantic 平均有 26.17 unique
candidates、8.43 unique selected，證明 route 不是空包裝，但現有受控池的品質／成本
不足以支持預設全開。

S01 semantic raw salience + SBERT similarity macro `0.349832`，比 S00 低 `0.057371`；
Multi-News 同接法也低 `0.022771`。因此依 §5.3 刪除的是這個 selector 接法，不再納入
搜尋；整條 semantic candidate route 因兩-primary S02b 正訊號暫留，但仍等待
capacity-matched route ablation 與 paired evidence。

**重現**：離線 pinned model 下執行
`python -m scripts.audit.run_greedy_sensitivity --dataset govreport --family semantic_route`；
主證據在 `runs_v2/d1_greedy_sensitivity/govreport/dev/semantic_route/`。family 為
2 success + 1 structural failure；成功列共 1,362，dev-test/test 未讀。

---

### 🟡 F-40. Capacity-correct 三路配置在兩 primary 都是目前 proposed 最高點估計，但成本與歸因仍未過關

**證據（2026-08-08）**：GovReport S02b macro `0.417862`，R-1/R-2/R-Lsum
`0.545091/0.196227/0.512267`，681/681 feasible、pool mean/max `78.74/80`。它高 S00
`0.010659`、全文 lexical L10 `0.002277`、graph G07 `0.003031` 與 Random `0.009018`；
Multi-News S02b 也以 `0.328077` 高於該資料集目前 proposed 配置與 Lead 點估計。因此
三路 capacity-correct design 已有跨資料集一致正向 point estimate，不應在 paired
analysis 前直接刪除 graph 或 semantic route。

**限制**：GovReport selection `1,106.27 s`，約 graph G00 的 `18.28×`；Multi-News 亦
明顯昂貴。S02b 同時改 total cap 與 guard cap，仍不是單一路徑的 capacity-matched
ablation。它尚未對 PacSum／SBERT+MMR 等 strong baselines，也沒有 paired bootstrap／
多重比較校正，因此不構成 promotion 或 dev-test 授權。

**重現**：`python -m scripts.audit.run_d1_three_route_followup --dataset govreport`；主證據在
`runs_v2/d1_three_route_followup/govreport/dev/S02b_three_route_capacity_80/`，
selected-indices SHA-256 `62987c3698e7adb0a0ba7d5d37c8795651347ffceb8909a4b9f8b5359a7f20d0`。

---

### 🟡 F-41. Multi-News capacity-matched ablation 顯示 semantic 與 graph 皆有正增量，尚待 paired 驗證

**證據（frozen dev 3,935 rows，2026-08-08）**：固定 S02b 的 total 80／guard cap 20，
A01 移除 semantic 得 macro `0.324209`，A02 移除 graph 得 `0.322719`；完整 S02b
`0.328077` 分別高 `0.003868`／`0.005358`。兩個 ablation 各 3,930 feasible + 5
既定 infeasible、pool max 80，config audit 證實除 `enabled_routes` 與 study metadata 外
沒有差異。A01 又與既有 G04 aggregate 完全一致，排除 guard cap 在此資料集造成分數差。

**成本**：A01 `330.05 s`，A02 `1,646.88 s`，S02b `1,467.99 s`。這支持 semantic
路徑很昂貴，但 A02/S02b 是不同 process，不能把約 12% wall-time 差直接歸因 graph。

**決策**：point estimate 尚未滿足保留 gate；先完成 GovReport 同 ablation，再以逐篇
paired bootstrap 與包含 31 configs 的多重比較治理判斷。此時不讀 dev-test。

**重現**：`python -m scripts.audit.run_d1_capacity_matched_route_ablation --dataset multinews`；
主證據在 `runs_v2/d1_capacity_matched_route_ablation/multinews/dev/`。

---

### 🟡 F-42. Capacity-matched route 增量在兩 primary 同號；semantic 昂貴、graph 邊際成本小

**證據（GovReport frozen dev 681 rows，2026-08-08）**：A01 移除 semantic macro
`0.403594`，A02 移除 graph `0.406381`；完整 S02b `0.417862` 分別高
`0.014268/0.011481`。兩個 ablation 均 681/681 feasible、pool max 80，config audit
證實只改 `enabled_routes`。A01 與既有 G04 aggregate 完全一致，因此 guard cap 不構成
分數混淆。連同 Multi-News 的 `+0.003868/+0.005358`，semantic 與 graph 的
capacity-matched point estimate 在兩 primary 都同號為正。

**成本與決策**：GovReport A01 `163.77 s`、A02 `1,093.27 s`、S02b `1,106.27 s`。
semantic 是主要額外成本；sparse graph 在 semantic 已開啟時成本很小。兩 route 暫不依
§5.3/§5.4 刪除，但 semantic 是否 always-on 必須由 paired quality gain 與 adaptive
quality-cost rule 決定。尚未做 31-config multiplicity-corrected paired analysis，故不晉級。

**重現**：`python -m scripts.audit.run_d1_capacity_matched_route_ablation --dataset govreport`；
主證據在 `runs_v2/d1_capacity_matched_route_ablation/govreport/dev/`。

---

### ✅ F-43. Semantic 與 graph 的 capacity-matched 增量通過跨資料集 paired／selection correction

**證據（2026-08-08，完整 frozen dev 分母）**：四個 route comparisons × 三 ROUGE
metrics 共 12 endpoints，mean difference 與 95% CI 全為正。每項 raw p=`0.000200`、
12-test Holm p=`0.002400`；再按 31 configs × 2 datasets × 3 metrics = 186 個搜尋機會
Bonferroni 後為 `0.037196`，仍通過預註冊 strong-endpoint rule。Multi-News 的 semantic/
graph macro 邊際 `+0.003868/+0.005358`；GovReport `+0.014268/+0.011481`。

**決策**：§5.3／§5.4 的 semantic/graph 直接刪除條件目前未觸發；兩路保留到 selector
與 strong-baseline gate。這只證明 quality 增量，不代表 semantic 值得 always-on；其
GovReport 成本約使 A01→S02b 從 `163.77 s` 增至 `1,106.27 s`，仍需 adaptive cost rule。

**重現**：`python -m scripts.audit.run_d1_paired_analysis`；主 evidence
`runs_v2/d1_paired_analysis_v1/summary.json`，完整逐篇檔案以 SHA-256 記錄但不進 Git。

---

### 🟡 F-44. 10,000 bootstrap resamples 無法解析 372-opportunity baseline correction；不得事後改協定

有限樣本修正使 10,000 resamples 的最小雙尾 p 為 `2/10001=0.00019998`；乘上預註冊
372 個 baseline-selection opportunities 後，最小 corrected p=`0.074393`。因此即使
S02b 對 Random 的六個 endpoints 都有正 CI，0/12 cheap-baseline endpoints 能通過
selection-aware strong rule。這是解析度上限，不是「證明不顯著」。

**處理**：遵守「看到分數後不得改顯著性方法」，不事後把 resamples 增至可過門檻的值，
也不刪除 selection correction。cheap baseline 結果只作 interim；正式 Gate 2 以完整
強 baseline matrix 與下一個事前凍結的 confirmatory protocol 處理。另有實質負結果：
Multi-News 對 Lead 的 R-2 `−0.008336`，CI `[−0.010951,−0.005706]`；GovReport 對 Lead
R-2 的 CI 跨 0，故目前本來就不符合跨 metrics 勝出。

### 🟡 F-45. PacSum 上游 repo 與 checkpoint 無足夠授權／完整性 metadata，不能冒充原版重現

**重現**：作者公開 repo `https://github.com/mswellhao/PacSum` 的 HEAD 固定為
`67cc8ad370eac160ede997b7c32eb74907728bf8`；tree 內沒有 `LICENSE` 或 `NOTICE`。
README 提供 fine-tuned BERT 的 Google Drive 連結，但沒有 checkpoint licence、SHA-256 或版本化
model card，且宣告 runtime 為 Python 3.6、舊 PyTorch／gensim／pyrouge。完整檔案級稽核見
`docs/research/evidence/f45_pacsum_upstream_audit.json`。

**風險**：直接複製上游程式或 checkpoint 會同時造成授權、供應鏈完整性與 evaluator 不一致；
即使勉強跑通，也不能合理宣稱 bit-exact reproduction。

**處理**：不 vendoring 上游 code／weights。依 ACL 2019 論文公開方程 clean-room 實作
`pacsum_tfidf` 與 `pacsum_sbert`，共同使用 frozen canonical sentences、共享 length/evaluator
contract、確定性 sentence-order tie-break，artifact 強制標記
`clean_room_protocol_adaptation`。前者分離 centrality 本身，後者測試同一 pinned MiniLM
representation 下的 directed-centrality 對照；論文不得簡寫成「官方 PacSum checkpoint 重現」。

### 🟡 F-46. PacSum 上游以未綁 seed 的 shuffle 打散排序 ties，不能作跨機器確定性依據

**重現**：固定上游 commit `67cc8ad` 的 `code/extractor.py::_select_tops` 在依 centrality
排序前呼叫 `random.shuffle(paired_scores)`，README／CLI 沒有 tie-break seed 或 per-row seed
contract。當 threshold 使多句同分（β 高或短文件很常見）時，selected indices 取決於未記錄的
process RNG state；artifact SHA 或單次 ROUGE 不能證明重現。

**處理**：本專案不複製該行為；clean-room PacSum adaptations 明訂以 frozen canonical sentence
order 作 stable tie-break，並在每列 `baseline_diagnostics.tie_break` 記錄。這是 protocol adaptation，
不是官方 stochastic tie policy 的重現；後續比較不替 PacSum 額外加 seed 平均，避免把未預註冊的
隨機性帶進 50-candidate Gate 2 搜尋。

### 🟡 F-47. PacSum beta=1 端點完全退化，較高分數來自 selection contract 而非 centrality

**重現**：執行 `python -m scripts.audit.summarize_gate2_baseline_family --dataset multinews
--family non_plm`。Multi-News frozen dev 的 `pacsum_tfidf_B10_beta_1.0` 在 3,935/3,935 rows
皆為 `score_degenerate=true`；beta=1 把 threshold 推到矩陣最大值，所有非 self edges 都被清空，
stable tie-break 退化成 canonical sentence order。它與 frozen Lead selected indices 完全相同的
比例只有 `17.41%`，平均 Jaccard `0.860104`，差異來自 rank-based baseline 遇到不合 cap 的句子
會繼續往後填，而非 directed centrality。

**處理**：beta=1 保留為預註冊 endpoint 與 canonical-order skip-tolerant control，不刪除分數，
但不得用來主張 PacSum centrality 有效。non-PLM dev winner 使用非退化 P08
（previous −0.8/following +0.2）：R1/R2/Lsum `0.442862/0.150064/0.402293`、macro
`0.331740`；其 1/3,935 退化 row 也完整記錄。完整證據見 Gate 2 `analysis_summary.json`。

### ✅ F-48. Gate 2 resume 原先封存 interrupted artifact，卻漏寫全域 search log

**重現**：第一次 P07 run 被外層 60-minute 限制中斷；`--resume` 正確把 partial run 移到
`attempt_01_interrupted/` 並寫 `interruption_evidence.json`，但舊 `_archive_partial()` 沒有呼叫
`_append_search_log()`，因此全域 registry 只看得到 final success，違反「失敗也要記錄」。

**處理**：`_archive_partial()` 現在回傳 evidence，resume 立即以 attempt directory 名稱作
去重鍵寫入 failed search-log row，保存 failure type、config/candidate hash、archive path 與
dev-test/test guards；既有 P07 failure 已回填。focused tests 9/9、完整回歸 375 passed。

### ✅ F-49. Gate 2 family verifier 曾把 GovReport A1 Lead 錯接到 Multi-News length path

**重現**：首次執行 `summarize_gate2_baseline_family --dataset govreport --family non_plm`
在 `runs_v2/a1_length_contract/govreport/dev/legacy_floor_200_cap_250/` 失敗；GovReport 的
實際 frozen contract 路徑是 `dev_iqr_band_500_650/lead/run`。正式 baseline family 不受影響，
但 verifier 因找不到 Lead artifact 無法完成 integrity comparison。

**處理**：dataset→A1 Lead 路徑改為各自 frozen policy，新增兩路 path regression；focused
10/10、完整回歸 **376 passed**。修正後 verifier 才產生 GovReport `analysis_summary.json`；
不得以手動填入 Lead 分數跳過逐篇 ID／selection comparison。

### 🔴 F-50. 兩 primary 的 proposed S02b 都輸 strongest completed non-PLM baseline

**證據（frozen dev，2026-08-09）**：Multi-News 的非退化 PacSum TF-IDF P08 macro
`0.331740`，高 S02b `0.003663`。GovReport LexRank macro `0.451620`，高 S02b
`0.033758`；TextRank `0.430966` 亦高 `0.013104`。GovReport LexRank 對 frozen Lead
高 `0.052388`，逐篇 selected indices exact match 0/681、mean Jaccard `0.059722`，不是
Lead bias 假象。GovReport S02b 僅略高最佳 PacSum P07 `0.000695`。

**成本（單 process wall time，只作同機方向性比較）**：TextRank `547.30 s`、LexRank
`768.11 s`、P07 `19.50 s`，S02b `1,106.27 s`。因此 S02b 不只比 LexRank 低，還約慢
`1.44×`；相對 P07 約 `56.7×` 時間只換 `+0.000695` macro。尚未重複量測，不把比值
宣稱為穩定 speedup，但足以判定目前沒有 quality–cost 優勢。

**解讀與決策**：F-43 證明 semantic／graph 在固定 S02b contract 下有正邊際增量，但 F-50
證明該增量尚未轉成整體 competitive quality；「route 有用」不等於「整體架構勝 baseline」。
目前不觸發 test／dev-test，也不刪除既有 route evidence。先完成 PLM 與 greedy-reference
matrix，再依預註冊 dev search 優化 selector/salience；若搜尋空間耗盡仍無 paired 顯著優勢，
依停止條件寫重新定位建議，不得投稿或用單一 ROUGE endpoint 掩蓋。

---

## Part 2 — 對研究主計畫的實證補充

`paper_revision_plan_IEEE_Access.md` 是研究標準來源。以下列出 legacy 程式與 artifact 對其中幾條的補充；任何數字仍依 evidence status 判讀。

| 研究主計畫 | 原本的建議 | 稽核補充 |
|---|---|---|
| **P0-1** oracle | 「重新檢查 oracle 計算程式碼，可能有 bug」 | ✅ 已找到 0.136 的來源：對資料集 `rouge_scores` 取平均；它不是 oracle。正式值仍須用官方 files2rouge 重現 52.4 |
| **P0-2** 效率論述 | 「誠實拆分兩種變體」 | 程式可確認每篇重載；載入佔比重跑後由 78% 變 93%，**不可引用**。必須先修 code 再依正式 protocol 重測 |
| **P0-3** BERT/RoBERTa 1.5× | 「若差異真實存在，說明原因（BPE vs WordPiece）」 | 程式可確認每篇重載，足以使舊 protocol 無效；1.04× 可由 `scripts/audit/plm_timing.py` 重跑，但協定未鎖定，重測前不可歸因 tokenizer 或 checkpoint I/O |
| **P0-4** PLM 貢獻為零 | 「先當實作問題排查，檢查 w_plm 是否太小 / pooling 策略」 | ✅ 方向對，但**原因不是那兩個**。pooling 其實已經是 mean pooling（不是 CLS）。真正原因是 **Stage 2 根本沒有接 PLM**（F-3）。改用 Sentence-BERT 的建議仍然正確且必要 |
| **P1-2** ROUGE-Lsum | 「若是，改用 ROUGE-Lsum 重算」 | ✅ **完全正確，且效果比預期大**。實測 Multi-News R-L：0.201 → **0.386**。注意 pred/ref 分句必須一致（見 F-2 的警告） |
| **P2-1** mutation 1.0 語義 | 「per-individual 還是 per-gene 待確認」 | ✅ **已確認**：pymoo `BitflipMutation()` 的 `prob=1.0` 是 **per-individual**；per-gene 預設 `1/n_var`。實測 n_var=50 時每基因翻轉率 0.0204 ≈ 1/50，約 63% 的個體至少被改動一個位元。**R4 擔心的「整條染色體隨機化」不會發生**，論文照實寫即可 |
| **P2-1** pop_size / generations | 「補上 NSGA-II 設定」 | ⚠️ **狀態見 §0.0 狀態表**：config 裡的值從未被讀取，實際跑的一律是 100/100。**不要照 YAML 抄進論文** |
| **P2-2** 多次執行取平均 | 「必須報告 mean ± std over ≥5 runs」 | ✅ 同意。補充：目前跨 process 重跑是可重現的（實測三次相同），但靠的是全域 seed 的巧合，應把 seed 顯式接線 |
| **P1-1** baseline | 「必補 Lead-3、PacSum、SBERT centroid、LLM zero-shot」 | ✅ 同意。**補充**：稽核當時 repo 沒有 baseline（F-9）；現在兩 primary non-PLM 各 23/23、PLM 各 27/27 已完成，但 greedy reference 與完整 paired matrix 仍缺 |

### 稽核補充、且已收斂進主計畫／行動清單的項目

0. 🔴 **F-0（legacy Multi-News 未贏 Lead）** —— 同資料同內部 evaluator 下，ExpB 只在 R1 高 0.0021，R2/R-Lsum 較低；因 ExpB test-tuned，只能觸發 redesign，不能當新結果。
1. **F-3（Stage 2 無 PLM）** —— 方法章與實作不符，必須在任何新實驗前解決。
2. **F-5（相似度矩陣就地竄改）** —— 會靜默改變實驗語意；狀態見 §0.0 狀態表。
3. **F-9（legacy 無 baseline、目前矩陣仍不完整）** —— 舊表不能回溯當作本地重現；兩 primary non-PLM 各 23/23、PLM 各 27/27 已補，Gate 2 尚缺 greedy reference、paired significance 與正式兩-primary矩陣。
4. **F-12（分句品質）** —— 855 words 的「句子」會直接破壞長度控制，且系統對它有正向偏好。
5. **F-13(f)（靜默退回 greedy）** —— 需先確認沒有既有實驗其實跑的是 greedy。

---

## Part 3 — 程式碼重構計畫

你說「就算整個代碼重構也沒關係」。我的評估是：**不需要打掉重練，但需要一次有紀律的中型重構。**

理由：核心演算法（NSGA-II、GRASP、greedy、TextRank）本身是對的，問題出在**接線（wiring）、評測、與計時**這三層。全部重寫的風險（引入新 bug、無法對照舊結果）大於收益。

### 建議的重構順序

**R-0. 先建立回歸基準（動任何 code 之前）**
- 固定 seed，跑一個 200 篇的小集合，存下 `predictions.jsonl` 作為 golden file
- 安裝 pytest，讓 `tests/` 能跑
- 目的：後續每次修改都能確認「只改變了我想改的東西」

**R-1. 評測層（最高投報率，先做）**
- `src/eval/rouge.py`：改 `rougeLsum`，pred/ref 都以 `\n` 分句
- 新增 `src/eval/oracle.py`：greedy extractive reference（不是 exact oracle／upper bound）
- 新增 `src/eval/bertscore.py`：BERTScore（回應 R1、R2）
- 新增 multi-reference 支援；內部 scorer 以最大 R1 選定同一 reference，正式 SciTLDR 仍須官方 files2rouge wrapper
- 全部指標統一走同一個 entry point，杜絕設定漂移

**R-2. PLM 層（修正 F-3、F-4）**
- `encoder_rank.py`：模型改為**模組級快取**，只載入一次
  ```python
  @functools.lru_cache(maxsize=4)
  def _get_model(model_name: str, device: str): ...
  ```
- 新增 Sentence-BERT / SimCSE 後端（`all-MiniLM-L6-v2` 已在你的 cache 中）
- **讓 Stage 2 真的使用 PLM embedding** 計算 `sem_scores` 與 `sim`
- 把 `w_bert` 更名為 `w_plm`，並確保它真的加權 PLM 分數
- 計時改為分離報告：載入（一次性）/ 推論 / 選句

**R-3. 資料層（修正 F-8、F-12）**
- 分句改用 NLTK punkt 或 spaCy
- Multi-News 正確處理 `|||||` 與換行；修正編碼
- SciTLDR 保留 `target` 為 list，不要串接
- 加入資料健全性檢查（句長分布、異常句偵測）

**R-4. 接線與正確性（F-5、F-6、F-10、F-13f 的原始修法規劃；各項狀態見 §0.0 狀態表）**
- `graph.py` 加 `.copy()`
- `pop_size` / `n_gen` / `seed` 正確接線
- τ 一致地傳給 graph 特徵與候選池兩處
- 移除吞例外的 `except (ImportError, Exception)`，改為 fail loud
- config 新增 schema 驗證：**未被程式使用的鍵值直接報錯**（這能一勞永逸防止 F-6 再發生；報告型檢查已有 `scripts/audit/config_key_audit.py`（唯讀、不阻擋執行），**強制型驗證仍未實作**）

**R-5. Baseline 模組（新增 `src/baselines/`）**
- `lead.py`（Lead-3 / Lead-K）
- `textrank.py`、`lexrank.py`（自己實作，不抄別人數字）
- `pacsum.py`（unsupervised，同 regime 最公平的對手）
- `sbert_centroid.py`（取代 raw BERT，讓 PLM baseline 合理）
- `llm_zeroshot.py`（R1 明確要求；prompt 附 appendix）
- 全部走同一條 preprocessing 與 ROUGE 管線

**R-6. 實驗編排**
- 一個 `scripts/run_all.py`，用 seed list 跑多次、自動彙總 mean ± std
- 輸出機器可讀的 `results.json`，論文表格由腳本產生（杜絕手抄錯誤）
- 整理 GitHub repo：README、requirements（修正重複宣告）、一鍵重現腳本（P2-3）

**R-7. 不要動的部分**
- `nsga2.py` / `grasp.py` / `greedy.py` 的核心演算法邏輯（除了 F-7 的目標函數調整）
- `frontend/`、`backend/`、`experimental/` —— 與論文無關，投稿前可從公開 repo 中排除

---

## Part 4 — 實驗重跑清單

> ⚠️ **本節保留 2026-07-26 稽核當時的三資料集建議，不是目前執行清單。**
> 2026-07-30 的 v1 決策已由 `ACTION_PLAN.md` §2.0 取代：必跑 GovReport + 原版 Multi-News，
> Multi-News frozen U+FFFD clean 作 paired validation；CNN/DailyMail 延後可選，SciTLDR 不排程。

**以下矩陣只用來理解舊稿缺過哪些對照，不得據此自行啟動 CNN/DM 或 SciTLDR。**

### 必跑（缺一不可）

| 組別 | 系統 | CNN/DM | SciTLDR | Multi-News |
|---|---|---|---|---|
| Trivial | Lead-3 / Lead-K | ✅ **R4 明確點名的缺口** | ✅ | ✅ |
| Unsupervised | TextRank（自跑） | ✅ | ✅ | ✅ |
| Unsupervised | LexRank（自跑） | ✅ | ✅ | ✅ |
| Unsupervised | PacSum | ✅ | ✅ | ✅ |
| Zero-training PLM | SBERT centroid | ✅ | ✅ | ✅ |
| LLM | zero-shot 句子選擇 | ✅ | ✅ | ✅ |
| **本文** | 純 meta-heuristic 變體 | ✅ | ✅ | ✅ |
| **本文** | 完整 fusion 變體（三軌） | ✅ | ✅ | ✅ |
| 診斷 | **Exact oracle（可行時）／明確標示的 greedy reference** | ✅ | ✅ | ✅ |
| 參考（標註 supervised，非同組） | BERTSumExt / MatchSum（引用文獻值） | ✅ | — | ✅ |

每個資料集內的系統與 baseline 必須共用 evaluator 與輸出限制。多句資料報 R1/R2/Lsum；SciTLDR 依官方 files2rouge 報 R1/R2/RL。BERTScore 作補充，不取代 dataset official metric；確定性 baseline 不虛構多 seed 變異，隨機系統則報預註冊 seeds、paired CI 與適當校正。

### Legacy 診斷數字（不可直接引用為新論文結果）

| 資料集 | 系統 | R-1 | R-2 | R-Lsum | n |
|---|---|---|---|---|---|
| Multi-News | 論文當家配置（K=20 Coverage） | 0.4352 | 0.1405 | 0.3880 | 5622 |
| Multi-News | **Lead（245 whitespace words，同預算）** | 0.4331 | **0.1453** | **0.3901** | 5622 |
| Multi-News | **Lead（逐篇對齊系統長度）** | 0.4325 | **0.1449** | **0.3895** | 5622 |
| Multi-News | Lead-3（僅 3 句，長度嚴重不足） | 0.2934 | 0.0954 | 0.2588 | 300 |
| Multi-News | **Greedy reference（245 whitespace words）** | **0.5910** | **0.2836** | **0.5340** | 300 |
| SciTLDR-AIC | Lead-3 | 0.3577 | 0.1069 | 0.3014 | 618 |
| SciTLDR-AIC | **Legacy greedy reference（3句、串接 reference，非官方）** | **0.5136** | **0.1931** | **0.4146** | 618 |
| SciTLDR-AIC | *論文誤稱的 "oracle"* | *0.1376* | — | — | 618 |

> ⚠️ Multi-News 的 greedy reference 與 Lead-3 是前 300 篇；Lead 與系統是全部 5622 篇。前者原分析流程未完整版本化，即使擴到全集也只能稱 greedy reference，不能稱 upper bound。
>
> 📌 注意 Lead-3 在 Multi-News 只有 0.2934，而同預算 Lead 有 0.4331 —— **差距全來自長度**。
> 這說明比較 baseline 時**長度預算必須對齊**，否則結論會完全相反。論文抄來的那組數字很可能就有這個問題。

### 補充實驗

- **Ablation**（修好 PLM 之後重做）：完整 / −NSGA-II / −PLM / −Graph，三個資料集都要
- **τ 敏感度**：5–7 個值，折線圖（**F-10 狀態見 §0.0 狀態表**）
- **Centrality 比較**：PageRank vs degree vs betweenness vs eigenvector
- **Fusion 權重敏感度**：`w_base` / `w_plm` 掃描
- **NSGA-II formulation**：sum vs mean importance（F-7）
- **Quality–latency Pareto 圖**：所有系統畫在同一張圖（**修好 F-4 之後**）
- **Qualitative analysis**：2–3 個案例，展示三軌各自選到什麼

---

## Part 5 — 誠實的總評與風險

我必須直說幾件事，包括你可能不想聽的：

1. 🔴 **F-0 是這篇論文的存亡問題，不是修修補補的問題。** 現有的核心實證主張（在 Multi-News 上顯著勝過所有 extractive baseline）被同資料、同內部 evaluator 的 legacy 診斷否定。**在決定怎麼處理 F-0 之前，不要開始改寫論文**；Phase −1 已將它列為最前置 gate。

2. **F-3 是必須處理的，不能用改寫文字繞過。** 論文方法章描述了一個沒有被實作的融合機制。必須以 validation ablation 決定真的接上 PLM，或刪除相關方法與貢獻主張；不能預設接上後一定改善。

3. **加速倍數會縮水。** 修正計時後「3×–170×」大概率不再成立。但用一個建立在 bug 上的數字投 IEEE Access，風險遠大於誠實呈現 trade-off。

4. **這輪修改的淨效果：技術面正面，論述面必須大改。**

   | 面向 | 方向 |
   |---|---|
   | ROUGE-L 0.201 → 0.388（F-2） | ✅ 正面 |
   | 已定位 0.136 不是 oracle；official 52.4 尚待 conformance（F-1） | ⚠️ 部分完成 |
   | PLM 真的接上後 ablation 才有意義（F-3） | ✅ 正面 |
   | 加速宣稱縮水（F-4） | ⚠️ 負面，可用 Pareto 定位吸收 |
   | **贏不過 Lead（F-0）** | 🔴 **需要整篇重新定位** |

5. **關於投稿**：IEEE Access 是新投稿；真正需要揭露與實質擴充的是已發表的 ICACT conference paper。內部保留 ICT Express response matrix 追蹤問題，但不要把新稿寫成對舊期刊的 response letter。

6. **時間估計（修正後）**：與主計畫及行動清單統一為 **8–12 週**；實際取決於新資料集取得、完整 regression 與計算資源。

7. **最後一句實話**：這篇論文目前的技術貢獻，比它自己宣稱的要小。但它**不是沒有價值** —— 一個 zero-training、CPU 可行、能逼近強 baseline 且有完整 trade-off 分析的框架，在 IEEE Access 是可以發表的。前提是**論述必須誠實地縮到證據支持的範圍內**。硬撐「outperforms SOTA」這條路，我判斷會再被拒一次。

---

## 附錄 A：本次已直接修改的程式碼

以下是初次 audit patch 與目前狀態的對照。pytest 已安裝，目前
**453 local tests 全過且 PR #16 Linux CI 綠燈（2026-08-10）**；這只代表 correctness regression、10-document snapshot、內部
hand-calculated golden 與 Lead plumbing 受測，不代表方法效果或 published-protocol parity 已通過。
Sentence-BERT production route、canonical NLTK segmentation、shared objective/selector
contract 與 Lead／Random／TextRank／LexRank／SBERT centroid／MMR baseline 已接線；centrality offline hotfix 與
兩 primary frozen-dev non-PLM 各 23/23、PLM 各 27/27、greedy reference 6/6 與 paired
finalists 與 D2/D3a/D3b redesign 已完成。GovReport D3b 通過 strongest-baseline gate，
Multi-News 未通過，故雙 primary promotion 失敗；clean sensitivity 暫停，等待重新定位決策。

| 檔案 | 修改內容 | 對應發現 | 驗證 |
|---|---|---|---|
| `src/eval/rouge.py` | ROUGE-Lsum；同一 reference 由最大 R1 選定；長度 mismatch fail；保留 legacy evaluator | F-2, F-8 | ✅ 兩個 5622 篇 artifacts 已重現 0.3857／0.3880；✅ regression tests；⏳ official files2rouge conformance（僅保留 SciTLDR 時） |
| `src/eval/oracle.py` | metric-specific greedy reference；canonical fail-loud、source-order search、`max_words` 明確化 | F-1, F-21 | ✅ correctness suite；兩 primary partitioned Gate 2 runs 6/6 完成 |
| `src/features/graph.py` | thresholding 前先 `.copy()`，不再就地竄改呼叫端矩陣 | F-5 | ✅ 呼叫前後矩陣一致 |
| `src/models/extractive/encoder_rank.py` | 模型快取、pinned revision、完整輸入 batch encode、截斷與成本 artifact | F-4 | ✅ CPU 與 3-row canonical smoke；⏳ 正式 cold/warm/GPU cost pilot |
| `src/pipeline/optimizer_dispatch.py`、`src/objectives/evaluator.py` | `pop_size` / `n_gen` / `seed` 接線；移除 fallback；Greedy／GRASP／NSGA-II 共用 objective/constraints，保存 Pareto front | F-6, F-13f, F-3, F-7 | ✅ hand-computed、seed、no-fallback、pipeline regression；⏳ MMR/exact baseline 與 validation isolation |

**執行 greedy reference 的指令**（例：Multi-News，245 whitespace-word 預算）

```bash
python -m src.eval.oracle --input tests/fixtures/multi_news_validation_diagnostic_sample.jsonl --max_words 220 --limit 3
```

**按目前多句內部協定重算已有 run（不等同 published-protocol parity）**

```bash
python -m src.pipeline.evaluate --pred runs/full_benchmark_result/final_summary/predictions.jsonl --gold data/processed/multi_news_test.jsonl --out runs/full_benchmark_result/metrics_fixed.csv --protocol multisentence_lsum
```

> ⚠️ **修改後必做**：`optimizer_dispatch.py` 現在會 fail loud。請重跑一次既有的主要實驗設定，確認**沒有任何一組實驗其實是靠 greedy fallback 跑出來的**（見 F-13f）。若有，那組的論文數字必須作廢重跑。

---

## 附錄 B：稽核方法

所有結論均在 repo 的 `.venv`（Python 3.12、pymoo 0.6.1.1、rouge-score、transformers）中實際執行取得：

| 驗證項目 | 方法 |
|---|---|
| F-1 oracle | 對 `scitldr_test.jsonl` 的 `rouge_scores` 欄位取全域平均，重現 0.13758 ≈ 論文的 0.136；另計算每篇最佳單句平均 = 0.4311 |
| F-2 ROUGE-Lsum | 對 `runs/full_benchmark_result/final_summary/predictions.jsonl`（5622 篇完整測試集）以四種設定重算 |
| F-3 Stage 2 無 PLM | 追蹤 `2_Fusion_Final.yaml` → `optimizer_dispatch.py:139` → `fast_fused.py:113` 的呼叫鏈 |
| F-4 計時 | 在 CPU 上分離量測三個 checkpoint 的 `from_pretrained` 與推論時間，各 3 次取平均，含 warm-up |
| F-5 就地竄改 | 對 8×8 隨機相似度矩陣呼叫前後做元素比對 |
| F-6 超參數 | 全域 grep `pop_size` / `n_gen`；三次獨立執行驗證可重現性 |
| F-7 基數偏誤 | 目標函數的數學性質分析 |
| F-8 SciTLDR | 全 618 篇檢查 `len(sentences)` vs `len(rouge_scores)`、reference 長度統計 |
| F-9 無 baseline | 對 `src/`、`scripts/`、`tests/` 全域關鍵字搜尋 |
| F-12 分句 | Multi-News 前 500 篇、37,349 個句子的長度分布與雜訊偵測 |
| P2-1 mutation | 對 2000 個個體實測 pymoo `BitflipMutation` 的每基因翻轉率 |

### 腳本版本化狀態（2026-07-26 更新）

原稽核腳本只曾位於 `%TEMP%\claude\...\scratchpad\audit_*.py`，不屬於可保存的研究 artifact。
**其中四項已移入 versioned `scripts/audit/`，並由該處重跑確認數字完全一致**：

| 結論 | 版本化腳本 | 重現狀態 |
|---|---|---|
| F-0 系統未贏 Lead | `scripts/audit/lead_vs_system.py` | ✅ 已重現（5,622 篇，數字完全相同） |
| 病因：選句位置與 Lead 重疊 61.7% / greedy ref 22.8% | `scripts/audit/selection_diagnostics.py` | ✅ 已重現（200 篇，數字完全相同） |
| 各資料集 headroom 與 lead bias | `scripts/audit/dataset_headroom.py` | ✅ 已重現 |
| F-4 PLM 載入 vs 推論分解 | `scripts/audit/plm_timing.py` | ⏳ 腳本已版本化，數字須依鎖定 runtime protocol 重測 |

用法與已重現的輸出表見 `scripts/audit/README.md`。

**但版本化不等於升格為論文結果。** 這些腳本仍受下列限制，標籤維持 diagnostic：

- 使用 `src.eval.rouge` 的**內部多句 Lsum 協定**，與 published Perl ROUGE 不保證可比
- greedy reference **不是** exact upper bound，也不是任何資料集的官方 oracle 協定
- 系統端輸入是 test-tuned legacy artifact
- headroom／位置分析是 200 篇抽樣，非全集
- **未做 paired significance test**

要升格為正式證據，必須走 `ACTION_PLAN.md` Phase 2–4 的鎖定流程（官方 split、freeze config、多 seed、paired bootstrap）。
## F-51 — Gate 2 PLM matrix 對每個候選重複編碼同一 frozen-dev 輸入（已修正並通過全量等價驗證）

**嚴重度：P1（成本／可恢復性；若快取未驗證也可能污染 correctness）**

### 重現與證據

在 commit `557b9c0` 的 Multi-News frozen-dev PLM family 中，`sbert_centroid` 與
`sbert_mmr_lambda_0.1` 分別耗時約 `1305.60 s` 與 `1502.09 s`；兩者使用完全相同的
3,935 篇 frozen-dev 文件、eligible sentence sequence、pinned
`all-MiniLM-L6-v2@c9745ed...`、batch size 32 與 256-token 上限，但原 runner 為每個候選
啟動獨立程序並重新做 Transformer inference。第三個候選
`sbert_mmr_lambda_0.3` 因整個 family 命令達 3,600 秒外層限制而中斷。

重現：

```powershell
.venv\Scripts\python.exe -m scripts.audit.run_gate2_baseline_matrix --dataset multinews --family plm
```

### 修正

- `src/models/extractive/encoder_rank.py` 新增預設關閉、只有
  `META_SUM_EMBEDDING_CACHE_DIR` 明示時才啟用的 content-addressed NPZ cache。
- key 綁定 ordered sentence bytes、模型名稱與固定 revision、batch size、token 上限、
  resolved device、Torch／Transformers 版本與 pooling/normalization contract。
- 寫入採同目錄 temporary file + `os.replace`；既存 artifact 損壞、dtype/shape/row count、
  identity 或非有限值不符時 fail loud，不會靜默重算或覆寫。
- cache 是 execution-only optimization，不進 scientific YAML，也不改 frozen candidate hash；
  Gate 2 evidence 另記 root、contract、每 run hit/miss counts 與 ordered row-key digest。
- `.gitignore` 排除 bulk cache；小型 provenance/evidence 仍版本化。

### 驗證狀態

F-51 實作當時完整回歸為 **386 passed**；目前加入 F-53/F-55～F-71 與 D2/D3 runners/analyzers 後為
**453 passed**。預註冊
`configs/preregistrations/f51_embedding_cache_equivalence_v1.json`（SHA-256
`ccdd5a66...66d98`）後，以既有 uncached full-dev SBERT-centroid 為基準完成 cold-populate
與 warm-hit：兩次皆為 3,935/3,935 rows，`selected_indices`、summary、feasibility、eligible
indices／centroid relevance／similarity hashes 與三個 ROUGE 的 mismatch 全為 **0**；三份
selected-index digest 皆為 `e93ee982...cc919`。cold 為 3,935 misses，warm 為 3,935 hits，
ordered row-key digest 同為 `3da91212...2afa6`。selection time 為 `1379.55 s` 與 `89.85 s`
（warm/cold 約 15.35×；只作同機 execution-cost 證據）。

證據：`runs_v2/f51_embedding_cache_equivalence_v1/multinews/dev/equivalence_summary.json`。
因此此一精確 cache contract 可用於續跑**已預註冊**的 Gate 2 PLM candidates；不得藉此
更改候選網格或把 warm search timing 冒充 uncached end-to-end 方法成本。dev-test/test 未讀。
## F-52 — 共用 run evidence 未記錄 PLM runtime 版本（已修正）

**嚴重度：P1（provenance）**

`scripts/audit/run_length_contract_study.py::_dependency_versions()` 原只記 Python、NumPy、
scikit-learn、NLTK、rouge-score 與 PyYAML；Gate 2 的 SBERT evidence 因此缺少實際影響
embedding 的 Torch／Transformers／tokenizers 版本。F-51 cache key 雖已綁 Torch 與
Transformers，run evidence 本身仍不完整。現已補記 `torch`、`transformers`、
`tokenizers` 與 `sentence-transformers`（未安裝時明示 `not-installed`），並由 regression
test 確認本 PLM 環境前三者存在。既有 evidence 保留原貌，不回填假 provenance；後續
cached audit 與 PLM runs 使用新 schema 內容。

## F-53 — Gate 2 family verifier 未驗證 execution-cache provenance（已修正）

**嚴重度：P1（provenance／fail-loud）**

`scripts/audit/summarize_gate2_baseline_family.py` 原先只驗證 candidate 數量、完成狀態、
frozen-dev partition 與 dev-test/test 未讀，之後就直接排名。即使 PLM evidence 宣告啟用
F-51 cache，family verifier 也沒有確認 cache summary 是否存在、列數與 partition 是否一致、
hit/miss 是否涵蓋全部列、contract／ordered-key digest 是否有效，亦未確認 F-52 的 PLM
runtime versions。這不代表既有分數已被證明錯誤，但會讓損壞或不完整的 provenance 通過
family 層驗收。

現已新增 fail-loud 驗證並把聚合診斷寫入 `analysis_summary.json`：只接受
`sbert_mean_pool_l2_npz_v1`、`hit`／`miss_written`，狀態總數必須等於 frozen partition rows，
ordered-key digest 必須是 canonical SHA-256；啟用 cache 的 run 亦必須記錄 Torch、
Transformers、tokenizers 與 sentence-transformers 版本。未使用 cache 的 non-PLM 或早於
F-51 的 legacy PLM run 會明確計入 `disabled_or_legacy_candidate_count`，不會偽造回填。

重現：

```powershell
.venv\Scripts\python.exe -m pytest tests/test_gate2_baseline_family_summary.py -q
```

F-53 目標測試 `7 passed`；當時完整回歸 **388 passed**，目前為 **392 passed**
（2026-08-09）。test split 未讀。

## F-54 — 兩 primary PLM baseline 完成後，S02b 仍未通過 strongest-baseline gate

**嚴重度：P0（方法有效性；目前是 frozen-dev point evidence）**

`gate2-baseline-matrix-v1` 的兩 primary PLM families 已各完成事前註冊的 27/27
candidates。Multi-News winner 是 PacSum-SBERT P03，macro `0.331458`，仍低同資料的
non-PLM PacSum TF-IDF P08 `0.000282`、高 S02b `0.003381`；整體 strongest P08 高
S02b `0.003663`。最佳 full-source SBERT-MMR λ=0.7 macro `0.322581`，低 S02b
`0.005496`。

GovReport winner 是 full-source SBERT-MMR λ=0.9，R1/R2/Lsum
`0.575010/0.241576/0.541718`、macro `0.452768`。三項 point estimate 都高於 LexRank，
但 macro 只高 `0.001148`；它高 proposed S02b `0.034906`。SBERT centroid macro
`0.451441`，也只低 LexRank `0.000179`，顯示 GovReport 的強 baseline 間差距很小，
不可在 paired bootstrap 前宣稱 MMR 顯著勝出。

兩 family 均由 F-53 verifier 通過，final candidate failure 為 0；Multi-News／GovReport
各保留一次 execution interruption 並成功 resume。所有結果只讀 frozen dev，
dev-test/test 未讀。這項結果不等於 semantic route 應刪除：external full-source baseline
與方法內 capacity-matched route ablation 的自變項不同。但它證明目前 S02b 不能晉級；
完成 greedy reference 與 paired baseline matrix 後，必須在 dev 改善 selector／salience，
否則依停止條件重新定位。

證據：

- `runs_v2/gate2_baseline_matrix_v1/multinews/dev/plm/analysis_summary.json`
- `runs_v2/gate2_baseline_matrix_v1/govreport/dev/plm/analysis_summary.json`

## F-55 — 舊 greedy-reference CLI 無 frozen partition／evidence／resume 治理（已修正）

**嚴重度：P0（資料邊界／研究 provenance）**

`src.eval.oracle` 已修正 canonical schema、metric-specific search 與 fail-loud，但它的通用
CLI 仍直接讀整個輸入 JSONL，提供 `--limit`，沒有 frozen-dev manifest／policy SHA 驗證、
逐列 checkpoint、search log 或正式 evidence。直接拿它跑 Gate 2 會把「演算法 correctness」
誤當成「研究流程 correctness」，也可能不小心量到完整 validation 而非 frozen dev。

現新增 `scripts/audit/run_gate2_greedy_reference.py`，固定讀 commit `6a6eddf` 先凍結的
`gate2-greedy-reference-v1`（SHA-256 `04235301...9f735`）。CLI 只有 dataset、target、
workers 與 resume，沒有 split；input／manifest／length-policy SHA、selected-ID digest 與
row count 全部 fail loud。R1／R2／Lsum 是 6 個獨立 configurations；process workers 只作
文件級 execution parallelism，`executor.map` 與逐列 checkpoint 保持 frozen ID order。
resume 必須匹配 exact prefix，外層中斷會另存 interruption evidence 並進 search log；任何
schema/row exception 使整個 configuration fail，不會跳列。

toy corpus exact-equivalence test 證明逐文件組裝與原 `greedy_reference_run(corpus)` 的
selected indices／三 metric 完全相同；CLI boundary 與 checkpoint tampering 另有 regression。
F-55 runner 當時自身 4 tests、與既有 greedy-reference 合併 11 tests 全過，完整回歸
**392 passed**。正式 6 個 runs 尚未完成，故本條只代表
runner 可以安全動工，不代表 headroom 已量得。dev-test/test 未讀。

## F-56 — 外層終止沒有帶走 Windows process tree，造成 checkpoint 雙 writer（已修正）

**嚴重度：P0（artifact integrity；正式分數尚未產生）**

Multi-News R1 的第一次正式 attempt 因 sandbox 禁止 multiprocessing pipe，在 0 rows
fail loud；改以允許 process workers 的環境 resume 後，為提高 CPU 使用率在 228 rows
終止外層 wrapper。但 Windows 上該終止沒有帶走 Python parent／workers；16-worker resume
又啟動第二個 writer。只檢查列數會錯把它當成正常進度，exact-prefix audit 則在 295 rows
中定位到 index 270 回跳到 position 259（期望 position 270）。任何 partial ROUGE 都未讀。

兩棵已辨識的 process tree 已精確終止。`recover_greedy_reference_checkpoint.py` 將原
295-row 檔完整封存（SHA-256 `b5357d17...705a7883`），丟棄污染尾端 25 rows，只保留並
重驗 270-row frozen-ID exact prefix（SHA-256 `f294388e...1104bce6`）；recovery evidence
與 failed search-log entry 均保留，dev-test/test 未讀。

永久修正是在每個 dataset/target 外包 OS-level non-blocking lock；即使 wrapper 消失，
仍存活的 Python parent 會持鎖，第二個 writer 必須 fail loud。另加 longest-exact-prefix
與 duplicate-writer regression；greedy-reference 相關測試 13 passed，完整回歸
F-56 當時完整回歸 **394 passed**；目前為 **431 passed**（2026-08-09）。後續不得再用外層 terminate 調整 worker 數；讓 invocation
正常完成或由 runner 的既有 checkpoint/resume 處理真實外部中斷。

## F-57 — Multi-News metric-specific greedy reference 完成，舊單一 R-1 診斷不足

**嚴重度：P1（headroom 診斷完整性）**

在 F-55/F-56 治理後，Multi-News frozen dev 3,935 rows 的三個獨立 target 全部完成：
R1 target 的 R1 為 `0.595288`，R2 target 的 R2 為 `0.345229`，Lsum target 的 Lsum
為 `0.558596`。對應平均摘要長度為 `213.29/170.51/212.66` words；R2 最短為 0 words，
代表該列沒有任何正 R2 gain，並非 skip 或 schema failure。這證明只拿 R1 greedy 選句再
報三個 metric 會低估 R2/Lsum 各自可達的 greedy 診斷值。

三份 evidence 都有 input／manifest／policy/preregistration SHA、3,935-row coverage、
selected-indices digest、dependency versions 與 dev-test/test=false。Lsum invocation 為
`2,531.55 s`，其中 frozen position 277 的文件有 2,128 句／40,773 source words，造成
ordered tail cost；未因耗時排除。這些仍不是 exact upper bound，也未宣稱 significance。

候選 overlap/headroom 公式已另在任何 recall 數字前凍結；v1 的 route-pool 結構錯誤
依 F-58 在計分前由 v2 supersede。GovReport 3/3 與分析後續亦完成；最終 Gate 2 結論見
F-62。test 未讀。

## F-58 — candidate-record route membership 不等於完整 route top-40（分析前已修正）

**嚴重度：P1（metric definition）**

最初的 `gate2_greedy_reference_analysis_v1.json` 將 route pool 定義為
`candidate_records.selected_by_routes`，同時標成 recall@route-top-40。但
`candidate_records` 只保存通過 union total-cap 後的候選；某 route 提出的 top-40 句若被
RRF/total cap 移除，就不在 records 內。照 v1 計算會把「保留在 union 內的 route
membership recall」錯稱完整 route recall@40，系統性低估各 route proposal 的覆蓋。

此錯誤在任何 candidate-overlap score 產生前由 artifact schema inspection 發現；v1 檔案
與 SHA 永久保留，不覆寫。新 `gate2_greedy_reference_analysis_v2.json`（SHA-256
`ef45c056...a3e39`）明確 supersede v1，route set 改讀
`candidate_pool.route_proposals[route]`，union 仍讀 cap 後 candidate records，final set
仍讀 selected indices。v2 明記 `overlap_scores_accessed_before_correction=false`。

分析器另驗每列 route top-K=40、total cap=80、actual size、三 route 完整性與 frozen ID
alignment；headroom/recall 三個 golden tests 通過。本條完成時正式 overlap 尚未執行；
其後結果見 F-59。dev-test/test 未讀。

## F-59 — Multi-News 候選池覆蓋尚可，但 selector 只保留約三成 greedy selections

**嚴重度：P0（方法瓶頸／promotion gate）**

依 F-58 v2 protocol 首次計算 3,935-row frozen-dev overlap。S02b union-cap-80 對
R1/R2/Lsum greedy selections 的 micro recall 分別 `0.858871/0.819072/0.843863`；
final selected set 則只有 `0.300666/0.296301/0.308456`。union 仍漏 14–18%，但從 union
到 final selection 的損失遠大於 route candidate generation，故目前主要瓶頸是
selector/salience/objective，而不是「再加一條 route」。

route-top-40 的 micro recall 中 graph 三項均最高（`0.736992/0.678234/0.721767`），
semantic 第二（`0.719397/0.661846/0.701237`），lexical 第三；graph exclusive hits 亦為
`1,729/2,297/2,048`，semantic 為 `1,276/1,651/1,420`。因此 semantic/graph 的候選
效用刪除條件未觸發，但 semantic always-on 的高成本仍待 GovReport 與 adaptive rule 決定。

metric-specific headroom 結果更嚴格：S02b 的 R1/R2/Lsum capture 為
`3.573%/−4.229%/4.892%`，平均 `1.412%`；P08 為
`4.886%/0.977%/4.047%`，平均 `3.303%`。S02b R2 不只沒有吃到 headroom，還低於
Lead。舊約 2.4% 估計不再作現行結論。

證據：`runs_v2/gate2_greedy_reference_v1/multinews/dev/analysis/analysis.json`
（SHA-256 `921363be...bb027`）與 `evidence.json`；輸入 SHA、3,935-row ID alignment、
route top-K/cap 與三 greedy evidence 全部 fail-loud 驗證。F-59 當時完整回歸 **397 passed**。
這是 reference-aware diagnosis，不可拿 reference 特徵進實際 selector；GovReport 後續結果
見 F-62，dev-test/test 未讀。

## F-60 — GovReport greedy R-Lsum 的 eager process queue 造成不必要的 CPU／memory 壓力（已修正，run 待續）

**嚴重度：P1（execution safety／可恢復性，不改科學協定）**

2026-08-09 的 GovReport R-Lsum invocation 明確指定 16 workers。canonical validation
檔為 222.1 MB；Python 3.12 `ProcessPoolExecutor.map` 會預先提交剩餘 iterable，將大型
document tasks 序列化進 process queue。實測 CPU 超過 90%；外層工作被強制中斷後，
Python parent 與 16 workers 仍存活。每個 worker 約 0.16 GB working set／1.31 GB
private bytes，parent 約 0.60 GB working set。這些 child processes 在工作管理員可能被
歸在 Codex 啟動的 process tree 下，但 Codex 本體當時約 0.21 GB working set，並非主要
RAM 消耗者。已依相同 2026-08-09 09:09:53–09:09:57 start time 精準終止 17 個程序；
checkpoint 是 exact frozen-dev prefix 257/681，completed evidence 不存在，沒有把部分結果
當成完成 run，dev-test/test 均未讀。

`run_gate2_greedy_reference.py` 現以 bounded submit/wait 取代 eager `executor.map`：
pending tasks 不超過 worker 數，完成的輕量 row results 可等待前方長尾，仍按 frozen
manifest 順序 flush。預設 worker 由 4 降為 2；明確提高 workers 仍屬 execution-only，scientific config
hash、metric、長度與 selected-index 語義不變。toy corpus 的 spawn/order/equivalence 與
runner governance 合計 15 tests 通過，F-60 當時完整回歸 399 passed。

GPU 不作為本 finding 的修正：目前內圈是 `rouge-score` 的 tokenization、stemming、
n-gram count 與 R-Lsum union-LCS，沒有 PyTorch/CUDA tensor route。另寫 GPU evaluator
反而需要完整 selected-index equivalence audit；RTX 4060 應用在 SBERT embedding 階段。
R-Lsum 只在使用者允許重新佔用算力後，以 `--resume` 從 257/681 繼續。

## F-61 — bounded ordered buffer 會在前方長尾時讓其餘 workers 閒置（已修正，run 進行中）

**嚴重度：P1（execution throughput／失敗紀錄，不改科學協定）**

F-60 第一版同時限制 `pending + completed-order-buffer <= workers`。GovReport R-Lsum
從 257/681 以 16 workers resume 後，第一輪約 4,923 CPU-seconds即顯示 15 workers 已
完成各自 row 並待機，只剩 position 257 長尾使用約一個 core；checkpoint 因 exact-prefix
規則仍為 257。該 invocation 由 operator 在 prefix 未變時停止，runner 的
`BrokenProcessPool` failure evidence/search-log 與人工說明的 `attempt_02_interrupted`
均保留，沒有覆寫或假裝成功；dev-test/test 未讀。

修正後只限制 pending tasks 不超過 worker 數；已完成 row result 只有 ID、indices、長度與
三個 metrics，可在記憶體等待 canonical 前方長尾，其他 worker 繼續取新 document。
第二次 16-worker resume 約 70 秒累積 1,043 CPU-seconds，接近 16-core 全速，working set
約 3.18 GB。這是 execution-only scheduler change；exact-order flush、scientific config
hash、selected indices 與 checkpoint contract 不變。scheduler targeted tests 8/8 通過；
完整回歸於 CPU-heavy R-Lsum 完成後為 403 passed。

同一 checkpoint 期間另凍結 `gate2_paired_finalists_v1.json`（在任何 paired resampling
outcome 前，但 aggregate/per-example scores 已存在）並加入 analyzer；正式 paired run
其後完成，見 F-62。其 toy governance tests 4/4 通過。

## F-62 — Gate 2 dev diagnosis 完成；S02b 對兩 primary adversarial winner 均顯著落後

**嚴重度：P0（方法有效性／promotion gate）**

GovReport R-Lsum 由 257/681 exact prefix 以修正後的 16-worker scheduler 安全完成；R1、
R2、Lsum 三個 metric-specific greedy-reference target 分數分別為
`0.726030/0.491599/0.704947`。三份 artifact 都有 681-row frozen ID order、input／manifest／
policy／preregistration SHA 與 `dev_test_accessed=false`、`test_split_accessed=false`。
GovReport S02b 對 Lead→greedy 的平均 headroom capture 是 `8.635%`，最強 SBERT+MMR 是
`22.980%`；S02b union recall 為 `56.21%/46.92%/53.86%`，final recall 僅
`12.36%/13.42%/12.32%`。因此 GovReport 不只 selector 弱，候選覆蓋也不足。semantic／
graph 都有 exclusive hits，route deletion gate 尚未觸發。

paired finalist protocol 在任何 resampling outcome 前凍結：2 datasets × 8 family finalists ×
4 endpoints，共 64 個 Holm endpoints；另以 31 proposed search operations、每資料集 52 baseline
candidates 計算 12,896-opportunity selection-aware Bonferroni diagnostic。10,000 bootstrap、
seed 20260809 的正式結果為：

- Multi-News adversarial winner PacSum TF-IDF P08：S02b macro delta `−0.003663`，95% CI
  `[−0.005776, −0.001638]`，Holm `p=0.021598`。
- GovReport adversarial winner SBERT+MMR λ=0.9：S02b macro delta `−0.034906`，95% CI
  `[−0.038498, −0.031373]`，Holm `p=0.012799`。
- selection-aware wins：`0`。

證據為 `runs_v2/gate2_greedy_reference_v1/govreport/dev/analysis/analysis.json`（SHA-256
`18e3c901...5a240`）與 `runs_v2/gate2_paired_finalists_v1/summary.json`（SHA-256
`1781ebe5...0941`）；完整回歸 **403 passed**。這完成 Gate 2 的 dev diagnosis，但不是通過
quality gate：S02b 不得晉級 dev-test。下一步只允許 frozen-dev redesign；test 仍鎖定。

## F-63 — 通用研究 runner 把 canonical MMR/NSGA-II 錯送 baseline CLI，且只認 baseline cache provenance（已修正）

**嚴重度：P0（方法身分／evidence plumbing；新 selector 分數尚未產生）**

`d2-selector-full-dev-v1` 在任何新 selector-swap score 前已提交預註冊。第一次 Multi-News
attempt 中，`_run_method()` 對所有非 `greedy` 名稱都呼叫 `src.baselines.cli --baseline`；
因此 MMR/NSGA-II 13 個候選在 method dispatch 前即由 argparse exit 2，並未被偷偷替換成
其他方法。Greedy+SBERT 完成 3,935-row selection，但 postprocessor 又只在
`baseline_diagnostics.representation` 尋找 cache provenance，沒有讀 canonical pipeline 的
`selector_inputs.representation`／`optimizer_diagnostics.selector_representation`，因此在
ROUGE 計算前 fail loud。沒有任何 D2 quality score 被讀取或寫入。

第一次修正後，Greedy+SBERT retry 成功；TF-IDF-similarity MMR 又顯示 semantic route 的
cache provenance 只存在 `candidate_records[*].route_scores.semantic.metadata`，並非 selector
representation。S02 selection 完成但仍在 ROUGE 前 fail loud；S03 partial 由精確辨識的
runner/child process tree 停止，避免其餘候選重複浪費。兩次失敗都保留且不計品質。

修正為 `_run_method(..., pipeline_selector=True)` 時 Greedy/MMR/NSGA-II 全走 canonical
`src.pipeline.select_sentences`，method 身分仍取 frozen config；cache summarizer 同時支援
baseline 與 canonical selector views，若同列多個 view 不一致則 fail loud。原 13 個 failed
search-log rows、config、command/evidence 與 Greedy+SBERT selection artifact 保留；retry
會先移入 numbered `attempt_*_failed`，不覆寫。新增 dispatch 與 cache-provenance regression。
dev-test/test 未讀。

## F-64 — 200-row MMR 優勢未外推到 Multi-News full dev；NSGA-II 高成本仍輸 Greedy

**嚴重度：P0（方法有效性／selector freeze）**

F-63 修正後，事前提交的 `d2-selector-full-dev-v1` 在 Multi-News frozen dev 3,935 rows
完成 14/14 candidates；study summary 明記 `dev_test_accessed=false`、
`test_split_accessed=false`。固定 S02b routes、candidate pool、RRF salience 與 A1
200–250 words，Greedy-TFIDF anchor macro `0.328077` 仍最佳。NSGA-II+TF-IDF
`0.322615`、最佳 TF-IDF-MMR（λ=0.3）`0.321744`、最佳 SBERT-MMR（λ=0.7）
`0.316612`、NSGA-II+SBERT `0.308827`。

NSGA-II+TF-IDF selection `4690.7s`，約為最佳 TF-IDF-MMR 的 `31.5×`，但比 Greedy
低 `0.005462` macro；SBERT NSGA 更低。兩個 NSGA 候選均未達預註冊多 seed 門檻，
因此不追加 seed。200-row pilot 的「MMR main」只能保留為早期 diagnostic，不能再當
full-dev 架構決策；Multi-News 暫以 Greedy 為 anchor。這加強 §7.3 將 NSGA-II 移出
核心與標題的依據，但當時還不能跨資料集宣布最終 selector；後續見 F-65。

## F-65 — D2 跨資料集要求 task-profile selector；GovReport MMR 顯著改善仍未勝強 baseline

**嚴重度：P0（方法有效性／promotion gate）**

GovReport 14/14 selector candidates 完成；TF-IDF-MMR λ=0.7 macro `0.446154` 最佳，
對 Greedy anchor `+0.028293`。事前凍結的 104-endpoint paired bootstrap 中，其 R1／
R2／R-Lsum／macro CI 全正；Holm `p=0.020798`、以 57 search operations × 4 endpoints
計算的 228-opportunity correction `p=0.045595`，四項均通過。NSGA-II+TF-IDF 僅
`0.426844`、selection `1708.5s`；MMR λ=0.7 selection `87.9s`，NSGA 約 `19.4×`
仍較差。NSGA-II+SBERT 更低至 `0.371552`，兩者均不觸發多 seed。

跨資料集沒有 shared selector 在兩邊都距 winner ≤`0.001`。依預註冊採顯式
task-profile policy：multi-document multi-sentence→Greedy-TFIDF；single-document
multi-sentence→TF-IDF-MMR λ=0.7。這不是 hidden dataset switch。

但 promotion gate 仍失敗：Multi-News winner 低 P08 `0.003663`；GovReport winner 低
full-source SBERT+MMR `0.006614`。`promotion_eligible=false`，dev-test/test 皆未讀。
完整證據：`runs_v2/d2_selector_full_dev_v1/analysis/paired_summary.json`。

## F-66 — Equal-weight RRF 無 route calibration；D3a 以 fail-loud weighted RRF 接線並在分數前凍結

**嚴重度：P1（provenance fusion／方法搜尋治理）**

D2 的 `rrf_fusion` 雖確實送入 selector，但三路固定等權；route raw score 尺度不可比，
目前也沒有證據證明 lexical／semantic／graph 應等權。這不是 correctness bug，卻是仍未
搜尋的架構假設。新接線 `candidates.route_weights` 只在 RRF contribution 上套顯式正值，
未設定時三路仍精確為 `1.0`，因此舊配置語義不變。disabled route、零、負值或非有限值
全部 fail loud；實際 weights、RRF constant 與 fusion method 寫入 candidate allocation。

任何 weighted-RRF 分數前，`d3a-router-fusion-screen-v1` 已凍結兩 profile 各 14 案：
equal-weight anchor、四個 bounded capacity 案、三個 lexical-salience 案與六個單一路權重
案。先用 reference-blind 200-row dev pilot 初篩，再依固定 family／effect 門檻每 profile
最多送四個非 anchor 到 full dev。執行器只接受 dev pilot manifest，worker 上限 16 且
每 worker 一個 BLAS thread；dev-test/test 無 CLI 入口。

兩個 pilot 28/28 已完成。Multi-News bigrams+position 對 anchor `+0.002559`，macro
95% CI `[+0.000502,+0.004677]`，但 Holm-104 未通過；GovReport lexical×0.5、
semantic×2、graph×2 分別 `+0.009648/+0.006849/+0.005620`。Gov 的前三個晉級正訊號
macro CI 全正且 Holm-104 `p=0.020798`，但以 83 configurations × 4 endpoints 的
332-opportunity correction 為 `p=0.066393`。因此它們只能依 frozen rule 取得 full-dev
確認資格，**尚不能宣稱 weighted RRF 有效或已勝 adversarial baseline**。finalists 與
104-endpoint evidence 在 `runs_v2/d3a_router_fusion_screen_v1/analysis/pilot_analysis.json`。
後續 7 個非 anchor finalists × 4 endpoints 的 full-dev confirmation 亦已在任何該層
分數前凍結；runner 以 16 個 bounded row workers 串流寫 predictions，只在記憶體保留
summary／selected-index digest，避免重現整批 provenance RAM spike。此處仍未讀 dev-test/test。

## F-67 — D3a full dev：GovReport 首次勝 strongest baseline；Multi-News 仍敗，且 10k bootstrap 無法解析 332-way gate

**嚴重度：P0（方法有效性／統計治理）**

事前由 pilot rule 選出的 7 個 non-anchor finalists 已在完整 frozen dev 跑完。Multi-News
最佳 bigrams+position macro `0.329704`，比 D2 anchor `+0.001627`，但對 PacSum P08
仍 `−0.002036`；正式 paired R-2 `−0.008971`、CI `[−0.011306,−0.006592]`，因此
不是只差 power。GovReport lexical×0.5 macro `0.456423`，比 D2 anchor `+0.010269`，
並高 full-source SBERT+MMR `+0.003655`；macro CI `[+0.001446,+0.005876]`、
Holm-28 `p=0.023798`，R-1/R-2 亦為正，沒有 component CI 全負。這是 proposed method
第一次在 full frozen dev 對 strongest local baseline 得到正式正證據。

但 D3a 預註冊同時要求 83 configurations × 4 endpoints 的 332-way Bonferroni。
10,000-resample finite-corrected two-sided bootstrap 的最小 p 是 `2/10001=0.00019998`；
乘 332 後理論下限就是 `0.066393`，故此 gate 在數學上不可能到 0.05。GovReport
winner 實際 macro raw p `0.001400`、selection-adjusted `0.464754`，仍依原規則判失敗；
不得事後增加 D3a resamples。下一個且最後允許的 D3b combination 必須在任何分數前
凍結足夠的 resampling resolution。Multi-News 與 GovReport 均未讀 dev-test/test。
完整 evidence：`runs_v2/d3a_router_fusion_full_dev_v1/analysis/paired_summary.json`。

## F-68 — D3b 最終組合在分數前凍結；提高 bootstrap resolution 並綁定終止決策

**嚴重度：P0（研究治理／停止條件）**

D3a 只留下兩個可交叉組合、且在各自 profile 上有獨立正訊號的方向。因此 D3b 不再開新
grid，而是在任何 D3b 分數前固定每個 profile 一案：Multi-News 將
bigrams+position 與 graph×2 合併；GovReport 將 bigrams+position 與 lexical×0.5
合併。預註冊檔 SHA-256 為
`58c6c98c1953416463bf000c102d351a320ef3634be91d3fd478827ac12bc112`。

正式比較固定為 100,000 次 paired bootstrap、2 profiles × 4 endpoints 的 Holm-8，
並保留歷次 85 configurations × 4 endpoints 的 Bonferroni-340。有限樣本雙尾 p 值最小
為 `2/100001`，乘 340 後為 `0.00679993`，因此不會重現 F-67 的數學不可能 gate。
兩個 profile 均須 macro point estimate 為正、macro CI lower > 0、Holm-8 與
Bonferroni-340 均 ≤ 0.05，且不得有 component CI 全負。全部通過才寫 freeze 建議；
任一失敗就寫重新定位建議並停止。runner 只接受 frozen dev，最多 16 個 bounded row
workers、每 worker 一個 BLAS thread，沒有 dev-test/test CLI 入口。當時完整回歸
**429 passed**；此時尚未觀察任何 D3b 組合分數。

## F-69 — D3b：GovReport 通過，Multi-News R-2 明確失敗；觸發重新定位停止條件

**嚴重度：P0（方法有效性／投稿決策）**

D3b 兩個單一組合都以 16 workers 完成 frozen dev。GovReport 681/681 rows、0
infeasible，macro `0.457404`，相對 D3a anchor `+0.000981`，相對 strongest
full-source SBERT+MMR `+0.004636`。正式 macro CI
`[+0.002507,+0.006745]`、Holm-8 `p=0.000240`、Bonferroni-340
`p=0.013600`；R-1／R-2 亦為正，故該 task profile 通過全部 preregistered checks。

Multi-News 3,935/3,935 rows，其中 3,930 feasible、5 個依既有 F-17 contract 記為
infeasible。macro `0.330417`，相對 D3a anchor `+0.000714`，但仍低 PacSum P08
`0.001323`；macro CI `[−0.003372,+0.000697]`。component 顯示 R-1
`+0.000391`、R-Lsum `+0.004207`，但 R-2 `−0.008565`、CI
`[−0.010924,−0.006221]` 全負。因此不是只有總平均 power 不足；方法的 multi-document
局部 bigram precision 仍有明確缺陷。

預註冊要求兩個 task profiles 都通過，故 `all_profiles_eligible=false`。搜尋依停止條件
結束，不得新增 dev grid、不得讀 dev-test/test。已寫
`REPOSITIONING_RECOMMENDATION.md`，建議將可辯護主張縮為 training-free、
provenance-preserving 的長篇單文件框架，並把 Multi-News 作為適用邊界；若要改 frozen
data policy／claim matrix，必須由老師與作者另行簽字。

第一次 GovReport invocation 因 Codex Windows sandbox 對 child process 的
`PermissionError` 在未完成前失敗，沒有可用分數；原始 summary、暫存 prediction 與
search-log record 已保留於 `govreport/attempts/attempt_01_sandbox_permission/`。成功重試
在 sandbox 外依同一 scientific config 執行，沒有覆寫失敗 evidence。完整回歸目前為
**431 passed**。完整 paired evidence：
`runs_v2/d3b_cross_profile_combination_v1/analysis/paired_summary.json`。

## F-70 — PR #16 暴露 CI dependency 與 pytest temp-path 的隱性本機假設

**嚴重度：P1（跨平台可重現性／CI correctness）**

PR #16 第一個 Linux run `31322928718` 在 collection 階段出現 5 個 error；共同原因不是
五個程式缺陷，而是 `requirements-ci.txt` 仍聲稱測試不需 PLM dependencies，但新增的
encoder cache、PacSum-SBERT 與 integration tests 已在 module scope 使用 `torch`。
CI 現在從 PyTorch CPU index 安裝 `torch==2.8.0`，並明列 `transformers==4.56.0`；
測試只使用小型 tensor 與 mocked checkpoint，不下載模型或 CUDA runtime。

依賴修正後的 run `31323125422` 已完成 421 passed／5 skipped，但另有 5 failures：
pytest 的 Linux `tmp_path` 位於 repo 外的 `/tmp`，而相關測試沒有宣告測試用 repo root，
觸發 production `_relative()` 的 repo-bound evidence guard。本機先前固定 `--basetemp`
於 workspace，因而遮蔽此假設。修正沒有放寬 production guard；五個 tests 改為將各自
`tmp_path` 明確注入為測試用 `REPO_ROOT`。以 Windows workspace 外 TEMP 重跑相關
22 tests 與完整套件分別為 22 passed、431 passed。

GitHub Linux run `31323270403` 最終 **pytest pass**（54 秒）。本修正沒有執行資料實驗，
也沒有存取 dev-test/test；只更正 CI dependency contract 與跨平台 test fixture。

## F-71 — CRLF-era SHA-256 pins 在 clean Linux checkout 失效（已以 fail-loud errata 修正）

**嚴重度：P0（provenance correctness／跨平台可重現性）**

PR #16 review 發現 19 組 text-provenance SHA-256 pins 是對 Windows `CRLF` bytes
計算，但 Git checkout 的 canonical content 為 `LF`；其中兩個 validation partition
manifests 與兩個 D3a pilot manifests 會讓 clean Linux/macOS checkout 在正式 runner
入口就 fail loud。三方 `selected_ids_sha256`（manifest、committed file、歷史
`partition_preflight.json`）與 reference-blind partition 重建均一致；這是換行字節
identity 問題，不是 dev/dev-test row assignment 或科學內容變更。

`sha256_file()` 現對 UTF-8 text provenance 正規化 `CRLF→LF`，並以 NUL guard
防止誤用在 binary artifact；GovReport archive 改用 raw-byte
`sha256_binary_file()`。`configs/pin_errata_lf_normalization.json` 以 legacy/canonical
雙 hash 同時符合才回報 `legacy`，單邊符合仍失敗。19/19 representative
files 已機械驗證 CRLF/LF 雙向 digest；同一 snapshot 的 19 個 legacy hashes 共
1,709 次 textual occurrences、分布於 1,134 個 unique files。

`scripts/audit/verify_provenance.py` 對 `runs_v2` 實測為
`468 legacy / 0 fail`。朋友的三個修正 commits 未改任何既有
preregistration、evidence 或實驗數字，也未存取 dev-test/test。Windows sandbox
內的 pytest temp ACL 會產生大量 setup errors；改用明確的 sandbox 外
system-temp basetemp 後，bare `pytest` 完整回歸為 **453 passed**，全樹
`compileall -f src tests scripts` 亦通過。此修正只修復 provenance/CI
portability，不構成新的方法效果證據。

## F-72 — GovReport 內部 evaluator 尚未證明等同 published official protocol

**嚴重度：P0（claim validity／final-evaluation governance）**

2026-08-10 檢查 GovReport 論文與作者官方 repository 後，確認官方評測不是本 repo
目前的 Google `rouge_score` + 共用 Punkt ROUGE-Lsum 路徑。官方
`LongDocSum` repository（檢查 commit
`ee0dd33f2fde9d19b9a15d81884418d68320e5ca`）在 `Model/eval_model.py` 使用
Stanza `Pipeline(lang='en', processors='tokenize,mwt')`，把 token 以空白連接、句子以換行
連接；`Model/evaluate.py` 再透過 `pyrouge.Rouge155` 呼叫 Perl ROUGE-1.5.5，參數含
`-c 95 -r 1000 -n 2 -m`。因此目前 D3b 的內部 evaluator 結果可作 development evidence，
但在 parity 完成前不能直接稱為 GovReport published-protocol main result。

**修正／守門：**

- 新增 `configs/preregistrations/govreport_centered_evidence_completion_v1.json`，固定在已
  凍結 GovReport dev outputs 上重算 proposed 與八個 baselines；official protocol 為
  manuscript authoritative，內部 evaluator 只作 secondary diagnostic。
- 若 official protocol 使 proposed 與預指定 strongest baseline 的排序反轉，立即撤回
  GovReport quality-superiority claim；不得再調 selector、route、budget 或 weights。
- 新增 `configs/preregistrations/govreport_centered_final_evaluation_v1.json`，但 execution
  維持 `locked`，且不建立 test data policy；老師／完整作者群簽字前不可執行。
- 新增 `scripts/audit/verify_govreport_freeze_package.py` 與 4 個 regression guards，驗證
  dataset role、frozen config、protected-split lock 與 final execution lock。此 verifier
  只雜湊版本化 metadata 與既有 dev evidence，不開啟任何 dataset split。

**重現來源：** GovReport paper `https://arxiv.org/abs/2104.02112`；官方程式
`https://github.com/luyang-huang96/LongDocSum`。本條沒有產生新的 ROUGE 分數，也沒有
存取 dev-test/test；它只登錄 evaluator mismatch risk 與 freeze 前必做的驗證。

驗證結果（2026-08-11）：freeze-package targeted 4/4、完整 pytest **457 passed**、
`compileall -f src tests scripts` 通過，provenance audit 為
`468 checked / 468 legacy / 0 fail`；freeze verifier 回報 protected splits locked、
`test_split_accessed=false`。
