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
> **2026-08-06 selector evidence**：200-row reference-blind matched pilot 與五 seed
> NSGA-II stability extension 已完成。MMR 對 Greedy 的 R-1/R-2 paired gain 為
> `+0.01488/+0.01472` 且 Holm-significant；NSGA-II 五 seed mean 均低於
> Greedy、selection Jaccard 僅 `0.639`。selector 因此採 MMR，NSGA-II 降為
> comparator；這仍未回答 full-source MMR／PacSum／兩 primary 的 system gate。
> **2026-08-08 development-split governance**：先前所有 validation pilot 都直接使用
> full validation，沒有可供「反覆搜尋」與「單次確認」分離的 manifest。F-20 已為
> Multi-News 與 GovReport 都已補上 reference-blind dev/dev-test 凍結與 runtime
> enforcement；GovReport 資料層詳見 F-22。兩者均未讀取 test split。

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
| **F-0** | 系統未贏 Lead | 🔴 成立 | ⏳ **未量測** | 待 MVP 跑完 validation 才知道。**這是 go/no-go 的核心,尚未回答** |
| F-1 | 論文 "oracle" 不是 oracle | 🔴 成立 | ✅ 已可正確計算 | `src/eval/oracle.py`；canonical 與三個 metric target 已修，詳見 F-21。舊稿 0.136 須撤回 |
| F-2 | ROUGE-L 應為 Lsum | 🔴 成立 | ✅ 已修 | `src/eval/rouge.py`；published-protocol parity 仍待驗證 |
| F-3 | Stage 2 沒有 PLM | 🔴 成立 | ✅ **已修** | 新增 semantic route + `selector.salience_source: rrf_fusion`。`fast_fused.py` 保持原狀 |
| F-4 | PLM 每篇重載模型 | 🔴 成立 | ✅ 程式已修 | `load_encoder()` 快取。**但正式計時數字仍須依鎖定 protocol 重測** |
| F-5 | 相似度矩陣就地竄改 | 🔴 成立 | ✅ 已修 + regression test | `src/features/graph.py` |
| F-6 | `pop_size`/`n_gen`/`seed` 未接線 | 🔴 成立 | ✅ 已修 | `optimizer_dispatch.py` |
| F-7 | salience 用總和 → 基數偏誤 | 🔴 成立 | 🟡 部分禁止（僅限有 `task_profile` 的 profiled multi_sentence；例外詳見 F-14） | `objectives/factory.py` 拒絕 profiled multi_sentence config 用 raw sum；legacy_unprofiled（無 `task_profile`）與 legacy 保留 sum |
| F-8 | SciTLDR 多重 reference 被串接 | 🔴 成立 | ✅ 已修 | `preprocess_scitldr.py` 改存 `references: list` |
| **F-9** | **repo 無任何 baseline 實作** | 🔴 成立 | 🟡 **部分解除** | Lead、Random、TextRank／LexRank、full-source SBERT centroid／MMR 程式已加入；舊 Multi-News full-validation rerun 只保留為 historical diagnostic。PacSum、partitioned SBERT run、GovReport 方法 runs與完整 paired matrix 尚未完成，Gate 2 未通過 |
| F-10 | 圖模組 τ 套用不一致 | 🔴 成立 | ✅ 已修 | τ 已傳入 `feature_builder.py` 與 graph route |
| **F-11** | `centrality` 與 `novelty` 完全反相關 | 🔴 成立 | 🔴 **仍然成立** | **未修**。新 MVP 兩者權重皆 0 所以不觸發,但退化仍存在 |
| F-12 | 分句用純正則 | 🔴 成立 | 🟡 部分修 | Multi-News／GovReport canonical 已改 NLTK Punkt；**legacy `preprocess.py` 未動，CNN-DM 僅在 Gate 3 後納入時處理** |
| F-13 | 其他（requirements 重複、pytest 缺、靜默 fallback…） | 🔴 成立 | ✅ canonical 主路徑已修 | requirements 去重 + pytest、fallback 移除；Greedy／GRASP／NSGA-II 共用同一 evaluator。legacy `fast_*` 路徑仍只供舊 artifact 重現 |

### 目前真正還開著的（不要被上面的 ✅ 誤導）

1. 🔴 **F-0 尚未在新 pipeline 上回答** —— 修好一堆東西不等於贏過 Lead
2. 🔴 **F-9 baseline 矩陣仍不完整** —— Multi-News TextRank／LexRank final rerun 已完成；PacSum、SBERT+MMR、GovReport 與 paired significance 未完成，F-0 仍無法回答
3. 🔴 **F-11 centrality/novelty 退化** —— 若日後啟用這兩個特徵會出問題
4. 🟡 **F-12 legacy 分句與條件式 CNN-DM 分句規則**
5. 🟡 **F-4 的正式計時數字**、**F-2 的 published-protocol parity**
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

**目前狀態（2026-08-06）**：Lead、Random、TextRank／LexRank 與 full-source SBERT
centroid／MMR 程式皆已接線，
所以「目前 repo 零 baseline」已不再成立。Lead／Random 共用 canonical data-policy
preflight 與 output upper-bound contract；TextRank／LexRank 包裝 pinned sumy。PR #15
已移除 offline `punkt_tab` regression，Windows 312 tests 全過且 Linux CI 綠燈；兩者也已
在 frozen Multi-News validation 以最終實作完成 5,621-row full-split rerun。PacSum、
SBERT full run、PacSum、GovReport 與完整 paired matrix 仍未完成。因此 F-9 仍只能列「部分解除」，
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

**影響範圍**：這是 dev finding，尚未看 dev-test，不能據此 freeze 200–250。四個協定仍依
preregistration 各評估一次 dev-test；若 floor 協定最後被選中，論文必須把它描述為
selector stopping/feasibility mechanism，並報 infeasibility rate，不能只寫成公平的輸出長度
上限。證據在 `runs_v2/a1_length_contract/multinews/dev/`；所有 12 method runs 均有
config/prediction SHA、逐篇 selected-indices digest、dependency versions，search registry
恰新增四筆，test split 未存取。

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
| **P1-1** baseline | 「必補 Lead-3、PacSum、SBERT centroid、LLM zero-shot」 | ✅ 同意。**補充**：稽核當時 repo 連舊論文報告的 Lead/TextRank/LexRank 都沒有實作（F-9）；現在已補實作與 historical Multi-News diagnostics，但 PacSum、partitioned SBERT+MMR、GovReport 方法 runs 與 paired significance 仍缺 |

### 稽核補充、且已收斂進主計畫／行動清單的項目

0. 🔴 **F-0（legacy Multi-News 未贏 Lead）** —— 同資料同內部 evaluator 下，ExpB 只在 R1 高 0.0021，R2/R-Lsum 較低；因 ExpB test-tuned，只能觸發 redesign，不能當新結果。
1. **F-3（Stage 2 無 PLM）** —— 方法章與實作不符，必須在任何新實驗前解決。
2. **F-5（相似度矩陣就地竄改）** —— 會靜默改變實驗語意；狀態見 §0.0 狀態表。
3. **F-9（legacy 無 baseline、目前矩陣仍不完整）** —— 舊表不能回溯當作本地重現；Multi-News TextRank／LexRank final rerun 已補，Gate 2 尚缺 PacSum、SBERT+MMR、GovReport、paired significance 與正式兩-primary矩陣。
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

以下是初次 audit patch 與目前狀態的對照。pytest 已安裝，2026-08-05 的 master
**332 local tests 全過（2026-08-08）且 PR #15 Linux CI 綠燈**；這只代表 correctness regression、10-document snapshot、內部
hand-calculated golden 與 Lead plumbing 受測，不代表方法效果或 published-protocol parity 已通過。
Sentence-BERT production route、canonical NLTK segmentation、shared objective/selector
contract 與 Lead／Random／TextRank／LexRank／SBERT centroid／MMR baseline 已接線；centrality offline hotfix 與
Multi-News TextRank／LexRank final rerun 已完成。PacSum、SBERT full run、GovReport、paired
significance、兩個 primary 的完整 baseline 矩陣與 proposed-method validation 仍未完成。

| 檔案 | 修改內容 | 對應發現 | 驗證 |
|---|---|---|---|
| `src/eval/rouge.py` | ROUGE-Lsum；同一 reference 由最大 R1 選定；長度 mismatch fail；保留 legacy evaluator | F-2, F-8 | ✅ 兩個 5622 篇 artifacts 已重現 0.3857／0.3880；✅ regression tests；⏳ official files2rouge conformance（僅保留 SciTLDR 時） |
| `src/eval/oracle.py` | metric-specific greedy reference；canonical fail-loud、source-order search、`max_words` 明確化 | F-1, F-21 | ✅ 46-test correctness suite；兩 primary partitioned Gate 2 run 尚未執行 |
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
