# 總體可行性評估 —— 設計是否有救、主場在哪、文件如何分工

> 初稿日期：2026-07-26 ｜ 程式／資料狀態覆核：2026-08-02
> 對象問題：「是不是設計本身就有問題？分數這麼低還投得出去嗎？我們的優勢在哪？架構還有救嗎？」
> 本文件是策略 memo，不是數字權威來源。Legacy Multi-News 的 ROUGE/Lead 已重現。
> headroom、位置與 oracle overlap 的腳本**已版本化**至 `scripts/audit/`
> 並由該處重跑確認一致，但仍使用小樣本與非官方 oracle，維持 diagnostic 標籤。
> 2026-08-06 更新：frozen 200-row matched-selector pilot 顯示 MMR 在 R-1／R-2
> 顯著優於 Greedy；NSGA-II 五 seed mean 均低於 Greedy且選句 Jaccard 0.639。
> selector 路線因此改為 MMR main／Greedy reference／NSGA-II comparator。這只是
> selector gate；兩 primary 的 strongest non-PLM baseline 都已勝 proposed S02b。Multi-News
> PLM 27/27 之後最強仍是 non-PLM PacSum P08；GovReport PLM winner MMR λ=0.9 只高
> LexRank `0.001148`、卻高 S02b `0.034906`。是否值得投稿仍取決於 dev 優化與完整 paired gate。

---

## 先回答你的三個問題

| 你的問題 | 我的答案 |
|---|---|
| 設計本身是不是有問題？ | **是，而且我找到了確切原因** —— 但這是可診斷、可修的具體缺陷，不是「整個想法錯了」 |
| 分數這麼低投得出去嗎？ | **現在這個版本投不出去。** 但問題不是「分數低」，是「分數低且沒有解釋」。修好後有機會 |
| 架構還有救嗎？ | **有救，但要動核心，不是補實驗。** 見第 3 節 |

**順帶更正**：你說「我們用兩個資料集」—— 其實論文用了**三個**：CNN/DailyMail、SciTLDR-AIC、Multi-News。這件事很重要，因為三個資料集的體質完全不同（見第 2 節）。

---

## 1. 設計到底哪裡有問題？—— 精確診斷

### 1.1 先看「有多少空間可以贏」

以下是探索性抽樣的 **greedy-reference headroom estimate**，不是 exact extractive oracle，也不是投稿級上界。它只能用來形成「哪些資料集值得做 pilot」的假設，不能證明理論上能贏多少。

| 資料集 | Lead R-1 | Greedy-reference R-1 | **估計差距** | greedy-reference 位置中位數 | 前 25% 佔比 |
|---|---|---|---|---|---|
| Multi-News | 0.4383 | 0.5901 | **0.1518** | 0.46（分散） | 31.3% |
| CNN/DailyMail | 0.4003 | 0.5709 | **0.1707** | **0.21（極度靠前）** | **57.4%** |
| SciTLDR-AIC | 0.1979 | 0.3876 | **0.1897** | 0.49（分散） | 33.0% |

*（各 200 篇；Multi-News 全 5622 篇的 Lead 為 0.4331，與此處一致）*
*（各資料集抽樣 200 篇，非全集；腳本見 `scripts/audit/dataset_headroom.py`。
CNN/DM 的本地 0.4003 與文獻 0.4042 接近，只能作 sanity signal，不能證明實作與文獻 evaluator 完全一致。）*

這些估計暗示三個資料集可能仍有改善空間，但必須先版本化分析程式、對齊正式協定並在 validation 重跑，才能形成結論。

### 1.2 那為什麼贏不了？—— 這才是真正的病因

我比對了三種選句策略在 Multi-News 上**實際挑了文件哪些位置的句子**：

| 誰在選 | 句子位置中位數 | 前 25% 佔比 |
|---|---|---|
| **Legacy greedy reference（探索性）** | 0.462 | 31.3% |
| Lead（最笨的 baseline） | 0.082 | 86.9% |
| **你們的系統** | **0.143** | **67.6%** |

再看重疊率：

| 指標 | 數值 |
|---|---|
| 系統選的句子中，**Lead 也會選**的比例 | **61.7%** |
| 系統選的句子中，**legacy greedy reference 也會選**的比例 | **22.8%** |

### 🔴 Legacy Multi-News 診斷：目前行為接近「昂貴版 Lead」

系統的探索性行為分布（中位數 0.143、前段 67.6%）較靠近 Lead（0.082 / 86.9%），而不是 legacy greedy reference（0.462 / 31.3%）。
六成以上的選句與 Lead 重疊，只有兩成命中該 greedy reference；這些數字尚待版本化重跑。

這是 F-0 的合理病因假說，且與候選池程式路徑一致；但 61.7%／22.8% 分析必須在版本化腳本與新 validation protocol 上重做，才能成為正式證據。
跑了 NSGA-II、BERT、PageRank，繞了一大圈，最後選出來的東西和「抄前面幾句」高度雷同。

### 1.3 為什麼會變成這樣？—— 根因在特徵設計

看 `configs/1_Base_NSGA2.yaml`：

```yaml
features:
  weights:
    importance: 0.8    # TF-ISF v1 —— 未做長度正規化，實質是「句子越長分越高」
    length:     0.2    # 直接獎勵長句
    position:   0.2    # 直接獎勵靠前的句子  ← lead prior
candidates:
  k: 40
  sources: [score]     # ← 候選池「只」用上面這個分數挑
```

問題鏈：

1. `position` 特徵直接給前面的句子高分 —— **這就是一個內建的 lead prior**
2. `length` 與 v1 的 TF-ISF 都偏好長句
3. **候選池（`sources: [score]`）完全由這個 lead-biased 分數決定**
4. NSGA-II、BERT、Graph 三軌**全部只在這個已經偏前的候選池裡運作**（研究主計畫 P0-03）
5. → 後段候選在進入最佳化前可能已被排除；這是假設，須用版本化的 candidate recall@K 分析確認

**病因確定：不是最佳化演算法不好，是候選池在最佳化開始前就已經把答案濾掉了。**

> 這是好消息。因為這是一個**具體、可修、可驗證**的缺陷，
> 而不是「meta-heuristic 做摘要這個想法行不通」。

---

## 2. 主場應該選哪裡？—— 用數據決定

### 2.1 三個現有資料集的體質

| 資料集 | 判決 | 理由 |
|---|---|---|
| **CNN/DailyMail** | 🔴 **不作主場** | lead bias 強且摘要短；論文 0.351 與 published Lead-3 0.4042 來自不同 split/evaluator，不能當公平差距，只能視為風險訊號 |
| **Multi-News** | 🟡 **可留，但不是最佳** | 探索性抽樣顯示 greedy-reference headroom 約 0.1518、句子位置較分散；須版本化重跑後才可形成結論 |
| **SciTLDR-AIC** | 🟡 **尚未有有效勝負；v1 不跑** | legacy Lead-1 與論文 0.234–0.239 使用錯誤的 reference 串接／非官方 evaluator；只有日後重新納入時才需 official single-sentence conformance 後重跑 |

### 2.2 SciTLDR 不適合當主場

SciTLDR 的舊勝負尚未成立，而且它也不適合當主戰場：

- 官方協定是**只抽一句**（one source sentence）
- 只選一句 → **redundancy 目標恆為 0，coverage 目標退化** → 你的多目標框架整個沒有意義
- 用一個讓自己核心方法失效的資料集當主場，說服力反而最低

官方表中的 SciTLDR-AIC **PACSUM（同為無監督）= 28.7**；論文的 23.91 來自不合規 evaluator，不能直接判定勝負，但即使只作風險訊號也不樂觀。

### 2.3 我的建議：主場要滿足「兩個條件同時成立」

從我的 headroom 分析可以歸納出，你們需要的資料集必須同時滿足：

1. **lead bias 較弱**（validated reference candidates 分散在全文，不是集中在開頭）
2. **摘要是多句、長度預算大**（coverage / redundancy 目標才有意義）→ 否則你的多目標框架退化

| 資料集 | 條件 1（lead bias 弱） | 條件 2（多句長摘要） | 綜合 |
|---|---|---|---|
| CNN/DailyMail | ❌ 極強 lead bias | ❌ 只有 3 句 | 🔴 最差 |
| SciTLDR-AIC | ✅ 弱 | ❌ 官方只抽 1 句 | ⚪ 與核心多句方法不匹配，v1 排除 |
| Multi-News | 🟡 中等 | ✅ 245 words | 🟢 可用 |
| **GovReport** | ✅ **長政府報告，資訊分散全文** | ✅ **摘要很長** | 🟢🟢 **理想** |
| **PubMed / arXiv** | ✅ 科學長文 | ✅ 長摘要 | 🟢🟢 **理想** |
| BillSum | ✅ 法案文本 | 🟡 中等 | 🟢 可考慮 |

**→ 建議依研究主計畫加入 GovReport 作第二主 benchmark。探索性 headroom 只提供 pilot 動機，不構成獨立證明。**

補充建議：**PubMed / arXiv 也很值得考慮**，它們是長文件摘要的標準 benchmark，lead bias 弱，且有大量可比較的文獻數字。

### 2.4 建議的資料集配置

| 角色 | 資料集 | 說明 |
|---|---|---|
| **主 benchmark 1** | **GovReport** | 長單文件、長摘要，多目標 coverage 與 scaling 有意義；PubMed 僅為 GovReport pilot 失敗時的備案 |
| **主 benchmark 2** | **Multi-News（原版）** | 多文件，可與既有文獻直接比較；validation 已從 pinned 原始資料重建並保存 document boundaries，train/test 仍待生成 |
| Required data-quality sensitivity | **Multi-News frozen U+FFFD clean sensitivity** | 與 5,621-row main 在共同 5,549 rows 作 paired analysis；不等於 retrieval-cleaned variant |
| Optional sanity | CNN/DailyMail | 只有 primary 過 Gate 3 且資源允許才跑 frozen official test；不阻塞主線 |
| Excluded from v1 | SciTLDR-AIC | 單句協定與核心多句方法不匹配；目前不跑，重新納入才需 official conformance |
| Reserve | Multi-News bad-retrieval-removed／Multi-News+、PubMed | 目前不跑；不能與已 frozen 的 U+FFFD clean sensitivity 混稱 |

---

## 3. 架構還有沒有救？—— 有，但要動核心

### 3.1 必須修的三件事（依重要性）

> ✅ **2026-07-26 更新：這三項的「程式契約」都已實作完成**，見 `ACTION_PLAN.md` Phase 1c/1d
> 與 `CODE_AUDIT_IEEE_Access.md` §0.0 狀態表。
> **但「實作完成」不等於「有效」** —— 三項的效益都還沒量測，仍須通過 validation pilot。
> 以下保留原始診斷與驗收條件，狀態另以標籤標示。

**① 打破候選池的 lead bias —— 這是最關鍵的一刀**　【✅ 契約已實作／⏳ 效益未驗證】

原診斷：候選池由 lead-biased 分數單獨決定 → 後續 route 無法找回已被排除的句子。

已實作：
- ✅ lexical / semantic / sparse graph 各自**在完整輸入上排名**，不再被共同候選池預先截斷
- ✅ `route_top_k` 只定義提案深度、`min_per_route` 是明確保留額、RRF 只能在 proposal union 與 guard 內填 total cap
- ✅ position / document / section strata coverage guard 可獨立設定，並標為 `guard:*` 而非第四個語意 route
- ✅ 不可行的保留額組合會 fail loud，不會默默犧牲某個 route

⏳ **仍未完成的驗收**：候選池對 validated oracle／greedy-reference 的 **recall@K**。
22.8% 是 legacy exploratory 值；新 pipeline 的數字**還沒量過**，提升門檻也還沒訂。

> 這仍是最關鍵的一刀 —— 但現在的問題已從「有沒有做」變成「做了有沒有用」。

**② 重新設計 position 特徵**　【🟡 部分處理】

原診斷：`position` 是單調遞減的 lead prior；legacy greedy-reference 的位置中位數探索值是 0.46。

- ✅ position 已不再控制唯一候選入口，可作為獨立的 coverage guard
- ⬜ **MVP config 目前把 `position` 權重設為 0，等於迴避而非解決**。
  是否需要非單調或 learned prior，仍待 validation pilot 決定

**③ 真的把語意訊號接上（F-3／研究主計畫 P0-04）**　【✅ 已接上／⏳ 增量未驗證】

原診斷：Stage 2 的「PLM 語意」其實是 TF-IDF。

- ✅ semantic route 已是完整輸入上的獨立候選 route，使用 pinned sentence-encoder revision
- ✅ **selector 真的收到 provenance**：`selector.salience_source: rrf_fusion`
  （實測：改變 semantic 分數會改變最終選句；用 `base_score` 則不會）
- ✅ `base_score` / `membership_only` 保留為可消融對照組
- ⏳ **仍未驗證**：unique candidate recall、quality delta、quality-cost 是否有增量。
  **若三者都沒有,semantic route 必須刪除** —— 這個刪除條件仍然有效

### 3.2 修好之後能到哪裡？—— 誠實的期望值

以下只能當情境門檻，不是由現有結果外推的預測：

| 情境 | 可能結果 |
|---|---|
| Go 情境 | 在 locked validation/test 上對強 no-task-training baseline 有顯著品質增益，或在預先定義 non-inferiority margin 內形成成本 Pareto 優勢 |
| 邊界情境 | 只對 Lead 小幅改善、對 PacSum/SBERT+MMR 無優勢 → novelty 與方法價值仍不足，需重新定位 |
| No-Go 情境 | 品質不優、成本更高、核心 route marginal contribution 為零 → 停止新方法投稿 |

是否足以投 IEEE Access 不能只由 R-1 點估計決定；必須同時通過 technical soundness、統計、novelty、成本與可重現性 gate。

### 3.3 你們真正的優勢在哪（這是可以賣的）

不要再賣「品質最好」。可以賣的是：

1. **Zero-training / 無標註資料**：不需要任何 summarization 標註就能運作。在低資源語言、專業領域（法律、醫療、政府文件）這是實際價值
2. **可配置成本／CPU-capable tier**：lexical route 加 deterministic selector 可在 CPU 運行；含 sentence encoder 的 full tier 是否需要 GPU、是否值得，必須由 cold/warm end-to-end 成本實測
3. **可審計的選句 provenance**：每句可追溯到 statistical / graph / semantic route；這比一般端到端 baseline 更直接，但不能宣稱 neural 系統做不到解釋
4. **顯式多目標 trade-off 控制**：coverage、redundancy、length 與 cost 的 policy 可重現；優勢在控制介面明確，不是 neural 模型原理上做不到控制
5. **長文件處理策略**：可避免把全文一次送入 512-token encoder，但 sentence encoder 仍有截斷、route 仍有計算複雜度，必須以 scaling 實驗證明

**第 3 點和第 4 點最有機會形成 IEEE Access 的方法敘事**，但目前只是待驗證的架構假設；仍需消融、matched baseline 與使用情境證明。

---

## 4. 文件整合與證據分工

`paper_revision_plan_IEEE_Access.md` 已定為研究標準與投稿 gate 的規範來源；本文件只保留策略判斷與探索性診斷，不與主計畫競爭版本權威。

### 4.1 研究主計畫提供的治理與風險項目

| 項目 | 說明 | 我驗證的結果 |
|---|---|---|
| **P0-01 在 test set 上調參選模型** | `quick_tune.ps1` 等直接在 `multi_news_test.jsonl` 跑 ExpA/B/C，論文再從中挑 best config | ✅ **已證實**：11 個 tuning/ablation run **全部都是 5622 篇 = test set**。這是 textbook test-set overfitting，**嚴重度不亞於我的 F-0** |
| **P0-02 CNN/DM 用 validation 卻對比文獻 test** | Table 1/4 對應 13,368 筆 | ✅ **已證實**：所有 CNN/DM run 都是 **13368 筆 = validation split**（官方 test 是 11,490） |
| **P0-03 Stage-1 top-K 與實作不符** | 三軌被共同候選池預先截斷，不是各自 top-K | ✅ 證實，而且**這正是我第 1.3 節診斷出的病因根源** |
| **SciTLDR 官方數字** | oracle 52.4 / PACSUM 28.7 / BERTSUMEXT 36.2 | 提供 conformance target；舊 23.91 協定錯誤，尚不能直接判定輸給 PACSUM |
| **P0-10 資料完整性** | 零句文件、test_4241 有 3295 句 | 具體且正確 |
| **ICACT extension 合規** | similarity < 35%、必須引用 ICACT | 我完全沒想到這一層，但這是**投稿硬性門檻** |
| **Go/No-Go gate、manifest、統計協定** | 整套研究流程治理 | 比我的重構計畫更完整、更專業 |

### 4.2 本策略 memo 補充的診斷

| 項目 | 為什麼重要 |
|---|---|
| 🔴 **F-0：legacy Multi-News 未勝 Lead** | ExpB 在同資料、同內部 evaluator 下 R-2 −0.0048、R-Lsum −0.0021；CNN/DM 因 split/evaluator 不同，尚無有效勝負 |
| 🔴 **病因診斷：系統 61.7% 與 Lead 重疊、只有 22.8% 命中 greedy reference** | 腳本已版本化（`scripts/audit/selection_diagnostics.py`）並重現；但仍跑在 legacy artifact、200 篇抽樣、非 official oracle，只支持「優先檢查候選池 lead bias」 |
| **headroom / lead bias 量化** | 腳本已版本化（`scripts/audit/dataset_headroom.py`）；200 篇抽樣，用來選 pilot，不可直接宣稱理論空間 |
| **完整 5622 篇的 ROUGE-Lsum** | full benchmark = 0.3857；ExpB = 0.3880，兩者不可混稱同一次 run。⛔ 兩值皆為舊 regex 分句器下的量測（2026-07-30 PR #9 已換共用 Punkt，實測位移 +0.0032），重算前不得引用 |
| **計時分解**：載入遠大於推論、純推論比值 ≈1.0 | 腳本已版本化（`scripts/audit/plm_timing.py`）。**載入佔比在兩次執行間為 78% 與 93%，不穩定，不可引用特定百分比**；只有「推論比值 ≈1.0」是穩定結論。須依鎖定 runtime protocol 重測 |
| legacy greedy references：SciTLDR 3 句 0.5136、Multi-News 約 0.59 | 只能診斷，非 exact upper bound、非 official protocol，不可直接引用 |
| **pymoo mutation 實測**：per-individual 1.0、per-gene 1/n_var≈0.02 | 直接回答 R4 的疑問 |
| **Phase 1 canonical 主路徑已重構** | 403 local tests、PR #15 Linux CI、snapshot、shared objectives、candidate provenance、兩 primary frozen policy/partitions 與 A1/D1 runners 已通過。D1 route evidence、兩 primary non-PLM 各 23/23、PLM 各 27/27、greedy reference 6/6 與 paired finalists 完成；published-protocol parity、clean sensitivity 與正式成本 reporting 尚未完成；CNN/DM 是 Gate 3 後 optional |

### 4.3 我必須修正自己的一個地方

研究主計畫 P0-05 指出 SciTLDR 官方協定是**單句抽取 + 以最高 R1 選同一 reference**，官方 oracle R1 = 52.4。

**我的 0.5136 不是官方協定** —— 我用的是「3 句 + 串接後的 reference」。兩個偏差（句數多 → 分數高；串接 reference → 分數低）剛好互相抵消，數字看起來接近但**不可直接與官方 52.4 比較**。

若正式論文決定保留 SciTLDR stress test，必須採用官方 files2rouge 協定，並先重現 52.4 作為 evaluator conformance test；若不保留 SciTLDR，這項工作不阻塞主線。

### 4.4 判決

| 面向 | 較佳的版本 |
|---|---|
| 研究流程治理（split、manifest、Go/No-Go、統計） | `paper_revision_plan_IEEE_Access.md` |
| 問題涵蓋廣度（10 個 P0） | `paper_revision_plan_IEEE_Access.md` |
| 投稿合規（ICACT extension、IEEE Access 規則） | `paper_revision_plan_IEEE_Access.md` |
| 新架構設計（provenance fusion、adaptive routing） | `paper_revision_plan_IEEE_Access.md` |
| **已重現的 legacy 數字** | `CODE_AUDIT_IEEE_Access.md`，仍須看每筆證據標籤 |
| **策略與 exploratory diagnostics** | 本文件，不作數字權威 |
| 目前程式狀態 | 以 `ACTION_PLAN.md` Phase 1 與 `CODE_AUDIT_IEEE_Access.md` §0.0 為準；canonical 主路徑已有完整內部 regression，但 Phase 1 仍因外部協定、多資料集與 freeze 工作未完成 |

**→ 以研究主計畫為規範，只有通過 evidence-level 檢查的結果才能回填；探索性數字不得因寫入本文件而升格。**

---

## 5. 合併後的行動順序

### Phase −1：先做決策（1 天）—— 已與研究主計畫 §15 對齊

在投入任何重構之前，先接受兩個事實：

- [ ] 承認 **F-0 的正確範圍**：legacy Multi-News 沒有贏過同協定 Lead；CNN/DM 尚無公平勝負，研究主計畫 §6.1 的 redesign gate 已觸發
- [ ] 承認 **P0-01**：11 個 tuning/ablation runs 在 Multi-News test 上選設定，受此流程影響的 legacy 主結果**不能用於新稿**
- [ ] 決定路線：建議研究主計畫的「路線 A（方法型）」，並把「打破候選池 lead bias」列為優先驗證的假設

### Phase 0–6：依研究主計畫與 `ACTION_PLAN.md` 執行

研究主計畫目前已包含 Phase −1（決策與凍結）及 Phase 0（專案治理）到 Phase 6，直接依 `ACTION_PLAN.md` 執行。

**在其中插入以下由我的診斷衍生的必做項目：**

| 插入位置 | 新增項目 |
|---|---|
| Phase 1（correctness refactor） | canonical 主路徑的 403 local tests、PR #15 Linux CI、snapshot、shared objectives、兩 primary frozen-policy preflight、partition enforcement、exact batched Greedy additions與 F-51～F-62 guards 已完成；published-protocol parity 與 validation-frozen output policy仍未完成 |
| Phase 2（baseline validation） | 兩 primary non-PLM 各 23/23、PLM 各 27/27、greedy reference 6/6 與 paired finalists 已完成。Multi-News S02b/P08 headroom `1.41%/3.30%`；GovReport S02b/SBERT+MMR `8.64%/22.98%`。S02b 對兩 adversarial winners 的 macro paired loss 均 Holm-significant，Gate 2 quality gate 失敗 |
| Phase 3（方法實驗） | screens、matched route ablations 與 paired inference 已完成。12/12 route endpoints 通過 Holm p=`0.002400` 與 186-opportunity p=`0.037196`，semantic/graph 暫留；semantic-direct selector 刪除。Multi-News 主要是 selector/salience 瓶頸；GovReport 同時有 candidate coverage 與 selector bottleneck，下一步只用 dev redesign，不能 promotion |
| Phase 1–2 | 重建 GovReport 與原版 Multi-News 作兩個 primary benchmarks；frozen U+FFFD clean 作 paired sensitivity，external retrieval-cleaned variants 與 PubMed 只作備案 |
| Phase 4（locked test）之前 | **先在 validation 上確認贏過 Lead**。沒贏就不要解鎖 test |

### 最重要的一個中途檢查點

> **先在 validation 重建位置、Lead overlap 與 candidate recall baseline；新架構應相對該同 split baseline 有實質改善。**
>
> 若這兩個診斷在新 validation pipeline 沒有改善，是強烈 redesign 訊號；它們不是 ROUGE 必然不改善的充分必要條件。

---

## 6. 最後的總體判斷

**還有沒有機會投 IEEE Access？有，但不是現在，而且不能只補實驗。**

- **現在直接投** → 我判斷**幾乎確定被拒**。test-set 調參（P0-01）+ 輸給 Lead（F-0）任一被抓到就結束
- **只修 evaluator、補 baseline、改寫文字後投** → 仍然 No-Go。因為修正後會**更清楚地顯示輸給 Lead**
- **照合併計畫執行（含打破 lead bias 的核心改動）** → **有實質機會**。前提是中途檢查點通過

**時間**：研究主計畫的 Phase −1 至 6 原始工作量約 6–9 週；納入核心方法改動與新資料集後，保守估計 **8–12 週**。

**最後一句實話**：核心概念仍有價值，但最新證據更嚴格：S02b 在 Multi-News 輸 P08 `0.003663`，在 GovReport 輸 MMR λ=0.9 `0.034906`；兩個 macro loss 的 paired CI 全負且 Holm-significant。semantic／graph matched 增量雖成立，整體方法仍弱於 strongest baseline。現在必須在 dev 優化 Multi-News 的 selector/salience，以及 GovReport 的 candidate coverage + selector；若搜尋空間掃完仍無顯著優勢，就依停止條件重新定位，不能直接投稿。
