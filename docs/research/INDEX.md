# 專案總索引

## 最新 checkpoint（2026-08-06）

`Greedy / candidate-matched SBERT-MMR / NSGA-II` 的 selector swap 已完成第一版
接線與 3-row 真實資料 smoke，三者逐列 candidate、salience、similarity、coverage
hash 全部相同；加入兩 primary partition／greedy-reference／GovReport data-layer／A1/D1 runners／D1 inventory／document-aware position regression 後 343 tests passed。SBERT centroid-only 與 full-source SBERT-MMR
baseline 亦已接入 shared baseline CLI。凍結的 200-row matched pilot 顯示 MMR
相對 Greedy 的 R-1／R-2 分別 `+0.01488`／`+0.01472` 且 Holm 校正後顯著；
NSGA-II 五 seed mean 三指標均低於 Greedy，且選句 Jaccard 僅 `0.639`；selector
決策因此固定為 MMR main／Greedy reference／NSGA-II comparator。這仍是 diagnostic，
尚未完成 full Multi-News validation、full-source MMR、PacSum 或 GovReport，
因此 Gate 2／Gate 3 仍未通過。單一規格來源見
[`SELECTOR_COMPARISON_PROTOCOL.md`](SELECTOR_COMPARISON_PROTOCOL.md)。

抽取式摘要研究 —— ICT Express 拒稿後改投 **IEEE Access** 的修訂工作。

- **ICACT**：已投稿，獲 outstanding paper award
- **ICT Express**：已拒稿（ICTE-D-26-00238），四位審稿人意見在 `Reviewer.docx`
- **現在**：修正中，目標 IEEE Access

---

## 📌 先讀這個

**現在最重要的兩件事：**

1. 🔴 **legacy Multi-News 當家配置**在同資料、同 evaluator 下沒有贏過本地 Lead —— R-2 輸 0.0048、R-Lsum 輸 0.0021；這是 test-tuned artifact 的診斷，不是新論文結果
2. 🔴 `runs/tuning_experiments/` 的 11 個 run 全是 5,622 筆 test，且用來選設定；相關 legacy 主結果不可作新稿證據

→ 詳見 `ACTION_PLAN.md` 的 Phase −1；這兩件事已被版本化為研究治理前提。**後續不得再以 legacy 結果直接改寫或支撐新論文。**

### 目前進度速覽（2026-08-08）

| | 狀態 |
|---|---|
| Phase 1 程式契約 | 🟡 **大部分完成** —— route 獨立排名、provenance 進 selector、shared objective/constraint evaluator、source-vs-candidate length feasibility、Pareto artifact、兩 primary canonical/frozen-policy/dev-partition 已完成；published-protocol parity、正式成本 pilot 與 validation-frozen output policy 仍未完成；CNN/DailyMail 是 Gate 3 後的 optional 工作；見 `ACTION_PLAN.md` Phase 1 |
| 測試 | ✅ **343 local tests 全過**（2026-08-08）；PR #15 Linux CI 綠燈（2026-08-05），CI 已接 GitHub Actions |
| **baseline** | 🟡 **Phase 2 進行中** —— shared contract、CLI、Lead、Random、TextRank／LexRank、full-source SBERT centroid／MMR 程式已接線；既有 Multi-News full-validation artifacts 只能作 historical diagnostics。PacSum、partitioned SBERT run、GovReport 方法 runs、paired significance 與正式兩-primary矩陣未完成，**Gate 2 未過** |
| 新 matched-selector pilot | 🟡 **200-row reference-blind diagnostic 完成** —— MMR vs Greedy：R-1 +0.01488、R-2 +0.01472（兩者 Holm-significant），R-Lsum +0.00770（校正後不顯著）。NSGA-II 五 seed mean 均低於 Greedy，selection Jaccard 0.639；已降為 comparator。完整 evidence：`evidence/selector_comparison_pilot_v1_summary.json`、`evidence/selector_comparison_nsga5_stability.json` |
| 舊新 pipeline 診斷 | 🟡 F-18/F-19 的 `length_normalized` 相對 Lead 為 R-1 +0.001465、R-Lsum +0.001906，但 R-2 −0.011423；它不是新 matched-selector pilot，不能混併數字 |
| ✅ **主線 selector F-17** | 已採 option 1：所有 lower-bound document infeasibility 都寫成完整 prediction row；candidate capacity、Greedy、GRASP、NSGA-II 與無 eligible sentence 共用 contract，upper-bound／config bug 仍 fail loud。F-17 的 5,621-row governed regression 與目前 289-test suite 全過；實測 5,620 feasible／1 recorded infeasible |
| 資料 | ✅ 兩 primary validation 已由 pinned source 建立並凍結 policy／fingerprint／dev-dev-test manifests；GovReport 973 rows（CRS 361／GAO 612）。test split 在 freeze 簽字前禁止讀取 |
| A1 長度協定 | ✅ 兩 primary 的唯一一次 dev-test 已完成：Multi-News 凍結 200–250 words（macro `0.310382`），GovReport 凍結 500–650（`0.393415`）；兩者對三案的 Holm-adjusted `p=0.000600`。Multi-News Greedy 3/1,686 floor shortfalls 完整保留（F-23）；GovReport floor-bound lexical Greedy 仍輸 Lead/Random（F-24） |
| D1 敏感度 | 🟡 19 組／90 個 effective config paths 已盤點；27 個 dev-only Greedy configs 已於分數前預註冊。本階段不看 dev-test。document-scoped position 已修（F-25）；candidate budget/RRF 在兩路 active context 測（F-26） |

> ⚠️ **「契約完成」不等於「方法有效」。** F-18 已有第一次 diagnostic
> validation pilot，但目前沒有任何新配置通過 Gate 2 或取得可投論文的正式證據。
> 每個 route 的刪除條件（`ARCHITECTURE.md` §5.3／§5.4／§7.3）都仍然有效。
>
> 各項稽核發現的 legacy／新 pipeline 現況對照見
> `CODE_AUDIT_IEEE_Access.md` **§0.0 狀態表**。

### 實驗資料集到底跑哪些（決策凍結 2026-07-30；狀態覆核 2026-08-02）

| 類別 | 決定 |
|---|---|
| **必跑 primary** | **原版 Multi-News + GovReport**：validation 做 baseline／方法選擇，configuration freeze 後才跑 official test |
| **必跑 sensitivity** | Multi-News frozen U+FFFD clean sensitivity：5,549 rows，與 5,621-row main 作 paired validation；它不是 Multi-News+ 或 bad-retrieval-removed |
| **延後可選** | CNN/DailyMail：只有 primary 過 Gate 3 且資源允許，才以 frozen method 跑 official test 11,490 作 appendix sanity；不阻塞主線 |
| **v1 不跑** | SciTLDR-AIC、Multi-News+、bad-retrieval-removed、PubMed、Multi-XScience |

完整執行規則以 `ACTION_PLAN.md` §2.0 為單一狀態來源；其他文件只解釋原因，不得自行擴張資料集。

---

## 文件導覽

| 檔案 | 用途 | 什麼時候看 |
|---|---|---|
| **`ACTION_PLAN.md`** | **要做什麼、什麼順序、完成定義** | ⭐ **日常執行看這份** |
| `ARCHITECTURE.md` | Target Architecture v1、schema、模組介面與 freeze gate | 要動資料層、候選路徑、objective 或 selector 時 |
| `CLAUDE.md` | AI 協作規則、已驗證事實、程式硬規則 | AI agent 開工前必讀 |
| `paper_revision_plan_IEEE_Access.md` | 研究流程治理、10 個 P0、投稿合規、新架構設計 | 需要「為什麼要這樣做」的完整論證 |
| `CODE_AUDIT_IEEE_Access.md` | 已驗證的程式缺陷 + 實測數字 + 已套用的修正 | 需要證據、需要引用數字 |
| `STRATEGY_ASSESSMENT.md` | 可行性評估、病因診斷、資料集選擇、兩份計畫對照 | 需要判斷「還有沒有救、主場選哪裡」 |
| `REPO_CLEANUP.md` | 專案整理 | Phase 0 |

### 文件權威順序

遇到衝突時依下列順序處理：

1. `paper_revision_plan_IEEE_Access.md`：研究標準、Go/No-Go、投稿合規的規範來源。
2. `ARCHITECTURE.md`：技術架構、schema、模組介面與 objective 啟用規則的單一規格來源；在 validation pilot 前仍是候選規格。
3. `ACTION_PLAN.md`：任務狀態與執行順序；只有通過 DoD 才能勾選完成。
4. `CODE_AUDIT_IEEE_Access.md`：commit `1b9fe6f` 與 legacy artifacts 的證據快照；不是目前程式正確性的保證。
5. `STRATEGY_ASSESSMENT.md`：由證據推導的策略判斷；情境估計不是測量結果。
6. `REPO_CLEANUP.md`：整理提案；其中 move/delete/tag 指令均須另行確認後才執行。
7. `CLAUDE.md`：協作護欄，只引用上面文件，不應另立數字真相。

### 三份分析文件的分工

它們分工如下；不再用「不同作者版本」互相比較，研究結論一律收斂到主計畫與行動清單：

- **`paper_revision_plan_IEEE_Access.md`** —— 最完整的研究流程治理。**這份是主幹。**
  抓到 test-set 調參、CNN/DM split 誤用、Stage-1 top-K 與實作不符、ICACT extension 合規等。
- **`CODE_AUDIT_IEEE_Access.md`** —— 稽核證據快照。
  Lead 與兩個 legacy ROUGE-Lsum 數字已重新核對；oracle/headroom/位置分析中使用非官方協定或未保存腳本者已降級為 diagnostic。
- **`STRATEGY_ASSESSMENT.md`** —— 戰略判斷。
  headroom 分析、為什麼輸給 Lead 的病因、主場資料集選擇。

> 濃縮時的建議：以 `paper_revision_plan_IEEE_Access.md` 的架構為骨幹，
> 把 `CODE_AUDIT` 的實測數字填進對應的 P0 條目，
> 把 `STRATEGY_ASSESSMENT` 的 F-0 與病因診斷放到最前面的決策段。

---

## 關鍵數字速查

| 項目 | 數值 |
|---|---|
| Legacy ExpB vs Lead（Multi-News 5622，同一新 evaluator） | `0.4352/0.1405/0.3880` vs `0.4331/**0.1453**/**0.3901**`；兩者皆只作診斷 |
| Full benchmark 的 ROUGE-L → Lsum | 0.2014 → **0.3857**；ExpB 則是 0.2019 → **0.3880** |
| ⛔ **所有 ROUGE-Lsum 數字已過期**（2026-07-30, PR #9） | 分句器換成共用 Punkt，實測 R-Lsum 位移 **+0.0032**（R-1/R-2 為 +0.0000，不受影響）。上面每個 Lsum 值重算前不得引用。詳見 `CLAUDE.md` §2 開頭 |
| **新 pipeline 首次量測（2026-08-03, diagnostic）** | Lead `0.4332/0.1468/0.3940`；系統最佳（greedy+`length_normalized`）`0.4347/0.1354/0.3960` —— 但**長度括弧顯示 R-1/R-Lsum 的領先完全由多用 10.4 字解釋**，等長下 Lead 三項全勝。詳見 `CODE_AUDIT_IEEE_Access.md` F-18 |
| **目標函數 > 最佳化演算法** | `mean`→`length_normalized` 值 **+0.0232** R-Lsum（10 分鐘）；greedy→NSGA-II 值 **+0.0039**（322 分鐘）。約 6 倍差距 |
| **系統低於 Random（`mean` 配置）** | Random `0.3788` R-Lsum vs greedy `0.3728`、NSGA-II `0.3767`；改 `length_normalized` 後三項均超過 |
| 新 pipeline 選句與 Lead 重疊率 | **24.3%–27.6%**（全量、`sentence_id` 比對）—— 漏斗確實打開，但未轉化為品質 |
| 系統選句與 Lead 重疊率（legacy） | **61.7%**；腳本已版本化並重現，但仍是 legacy artifact 上的 diagnostic；**與上一列不同 split／樣本，不可相減** |
| 系統選句命中 legacy greedy reference 的比例 | **22.8%**；不是 official oracle recall，須以 validated oracle 重做 |
| Headroom（200 篇抽樣 diagnostic） | Multi-News 0.152 / CNN-DM 0.171 / SciTLDR 0.190；非全集，不可引用為正式結果 |
| 論文的 SciTLDR "oracle" 0.136 | 是 `rouge_scores` 欄位全句平均，**不是 oracle** |
| PLM 計時 | 載入遠大於推論（兩次量到 78% / 93%，**佔比不穩定**）；純推論 BERT/RoBERTa 比值 **≈1.0**（1.04× / 1.02×）為穩定結論。須依鎖定 protocol 重測 |

> 上列 diagnostic 的重現腳本在 `scripts/audit/`，
> 用法與已重現輸出見該目錄的 `README.md`。**版本化 ≠ 可作論文結果** ——
> 仍需官方 split、freeze config、多 seed 與 paired bootstrap。

---

## 目錄說明

| 目錄 | 說明 |
|---|---|
| repo 根目錄 | 研究程式碼 |
| `ICACT/` | ICACT 得獎論文與投稿檔 |
| `ICT_Express/` | 被拒的 19 頁投稿 PDF |
| `Reviewer.docx` | 四位審稿人完整意見 |
| `cnn_dailymail/` | 資料 |
| `初步提案/`、`結案報告/`、`論文/` | 與技術判斷無關，可略過 |

---

## 快速開始

```bash
cd metaheuristic-summarization
```

按目前多句內部協定重算某個 run（ROUGE-Lsum；不可自動視為 published-protocol parity）：

```bash
.venv/Scripts/python.exe -m src.pipeline.evaluate --pred runs/<run>/predictions.jsonl --gold data/processed/<dataset>_<split>.jsonl --out runs/<run>/metrics_fixed.csv --protocol multisentence_lsum
```

計算 metric-specific greedy reference smoke（不是 exact upper bound；目前的
`max_words` 是空白切詞；正式 Gate 2 必須依 frozen partition 過濾）：

```bash
.venv/Scripts/python.exe -m src.eval.oracle --input tests/fixtures/multi_news_validation_diagnostic_sample.jsonl --max_words 220 --limit 3
```

> ⚠️ `runs/` 底下的既有數字全部視為 invalid，不要寫進論文。
