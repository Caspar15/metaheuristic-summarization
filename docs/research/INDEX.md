# 專案總索引

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

### 目前進度速覽（2026-08-02）

| | 狀態 |
|---|---|
| Phase 1 程式契約 | 🟡 **大部分完成** —— route 獨立排名、provenance 進 selector、shared objective/constraint evaluator、source-vs-candidate length feasibility、Pareto artifact、canonical schema 與 frozen-policy preflight 已完成；published-protocol parity、GovReport 資料層、正式成本 pilot 與 validation-frozen output policy 仍未完成；CNN/DailyMail 是 Gate 3 後的 optional 工作；見 `ACTION_PLAN.md` Phase 1 |
| 測試 | ✅ **217 tests 全過**（PR #10 合併後）；CI 已接 GitHub Actions |
| **baseline** | 🟡 **Phase 2 已開始** —— shared baseline contract、CLI 與 Lead 已進 master；兩個 primary 尚無正式 Lead 結果，TextRank／LexRank／PacSum／sentence-encoder／Random 尚未完成，Gate 2 未過 |
| 新 pipeline 的實測結果 | 🔴 **零** —— 還沒跑過任何 validation 評估 |
| 資料 | 🟡 Multi-News validation main/clean policy、fingerprints 與 manifests 已凍結；GovReport 與其餘 splits 尚未完成 |

> ⚠️ **「契約完成」不等於「方法有效」。** 目前沒有任何一個新配置被評估過。
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
| 系統選句與 Lead 重疊率 | **61.7%**；腳本已版本化並重現，但仍是 legacy artifact 上的 diagnostic |
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

計算 greedy oracle reference（不是 exact upper bound；目前的 `max_words` 是空白切詞）：

```bash
.venv/Scripts/python.exe -m src.eval.oracle --input data/processed/multi_news_test.jsonl --max_words 245 --limit 300
```

> ⚠️ `runs/` 底下的既有數字全部視為 invalid，不要寫進論文。
