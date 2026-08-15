# 專案總索引

## 最新 checkpoint（2026-08-15）

作者端已批准 GovReport-centered 選項 A。GovReport 現為唯一 primary quality domain；
Multi-News 的既有 D3b 負結果保留為 boundary-condition evidence，不再是 final-test
promotion 的共同 primary。機器可讀 addendum、claim matrix、非搜尋 evidence preregistration
與仍鎖定的 final protocol 已版本化；protected dev-test/test 仍未解鎖。

E1 官方 evaluator 已完成：Proposed 在 681-row frozen dev 的官方尺度 macro
`0.458257`，相對 full-source SBERT+MMR `+0.004569`，95% CI
`[+0.002467,+0.006680]`、`p=0.000020`；R-L 單項未顯著。E2 也已完成：Proposed
cold/warm `176.03/10.52s`，SBERT+MMR `177.81/12.45s`，NSGA-II
`222.61/56.74s`；54 measured repetitions identity 全部一致。E3 五個已預註冊
route/provenance ablations 亦已完成，五個 macro 差異 CI 全正且 Holm-20 均通過；
681/681 rows 可行。完整表見 [`E1_OFFICIAL_EVALUATOR_STATUS.md`](E1_OFFICIAL_EVALUATOR_STATUS.md)、
[`E2_COST_SCALING_STATUS.md`](E2_COST_SCALING_STATUS.md) 與
[`E3_ROUTE_PROVENANCE_ABLATION_STATUS.md`](E3_ROUTE_PROVENANCE_ABLATION_STATUS.md)。
Pre-test freeze audit 曾發現 frozen policy 對 materialization 與簽字先後形成循環；
2026-08-16 Stage A 由提出請求的作者轉述老師／作者端核准兩階段順序後，official test policy
已在零 prediction／零 score 下 materialize。973/973 rows 保留、0 exclusions、2 個
U+FFFD source characters 原樣記錄。下一步是凍結 fail-closed one-shot runner。

Multi-News full-dev D2 已依事前凍結規格完成 14/14 selector candidates，只讀 dev
3,935 rows。固定 S02b 候選時 Greedy-TFIDF macro `0.328077` 勝 NSGA-II+TF-IDF
`0.322615`、最佳 TF-IDF-MMR `0.321744` 與最佳 SBERT-MMR `0.316612`；200-row
pilot 的 MMR 優勢未外推。GovReport D2 最佳 TF-IDF-MMR λ=0.7 macro `0.446154`，
對 Greedy `+0.028293` 且 multiplicity-corrected paired gate 通過，但仍低 strongest
baseline `0.006614`。依預註冊採 task-profile policy；兩 profile 均不具 promotion 資格。
完整 evidence 見 `runs_v2/d2_selector_full_dev_v1/analysis/paired_summary.json`。

D3a 已在任何新分數前凍結，兩個 profile 各 14 個 candidate-capacity／lexical-salience／
weighted-RRF 案；固定 200-row reference-blind dev pilot 28/28 已完成。Multi-News
bigrams+position `+0.002559`；GovReport lexical×0.5 `+0.009648` 最佳。Gov 三個正訊號
macro CI 全正且 Holm-104 通過，但 pilot 的 332-opportunity correction 未通過，故只取得
full-dev 資格。版本化 analyzer 已按原規則固定 finalists；28-endpoint full-dev
confirmation 與 streaming 16-worker runner 已在分數前凍結並完成。Multi-News winner
仍低 PacSum `0.002036`；GovReport winner 高 SBERT+MMR `0.003655`、CI 全正且 Holm-28
通過，但 selection correction 未通過。兩 profile 均不得 promotion；dev-test/test 未讀。

D3b 是停止條件前最後一個 dev 實驗，且已在任何組合分數前凍結並完成。GovReport
bigrams+position+lexical×0.5 macro `0.457404`，對 SBERT+MMR `+0.004636`，
95% CI `[+0.002507,+0.006745]`、Holm-8 `p=0.000240`、Bonferroni-340
`p=0.013600`，通過全部條件。Multi-News bigrams+position+graph×2 macro
`0.330417`，仍低 PacSum `0.001323`；R-2 `−0.008565` 且 CI 全負。雙 primary
promotion 因此失敗，配置搜尋已停止；不得新增 grid 或讀 dev-test/test。這段是 v1
雙-primary gate 的歷史判定；v2 已縮窄為 GovReport-centered，且 official evaluator、
成本／scaling 與 route/provenance ablation 均已完成。下一步是解決 freeze-policy ordering、
建立 immutable test policy 與 exact execution package，再由老師與完整作者群兩階段簽字。

`Greedy / candidate-matched SBERT-MMR / NSGA-II` 的 selector swap 已完成第一版
接線與 3-row 真實資料 smoke，三者逐列 candidate、salience、similarity、coverage
hash 全部相同；加入兩 primary partition／greedy-reference／GovReport data-layer／A1/D1/D2 runners／D1 inventory／document-aware position、Gate 2 audits 與 F-51～F-63 guards 後 409 tests passed。SBERT centroid-only 與 full-source SBERT-MMR
baseline 亦已接入 shared baseline CLI。凍結的 200-row matched pilot 顯示 MMR
相對 Greedy 的 R-1／R-2 分別 `+0.01488`／`+0.01472` 且 Holm 校正後顯著；
NSGA-II 五 seed mean 三指標均低於 Greedy，且選句 Jaccard 僅 `0.639`；selector
當時決策因此暫定為 MMR main／Greedy reference／NSGA-II comparator；上方 D2 已更正。這仍是 diagnostic，
兩 primary frozen-dev non-PLM 各 23/23、PLM 各 27/27、greedy reference 6/6 與
64-endpoint paired finalists 均已完成。S02b 對 Multi-News P08 macro `−0.003663`、對
GovReport SBERT+MMR `−0.034906`，兩者 Holm 校正後仍顯著；Gate 2 quality gate 失敗。
其後 D2/D3a/D3b dev redesign 已完成並停止搜尋；E1/E2/E3 已完成，現在只整理 freeze package。單一規格來源見
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

### 目前進度速覽（2026-08-15）

| | 狀態 |
|---|---|
| Phase 1 程式契約 | 🟡 **大部分完成** —— route/provenance/shared objectives、兩套歷史資料 policy 與 partition、official evaluator parity、成本量測與 final ablation 已完成；只剩 final-output policy／freeze audit 與簽字；見 `ACTION_PLAN.md` Phase 3d |
| 測試 | ✅ Stage A policy 後完整回歸 **473 passed／5 subtests passed**；compileall 通過。PR #17 clean-clone Linux CI **454 passed / 5 skipped** |
| **baseline** | ✅ **矩陣與 diagnosis 完成；quality gate 失敗** —— 兩 primary non-PLM 各 23/23、PLM 各 27/27、greedy reference 6/6、paired finalists 均完成。Multi-News S02b 平均 headroom `1.41%`，對 P08 macro `−0.003663`；GovReport headroom `8.64%`，對 SBERT+MMR `−0.034906`。兩個 paired loss 均 Holm-significant，selection-aware wins 0；其後已完成 D2/D3a/D3b，不再重開 baseline grid。見 `GATE2_BASELINE_STATUS.md` |
| Gate 2 prereg | ✅ `gate2-baseline-matrix-v1` 已在正式 baseline scores 前凍結：每資料集 non-PLM 23／PLM 27 candidates，runner 只允許 frozen dev；dev-test/test 禁止 |
| 新 matched-selector pilot | 🟡 **200-row reference-blind diagnostic 完成** —— MMR vs Greedy：R-1 +0.01488、R-2 +0.01472（兩者 Holm-significant），R-Lsum +0.00770（校正後不顯著）。NSGA-II 五 seed mean 均低於 Greedy，selection Jaccard 0.639；已降為 comparator。完整 evidence：`evidence/selector_comparison_pilot_v1_summary.json`、`evidence/selector_comparison_nsga5_stability.json` |
| D2 full-dev selector | 🔴 **兩 primary 各 14/14 + paired analysis 完成，promotion失敗** —— Multi-News Greedy `0.328077` 最佳且低 P08 `0.003663`；GovReport TF-IDF-MMR λ=0.7 `0.446154` 最佳、顯著高 Greedy，但低 full-source SBERT+MMR `0.006614`。採 task-profile policy；未讀 dev-test/test |
| 舊新 pipeline 診斷 | 🟡 F-18/F-19 的 `length_normalized` 相對 Lead 為 R-1 +0.001465、R-Lsum +0.001906，但 R-2 −0.011423；它不是新 matched-selector pilot，不能混併數字 |
| ✅ **主線 selector F-17** | 已採 option 1：所有 lower-bound document infeasibility 都寫成完整 prediction row；candidate capacity、Greedy、GRASP、NSGA-II 與無 eligible sentence 共用 contract，upper-bound／config bug 仍 fail loud。F-17 的 5,621-row governed regression 通過；現行全套為 459 local passed、PR #17 Linux 454 passed／5 skipped。實測 5,620 feasible／1 recorded infeasible |
| 資料 | ✅ 兩 primary validation 已由 pinned source 建立並凍結 policy／fingerprint／dev-dev-test manifests；GovReport 973 rows（CRS 361／GAO 612）。test split 在 freeze 簽字前禁止讀取 |
| A1 長度協定 | ✅ 兩 primary 的唯一一次 dev-test 已完成：Multi-News 凍結 200–250 words（macro `0.310382`），GovReport 凍結 500–650（`0.393415`）；兩者對三案的 Holm-adjusted `p=0.000600`。Multi-News Greedy 3/1,686 floor shortfalls 完整保留（F-23）；GovReport floor-bound lexical Greedy 仍輸 Lead/Random（F-24） |
| D1 敏感度 | ✅ 原 screens／matched route ablations／paired analysis 完成。12/12 route endpoints 通過 Holm 與 186-opportunity correction，semantic/graph 暫留；Multi-News 對 Lead R-2 顯著 `−0.008336`。其後強 baseline、D2/D3a/D3b 均完成；D1 不重開、不看 dev-test |

> ⚠️ **「契約完成」不等於「方法有效」。** F-18 已有第一次 diagnostic
> validation pilot；D3b 已在 GovReport 取得 multiplicity-corrected 正證據，但
> Multi-News 未通過，因此仍不能解鎖 protected split 或宣稱跨資料集方法有效。
> 每個 route 的刪除條件（`ARCHITECTURE.md` §5.3／§5.4／§7.3）都仍然有效。
>
> 各項稽核發現的 legacy／新 pipeline 現況對照見
> `CODE_AUDIT_IEEE_Access.md` **§0.0 狀態表**。

### 實驗資料集到底跑哪些（v2 addendum：2026-08-10）

| 類別 | 決定 |
|---|---|
| **唯一 primary** | **GovReport**：已完成已預註冊的 E1～E3 dev evidence；freeze audit 與老師／完整作者群簽字後，才可能一次性解鎖 official test |
| **boundary evidence** | **原版 Multi-News**：保留既有 frozen-dev D3b 負結果，明示 R-2 邊界；不做新搜尋、不跑 dev-test/test、不以 clean sensitivity 改寫結論 |
| **不納入 v2 主線** | CNN/DailyMail、SciTLDR-AIC、Multi-News+、bad-retrieval-removed、PubMed、Multi-XScience；任何新增都算修改已凍結 dataset matrix，須另行決策 |
完整執行規則以 `ACTION_PLAN.md` §2.1 與
`configs/data_policies/govreport_centered_repositioning_v2.json` 為準；§2.0 保留為 v1 歷史，
不得用它解鎖 Multi-News 或其他資料集。

---

## 文件導覽

| 檔案 | 用途 | 什麼時候看 |
|---|---|---|
| **`ACTION_PLAN.md`** | **要做什麼、什麼順序、完成定義** | ⭐ **日常執行看這份** |
| `ARCHITECTURE.md` | Target Architecture v2、schema、模組介面與 freeze gate | 要動資料層、候選路徑、objective 或 selector 時 |
| `CLAUDE.md` | AI 協作規則、已驗證事實、程式硬規則 | AI agent 開工前必讀 |
| `paper_revision_plan_IEEE_Access.md` | 研究流程治理、10 個 P0、投稿合規、新架構設計 | 需要「為什麼要這樣做」的完整論證 |
| `CODE_AUDIT_IEEE_Access.md` | 已驗證的程式缺陷 + 實測數字 + 已套用的修正 | 需要證據、需要引用數字 |
| `GATE2_BASELINE_STATUS.md` | 最新 baseline family 分數、證據與未完成項目 | 追 Gate 2 進度時 |
| `E3_ROUTE_PROVENANCE_ABLATION_STATUS.md` | 五個 GovReport route/provenance matched ablations | 判斷各模組能否列為貢獻時 |
| `GOVREPORT_PRETEST_FREEZE_RECOMMENDATION.md` | E1～E3 freeze 建議、政策衝突與 Stage A 簽核欄 | 準備請老師／作者核准時 |
| `GOVREPORT_TEST_POLICY_STATUS.md` | Official-test Stage A policy 與 canonical health | Runner freeze 前核對 |
| `STRATEGY_ASSESSMENT.md` | 可行性評估、病因診斷、資料集選擇、兩份計畫對照 | 需要判斷「還有沒有救、主場選哪裡」 |
| `REPOSITIONING_RECOMMENDATION.md` | D3b 停止決策、凍結數字與已批准的 GovReport-centered 選項 | 查決策沿革時 |
| `GOVREPORT_CLAIM_MATRIX_V2.md` | v2 主張、證據與禁止事項 | 寫摘要／結果／結論前 |
| `ICACT_IEEE_ACCESS_EXTENSION_MATRIX.md` | ICACT→IEEE Access 逐頁技術差異 | 核對 conference extension；DOI／獎項證明仍待補 |
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
# 2026-08-09 F-51～F-53／Multi-News PLM checkpoint

Multi-News PLM 已完成 27/27。winner 是 PacSum-SBERT P03（macro `0.331458`），仍比
non-PLM PacSum P08 低 `0.000282`；最佳 full-source SBERT-MMR λ=0.7（`0.322581`）低
S02b `0.005496`。execution-only cache 已通過 3,935-row cold/warm exact audit，F-53 verifier
另驗 25 個 cached runs 共 98,375 hits、contract/digest/dependencies。dev-test/test 未讀。
詳見 `GATE2_BASELINE_STATUS.md` 與 F-51～F-53 evidence。

GovReport PLM 亦已完成 27/27。winner 是 full-source SBERT-MMR λ=0.9（macro
`0.452768`），只高 LexRank `0.001148`、高 S02b `0.034906`；前一差距未做 paired
inference，不能宣稱顯著勝出。27 runs 的 18,387 row accesses 全數通過 F-53，
dev-test/test 未讀。
