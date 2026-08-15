# IEEE Access 重新定位建議書

> **2026-08-15 執行狀態**：選項 A 已定案，無須再選 A/B/C。配置搜尋已停止；
> E1～E3 與 ICACT 六頁技術 audit 已完成。Pre-test freeze 建議另見
> `GOVREPORT_PRETEST_FREEZE_RECOMMENDATION.md`。2026-08-16 兩階段順序已由提出請求的
> 作者轉述核准，GovReport test policy 在零 prediction／零 score 下完成；目前只進行
> fail-closed runner 與 exact execution freeze，尚未產生正式分數。

## 決策摘要（2026-08-09）

**目前不得進 dev-test 或 test，也不建議以「跨資料集品質優於強 baseline」投稿 IEEE
Access。** 最後一個預註冊的 D3b dev 實驗只有 GovReport 通過，Multi-News 未通過；
依事前停止條件，配置搜尋在此停止。

這不是說整個研究沒有結果。最新證據支持一個較窄、但誠實且可延續 ICACT／ICT
Express 稿件的結論：**training-free、可審計的多路抽取架構在長篇單文件 GovReport
上有顯著優勢，但沒有形成跨 task profile 的一致品質優勢。** Multi-News 的主要失敗是
ROUGE-2，而不是所有指標都退步。

完整機器可讀證據：
`runs_v2/d3b_cross_profile_combination_v1/analysis/paired_summary.json`。

## 定案記錄（2026-08-10 作者端決策）

**選項 A 已由提出請求的作者端定案**：GovReport 改為 IEEE Access
修訂稿的唯一主要品質資料集，Multi-News 保留為 boundary-condition
dataset。這不改寫原雙-primary gate 的失敗，也不解鎖 protected split。
老師／完整作者群的 final freeze 簽字仍待完成。

已新增：

- `configs/data_policies/govreport_centered_repositioning_v2.json`：機器可讀角色與主張 addendum。
- `docs/research/GOVREPORT_CLAIM_MATRIX_V2.md`：人可讀 claim matrix。
- `configs/preregistrations/govreport_centered_evidence_completion_v1.json`：official evaluator、成本／scaling、route/provenance ablation 的非搜尋預註冊。
- `configs/preregistrations/govreport_centered_final_evaluation_v1.json`：仍鎖定的 one-shot final protocol。

預註冊 evidence 已完成；現在的下一步是解決 freeze policy 順序並簽核，不是再選資料集或繼續調參。

## 凍結證據

| Primary profile | D3b macro | Strongest local baseline | Δ macro | 95% CI | Holm-8 | Bonferroni-340 | 決策 |
|---|---:|---:|---:|---:|---:|---:|---|
| Multi-News（multi-document） | 0.330417 | PacSum TF-IDF 0.331740 | −0.001323 | [−0.003372, +0.000697] | 0.412796 | 1.000000 | 失敗 |
| GovReport（single long document） | 0.457404 | full-source SBERT+MMR 0.452768 | +0.004636 | [+0.002507, +0.006745] | 0.000240 | 0.013600 | 通過 |

Multi-News 的 component 差異是 R-1 `+0.000391`、R-2 `−0.008565`、R-Lsum
`+0.004207`。R-2 的 CI `[−0.010924,−0.006221]` 全負，不能解釋成單純統計 power
不足；R-Lsum 的 CI `[+0.002192,+0.006203]` 全正，表示方法確實改善了部分長範圍內容
覆蓋，但犧牲了局部 bigram precision。

GovReport 的 component 差異是 R-1 `+0.004097`、R-2 `+0.007804`、R-Lsum
`+0.002006`；R-1／R-2 CI 全正，R-Lsum CI 跨 0。整體 macro 在 100,000 次 paired
bootstrap、跨歷次 340 個 endpoint 的 selection correction 後仍通過。

兩個 runner 都只讀 frozen dev；`dev_test_accessed=false`、`test_split_accessed=false`。
第一次 GovReport attempt 因 Codex Windows sandbox child-process 權限失敗，沒有形成分數，
已完整保存在 `govreport/attempts/attempt_01_sandbox_permission/`；成功重試未覆寫它。

## 建議的論文定位

### 建議主張

將主張由「metaheuristic 方法在一般抽取式摘要上品質最好」改為：

> 一個 training-free、task-profile-aware、provenance-preserving 的多路抽取框架，提供
> 可稽核的 lexical／semantic／graph evidence fusion 與可替換 selector；它在長篇
> 單文件 GovReport 上對強 local baseline 有 multiplicity-corrected 優勢，並透過
> Multi-News 負面結果界定適用邊界。

這仍是 ICACT／ICT Express 工作的修訂延伸，不是換成無關的新方法：三路候選生成、
多目標品質控制與 metaheuristic selector 都保留在同一系統中；差別是新稿不能再把
NSGA-II 寫成必然的主方法。它應作為 matched-input comparator，用實驗證明何時昂貴的
多目標搜尋不值得。實際主 selector 可依 task profile 使用 Greedy 或 MMR。

### 可保留的貢獻

1. **Training-free task-profile policy**：不做摘要標註微調，且不是把同一 selector
   強套所有資料集。
2. **Provenance-preserving weighted fusion**：lexical、semantic、graph route 的來源、
   rank contribution、融合權重與最後選句可以逐句追溯。
3. **Matched selector evidence**：Greedy、MMR、NSGA-II 接收相同 candidate、salience、
   similarity、coverage input；因此可以把 selector 效果和上游改動分開。
4. **嚴格的負面邊界**：GovReport 通過，Multi-News 因 R-2 明確失敗。這比只挑有利
   dataset 更可信，但必須在標題、摘要與結論中明說 domain dependence。
5. **研究治理與可重現性**：canonical schema、資料 manifest、split guard、失敗 attempt
   保留、逐篇 paired evidence 與多重比較校正。

### 不可再宣稱

- 不可宣稱跨資料集 SOTA 或全面勝過 PacSum／SBERT+MMR。
- 不可把 greedy reference 稱為精確 oracle。
- 不可宣稱 NSGA-II 帶來最佳品質；目前它更慢且 full-dev 未勝。
- 不可用舊 test-tuned 11 runs、舊 evaluator 數字或外部論文不同 split 的數字作勝負。
- 不可把 Multi-News macro 未顯著寫成「與 PacSum 相同」；point estimate 仍較低，且
  R-2 顯著較差。

## 投稿選項

### A. 長篇單文件／可審計 training-free 摘要（建議）

把 GovReport 定為主要適用情境，Multi-News 保留為 boundary-condition dataset。這條路
最貼近現有正證據，也最能延續原稿。不過它會改變原先「兩 primary 都要通過」的投稿
矩陣，因此必須由老師與作者明確批准新的 data policy／claim matrix，另做預註冊後才可
解鎖任何 protected split。

即使批准，IEEE Access 前仍需補齊：published evaluator parity、cold/warm 成本與 scaling、
provenance/route 消融的最終表格、限制與威脅、以及 ICACT extension 的差異表。這些是
證據補完，不是再用 dev 搜尋分數。

### B. Quality–cost／auditability 系統論文

若不願縮窄資料情境，可把主要貢獻改為可控成本、選句 provenance 與 selector trade-off，
品質採 non-inferiority 而非 superiority。但 non-inferiority margin 尚未事前凍結，不能用
目前結果事後補定；需要新的研究決策與完整成本實驗。IEEE Access 機會低於 A，但仍比
假裝全面品質領先合理。

### C. 停止 IEEE Access 方法稿

若老師要求的硬條件仍是兩個 primary 都顯著贏 strongest baseline，現有方法已依停止
規則判定 No-Go。此時應停止投入同一搜尋空間，將工程與負面結果整理為技術報告、資料／
重現性資源，或等待真正不同的研究假設；不能再從同一 dev 反覆找組合。

## 下一個人工簽字點

資料集方向已定案，不再重選 A/B/C。下一個簽字點是：當 official
evaluator parity、cold/warm cost/scaling、route/provenance ablation 與 ICACT extension
matrix 完成後，由老師與完整作者群檢查 final freeze package，決定是否
解鎖 GovReport official test 的唯一次執行。

在該簽字完成前，**dev-test 與 test 都維持鎖定**。
