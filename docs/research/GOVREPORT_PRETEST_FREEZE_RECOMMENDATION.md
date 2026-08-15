# GovReport pre-test freeze recommendation

## 決議摘要（2026-08-15）

**2026-08-16 更新：Stage A 已核准並完成；Stage B 已獲原則核准，但只在 exact runner
與 dry run 全部通過後才自動啟動。**

品質搜尋已在 D3b 後停止，E1 official evaluator、E2 controlled cost/scaling、E3
route/provenance ablation 全部完成且通過各自的預註冊判準。這代表「方法是否值得送 final
gate」已有肯定答案；但 test policy、exact final command/runner、execution commit 與完整
作者簽字尚未成立。任何人在這些欄位完成前讀取 GovReport test membership、payload、
references 或 scores，都會違反既有 frozen protocol。

## 已完成證據

| Gate | 結果 | 判定 |
|---|---|---|
| E1 official evaluator | Proposed official macro `0.458257`；對 SBERT+MMR `+0.004569`，95% CI `[+0.002467,+0.006680]`，`p=0.000020`；R-L 單項未顯著 | 通過，保留 GovReport superiority 假說到 untouched test |
| E2 cost/scaling | Proposed cold/warm `176.03/10.52s`；SBERT+MMR `177.81/12.45s`；NSGA-II `222.61/56.74s`；54/54 measured identity 一致 | 通過；主方法無 wall-time 劣勢證據，但 RAM 略高 |
| E3 ablation | 五個 C01−ablation macro CI 全正，Holm-20 均 `0.000400`；每案 681/681 feasible | 通過；semantic、graph、capacity-matched non-lexical routes、weighted RRF、selector provenance 均可保留 |
| Regression | `473 passed / 5 subtests passed`；compileall pass | 通過（Stage A 後） |
| ICACT content extension | 六頁 PDF 已逐頁核對，方法、Eq. (1)–(13)、Tables 1–6 與舊 claims 已定位 | 技術 audit 完成；DOI／獎項證明／similarity report 仍屬投稿行政待辦 |

Machine-readable evidence index：
`configs/preregistrations/govreport_pretest_evidence_index_v1.json`。Freeze verifier 必須回報
`evidence_completion_status=E1_E2_E3_complete`、`test_split_accessed=false`。

## Frozen scientific decision

- Primary domain：GovReport long single-document, multi-sentence summarization。
- Boundary evidence：Multi-News；不跑 Multi-News test，也不隱藏其 R-2 負結果。
- Candidate：`C01_combined_salience_route_weight`。
- Selector：TF-IDF MMR λ=0.7；NSGA-II 僅作 matched comparator。
- Routes：lexical + semantic + sparse graph；weighted RRF lexical=0.5。
- Candidate budget：route top-40、requested reservation 20、total 80。
- Length：500–650 whitespace words。
- Frozen config SHA-256：`c2088ca4dd794ba08c4c9544d4d75f323c4c3fd749dbac7cdf4b611da60bd481`。
- Authoritative final evaluator：Stanza `tokenize,mwt` + Perl ROUGE-1.5.5
  `-c 95 -r 1000 -n 2 -m`。
- Primary final comparison：Proposed vs full-source SBERT+MMR λ=0.9，100,000 paired
  bootstrap，seed 20260830；test 後不得改規則或救分。

## 發現的 frozen-policy ordering conflict

目前兩份已凍結規格形成循環：

1. `govreport_centered_repositioning_v2.json` 把「immutable final test policy」列為完整
   作者簽署 final freeze **之前**的必要證據；
2. 同檔 protected policy 與 `govreport_centered_final_evaluation_v1.json` 又要求在 unlock／
   簽字前，`govreport_test_v1.json` 必須不存在，且不得讀 test membership/payload。

正式 test policy 必須包含 membership count、exclusions、source checksums 與 fingerprint；
不讀 official test 就不可能誠實填入。因此現行 one-stage 簽字流程無法同時滿足兩條規則。
Freeze verifier 將此狀態明列為
`policy_sequence_status=blocked_by_frozen_contract_ordering_conflict`，不會假裝可執行。

## 建議採用的兩階段簽核（需老師／完整作者群明確同意）

### Stage A — 授權 materialize policy，不授權跑分

全體簽署只允許指定 data steward 讀 official test membership/payload，建立 canonical policy、
exclusion manifest、checksums、fingerprint 與 health report。這一階段禁止執行任何 proposed
或 baseline prediction，禁止計算 ROUGE。

### Stage B — Final execution freeze

Policy 完成後，作者群再核對 exact policy hash、scientific code commit、dependency/hardware
manifest、九系統 commands、output paths、official evaluator 與 inference command。全部簽字
後，one-shot runner 才能解鎖。若執行失敗，只能 exact checkpoint resume；看到任何 test
score 後不得修改 candidate、baseline、長度、evaluator、排除列或統計方法。

## 現在仍缺的完成條件

- [x] 老師／作者端同意兩階段順序（由提出請求的作者轉述；未偽造外部簽名）。
- [x] Stage A authorization 已記錄於 `govreport_test_authorization_v1.json`。
- [x] `configs/data_policies/govreport_test_v1.json` 與 test canonical health evidence 已在零分數狀態建立並 pin。
- [ ] Test-only preprocessing／nine-system one-shot runner／official evaluator adapter 綁定該 policy，完成 fail-closed dry run。
- [ ] Exact scientific code commit、environment、commands 與 output locations 凍結。
- [~] Stage B 原則核准已由提出請求的作者轉述；仍須 runner／environment／commands freeze 完成才可啟動。

上列任何一項未完成，`ready_for_test` 必須保持 `false`。

## Stage A 簽核欄（目前全部留白）

| 角色／作者 | 同意只 materialize policy、不跑分 | 日期 | 簽名／可稽核核准紀錄 |
|---|---|---|---|
| Teacher / corresponding author |  |  |  |
| Shih-Wei Yang |  |  |  |
| Bo-Yu Chen |  |  |  |
| Shao-Chi Kuan |  |  |  |
| Hau-Ching Chen |  |  |  |
| Sy-Yen Kuo |  |  |  |
| Jiann-Liang Chen |  |  |  |

簽署者也必須確認：ICACT prior publication 將在 IEEE Access 新稿中正式引用；ICT Express
被拒稿不是 prior publication；Multi-News 負結果不會被刪除；final test 失敗也會照實回報。
