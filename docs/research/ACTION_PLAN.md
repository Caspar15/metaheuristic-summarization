# ACTION PLAN —— 到底要做什麼

## 2026-08-08 freeze 前自治執行狀態

- [x] 在任何新配置分數產生前，將 frozen Multi-News validation 以 reference-blind
      SHA-256 排序、固定 seed `3407` 凍結為 dev `3,935`（70.005%）與
      dev-test `1,686`（29.995%）；兩者互斥且完整覆蓋 5,621 rows。manifest：
      `configs/validation_partitions/multinews_validation_dev_v1.json`，file SHA-256
      `e61405482cda203c0bd50dda3e958986b11a51b48124129e68617f97ce9e42ee`。
- [x] proposed-method 與 baseline runner 均先驗證完整 canonical validation／frozen
      data policy，之後才依 manifest ID 過濾；canonical row 的 split 仍誠實保留
      `validation`，不把 dev/dev-test 偽裝成上游資料 split。每個 run 另寫
      `partition_preflight.json`。
- [x] 分割與 runtime enforcement 已版本化於
      `scripts/audit/freeze_validation_partitions.py`、`src/data/partitions.py`；manifest
      漂移、輸入 SHA 不符、遺失 ID、重複 ID 都 fail loud。
- [x] GovReport 已由官方作者 archive 重建；官方 validation 974 筆中，空 reference
      的 CRS `98-228` 依 frozen manifest 排除，留下 973 筆。資料政策綁定 archive／
      canonical SHA、dataset fingerprint、CC-BY-4.0、section/paragraph metadata 與
      0 個 U+FFFD；dev 681／dev-test 292 已在任何方法分數前凍結，manifest SHA-256
      `7a15ffbb87abe690fe4e72a1e0daf27bf34b3a3293371983ae8e362d06e2717e`。
- [x] A1 已完成兩 primary 的 reference-only 統計、預註冊、dev 與唯一一次 dev-test；
      Multi-News 依預註冊規則選定 200–250 words，GovReport 選定 500–650 words；
      A2 greedy-reference correctness、A3 GovReport 資料層與兩 primary 的 B 階段
      partition freeze 已完成；Gate 2 baseline 矩陣與 dev search 尚未完成。
      **test split 仍為硬禁止；到 freeze 簽字前不執行。**
- [x] A1 的 Multi-News dev reference-only 統計已完成（3,935 rows：mean 215.52、
      median 218、p75 260），且四個候選協定、三個 cheap method、dev/dev-test
      勝出與 Holm/tie 規則已在任何候選 system score 前預註冊於
      `configs/preregistrations/a1_length_contract_v1.json`。GovReport 數值只可在其
      data layer/partition 凍結後依同一 rule family 實例化，不能照抄 Multi-News。
- [x] A1 GovReport addendum 已在任何 method score 前凍結（681 dev rows：mean 567.08、
      median 572、p25 500、p75 653）。因作者論文未提供 extractive word cap，候選為
      IQR band 500–650、median cap 570、paper-mean cap 550、p75 cap 650；若 Holm 後
      無勝者，採最簡單的無下限 median-cap。見
      `configs/preregistrations/a1_length_contract_govreport_v1.json`。
- [x] A1 執行器已在分數前版本化：由 frozen preregistration 產生 exact resolved config，
      對 Lead／Random／lexical Greedy 寫逐 run evidence、prediction SHA、逐篇 selected-
      indices digest、dependency versions、per-example ROUGE 與 `search_log.jsonl`；已完成
      的 logical candidate 不可第二次觀察 dev-test。4 個 pure-function regression 通過。
- [x] A1 Multi-News dev 四候選 × 三方法已完整跑完（每個 3,935 rows，12/12 run
      evidence 完成，test 未存取）。cross-method macro 排名：legacy floor 200–250
      `0.314581` > p75 cap260 `0.305578` > max-only250 `0.304407` > median cap220
      `0.298191`。同 250 cap 的 Lead／Random 完全相同；差距來自 Greedy：floor 版本
      242.60 words／`0.310353`，no-floor 200.45 words／`0.279831`。
- [x] A1 Multi-News 唯一一次 dev-test 已完成（每候選 1,686 rows；12/12 evidence；
      test 未存取）。legacy 200–250 的 cross-method macro `0.310382` 居首；相對
      max-only250／median220／p75-cap260 的 paired-bootstrap mean difference 分別
      `+0.009185/+0.015510/+0.008022`，三個 95% CI 全為正，Holm-adjusted
      `p=0.000600`，故依預註冊規則選定 200–250 words。
      凍結機器可讀 policy：`configs/length_policies/multinews_v1.json`。
- [~] F-23：floor-bearing Greedy 在 dev 有 4/3,935、dev-test 有 3/1,686 rows 未達
      effective minimum 200；均依 F-17 完整記錄並以 all-row denominator 計分，未放寬、
      補句或刪列。A1 已選定 floor，但這仍是待 selector feasibility 改善並須正式報告的
      limitation；不得把 floor 僅描述成外部公平長度上限。
- [x] A1 GovReport dev 四候選 × 三方法已完整跑完（每個 681 rows，12/12 evidence，
      test 未存取）。排名：IQR floor500-cap650 `0.392821` > p75 cap650 `0.328979` >
      median cap570 `0.324044` > paper-mean cap550 `0.322049`。no-floor Greedy 僅
      165.6–184.5 words／macro `0.175990–0.178859`；floor 使其到 638.4 words／
      `0.370385`。這是 dev 階段結果；最終一次性決選見下一項。
- [x] A1 GovReport 唯一一次 dev-test 已完成（每候選 292 rows；12/12 evidence；test
      未存取）。IQR 500–650 的 cross-method macro `0.393415`，相對 median570／
      p75-cap650／paper-mean550 的 paired mean difference 為 `+0.065281/+0.060642/
      +0.067218`，95% CI 全為正、Holm-adjusted `p=0.000600`，依規則選定 500–650。
      dev→dev-test 僅 `+0.000595`，292/292 feasible。機器可讀 policy：
      `configs/length_policies/govreport_v1.json`。
- [~] F-24：即使 floor 拉長，GovReport lexical-only Greedy 在 dev-test 的 `0.375048`
      仍低於 Lead `0.394815` 與 Random `0.410382`（dev 同方向）；這是 objective/
      candidate salience 的 reality
      warning，不可由 length tuning 掩蓋。它不直接否定尚未加入的 semantic/graph route。
- [x] D1 所有 canonical runtime knobs 已逐模組盤點為 19 組／90 個 dotted paths，
      證據與驗證器分別在 `docs/research/evidence/d1_effective_tunable_inventory.json`、
      `scripts/audit/effective_tunable_inventory.py`。27 個 dev-only Greedy screening configs
      已在任何 D1 score 前預註冊；本階段不看 dev-test。
- [x] D1 runner 與兩 dataset base configs 已在分數前版本化；每個 family 可獨立提交，
      只接受 frozen dev，CLI 沒有 split 參數，所有成功／失敗均寫 method evidence 與
      `search_log.jsonl`。resolved parent delta、dataset guard 與 hash tests 全過。
- [x] F-25 已修：base-feature position 新增明確 `scope=document`，從 canonical
      `document_id/document_position` 重置；缺 provenance、非連續 position 或 records
      長度不符均 fail loud。兩文件 golden `[1,0,1,0]`、pipeline snapshot 與完整
      **343 tests passed**，現在才允許跑預註冊 position variant。
- [~] F-26：A1 lexical-only context 下 `min_per_route=20`、`total=60` 與 RRF constant
      結構上不能改變候選 membership/rank；這不影響 A1 長度決選，但不能拿來排序候選
      預算旋鈕。D1 會在 lexical+graph 的兩路 active context 各動一次。
- [x] D1 Multi-News lexical/objective family 已完成（12/12 configs、每個 3,935 frozen
      dev rows；dev-test/test 未存取）。L10 全文候選 macro `0.320912`、相對 base
      `+0.010559` 居首；coverage 加倍 `+0.006216`、document position `+0.002568`。
      但 L10 仍低於同協定 Lead macro `0.326291`，且約為 base wall time `4.86×`，不能
      晉級或宣稱優勢。完整表見 `D1_SENSITIVITY_STATUS.md`。
- [x] F-27/F-28 已修：D1 runner 現可驗證式 `--resume`，外部 timeout 的 partial 與
      failure attempt 會保留；diagnostics v2 分開 actual selector pool 與 provenance
      pool。L11 首次 attempt 因外層 60-minute timeout 失敗、重試成功，search log 保留
      **12 success + 1 failure**。完整 regression **345 tests passed**。
- [~] F-29：取消 top-40 prefilter 是目前最大正向 OFAT，但 selector pool 由平均
      `35.95` 增至 `81.85`、最大 `3,318`，不是可接受的最終解。待兩資料集的 graph／
      semantic／candidate-budget family 判定能否用受控 pool 回收品質。Multi-News 已有
      正向答案：lexical+graph G02 macro `0.324305`，平均 pool `48.03`／最大 `60`，比
      全文 lexical L10 高 `0.003393`；GovReport 與 semantic 尚待量測。
- [x] F-30：GovReport L10 暴露 Greedy 對每個候選重算兩次 full-source facility
      coverage。pre-fix attempt `639.47 CPU s` 後封存 152-row prefix；更正後全 dev
      平方 proxy 約 `0.766 h`（原 `4.05 h` 推論作廢，F-31）。已實作等價 batched
      additions，targeted 43／完整 357 tests 通過；L10/L11 resume 完成。post-F-30
      L00 reference-blind audit 比對 681 rows，逐篇 selected indices **0 差異**，digest
      同為 `8273f162...d7982`，且未讀 dev-test/test。
- [x] D1 GovReport lexical/objective family 已完成（12/12 configs、每個 681 frozen dev
      rows；另保留 1 個 L10 interruption failure）。L10 全文候選 macro `0.415585`、
      對 base `+0.045200`；dev point estimate 高於 Lead `0.399232` 與 Random `0.408844`，
      但尚未對強 baseline、做 paired significance 或看 dev-test，不得晉級。完整表見
      `D1_SENSITIVITY_STATUS.md`。
- [x] D1 Multi-News cheap-multiroute family 已完成（12/12 configs、每個 3,935 frozen
      dev rows；另保留 G00/G01 各一個 external interruption failure）。G02 route-top-K 80
      居首，macro `0.324305`；相對純 lexical L00 `+0.013952`，受控 pool mean/max
      `48.03/60`，但仍低於 Lead macro `0.326291`，不得晉級。G06 membership-only 比
      graph base 低 `0.004023`，route-aware salience 不可刪。

## 2026-08-06 selector milestone

- [x] 寫定 `SELECTOR_COMPARISON_PROTOCOL.md`：區分 candidate-matched selector
      swap 與 full-source strong baseline，並預先定義 NSGA-II 去留規則。
- [x] 修正 SentenceTransformer mean-pooling + Normalize 契約；本地 pinned model
      與官方 SentenceTransformer 的 centroid scores 最大差降至約 `6e-8`。
- [x] 實作 deterministic MMR、`sbert_centroid`／`sbert_mmr` baseline、SBERT
      selector matrices、單次 encode reuse、matched-input fingerprints。
- [x] 312 tests passed；真實 canonical Multi-News 3-row smoke 三方法輸入 hashes
      完全一致，3/3 可行。這一輪無 ROUGE，只是 correctness/cost evidence。
- [x] 在看 comparative ROUGE 前凍結 reference-blind 200-row pilot manifest
      （SHA-256 `b0562eb4...c31b2e`）；NSGA-II pilot 固定 64×80、seed 2024。
- [x] 完成 200-row matched-selector pilot：全量 policy preflight、完整 predictions、
      per-example ROUGE、10,000 次 paired bootstrap、Holm correction 與長度統計。
      MMR vs Greedy：R-1 `+0.01488`（95% CI `[+0.00697,+0.02272]`，
      Holm `p=0.0024`）、R-2 `+0.01472`（`[+0.00523,+0.02371]`，
      `p=0.0100`）；R-Lsum `+0.00770` 但校正後不顯著。MMR 平均只多
      `3.08` words、少 `3.05` sentences。NSGA-II seed 2024 對 Greedy 三指標
      全不顯著、總時間 `411.7s vs 90.0s`。暫定 MMR 主線、NSGA-II comparator。
- [x] NSGA-II pilot stability extension：完整報告 seeds
      `[7,42,2024,2025,3407]`。五 seed 平均 R-1/R-2/R-Lsum
      `0.40376/0.10970/0.36820`，全部低於 Greedy；seed 間 mean pairwise
      selection Jaccard `0.639`，僅 `9.5%` rows 五 seed 完全相同。依 gate，
      **selector 架構確定以 MMR 為主，NSGA-II 降為 comparator**；是否在 full
      validation 保留 NSGA diagnostic，由成本與審稿敘事另決定，不再阻塞主線。
- [ ] 舊的「跑 full Multi-News validation」排程已被 2026-08-08 治理規則取代：
      配置搜尋只跑 dev；每個候選配置只能看一次 dev-test。Greedy/MMR/NSGA-II、
      full-source SBERT baselines 與 paired bootstrap 必須分別按此 partition protocol 執行。
- [ ] PacSum 與 GovReport 仍未完成；test split 仍不得執行。

> 這是**唯一的執行清單**。研究標準以 `paper_revision_plan_IEEE_Access.md` 為準；程式稽核與策略評估的結論全部收斂到這裡。
> 每天工作看這份就好，需要理由再回去翻對應的分析文件。
>
> 版本：2026-08-02 ｜ 進度標記：`[ ]` 未開始 `[~]` 進行中 `[x]` 完成 `[!]` 卡住

---

## 怎麼用這份文件

- 階段是**有順序的**，`Gate` 沒過就不要進下一階段
- 每個任務都有 **DoD（完成定義）** —— 沒達到就不算完成
- 標 🔴 的是**擋路項目**，不做完後面全部白做
- 標 ⏱️ 的是估時（單人工作天）

**總估時：8–12 週**。不建議壓縮，半套修改會再被拒一次。

### 與研究主計畫的對齊

本文件的 **Phase 1–6 編號刻意與 `paper_revision_plan_IEEE_Access.md` §15 完全一致**，
方便兩份文件交叉對照。差別只有：

| | `paper_revision_plan_IEEE_Access.md` §15 | 本文件 | 說明 |
|---|---|---|---|
| Phase −1 | §15（已補入） | 決策與凍結 | legacy Multi-News 診斷顯示舊方法未勝 Lead，先做研究路線決策 |
| Phase 0 | 專案治理與可重現性整理 | 專案整理 | ✅ 相同；清理動作仍須另行確認 |
| Phase 1–6 | ✅ 相同 | ✅ 相同 | 內容以研究主計畫為主幹，插入實測衍生的必做項目 |

**分工原則**：研究主計畫負責「為什麼、要達到什麼標準」，本文件負責「今天做什麼、怎麼算完成」。
兩份有衝突時，以研究主計畫的 gate 為準；數字則一律回到 artifact、程式版本、資料 fingerprint 與 evaluator protocol 驗證，不能由本文件自行取得權威地位。

---

## Phase −1：決策與凍結 ⏱️ 1–2 天

> 動任何程式之前先做完這階段。這裡沒想清楚，後面全是白工。

### 必須先接受的三個事實

- [x] 🔴 **接受 F-0 的正確範圍**：legacy Multi-News ExpB 沒有贏過同資料同 evaluator 的 Lead（R-2 −0.0048、R-Lsum −0.0021）；CNN/DM 尚無同 split 同 evaluator 的有效勝負
      → 研究主計畫 §6.1「連 Lead 都贏不過就停止投稿並重新設計」的 No-Go 條件**已經觸發**
- [x] 🔴 **接受 P0-01**：`runs/tuning_experiments/` 全部 11 個 run 都是 test set 調出來的 → **既有結果全部作廢**
- [x] 🔴 **接受病因假說**：系統選句 61.7% 與 Lead 重疊、22.8% 命中 legacy greedy reference
      → 重現腳本已版本化於 `scripts/audit/selection_diagnostics.py`，數字可重跑確認；
        但仍跑在 test-tuned legacy artifact 上、200 篇抽樣、非 official oracle，須在新 validation pipeline 重做

### 決策

- [x] 選定研究路線：**研究主計畫路線 A（方法型）**；是否成功仍由 validation Go/No-Go 決定
      → 核心方法貢獻 = **打破候選池的 lead bias + provenance-aware fusion + budget-aware routing**
- [x] 選定主 benchmark：**GovReport + 原版 Multi-News**；目前 frozen 的 Multi-News clean variant 只作 U+FFFD paired sensitivity，PubMed 是需另作資料／成本 pilot 的替代
      → CNN/DM 是通過 Gate 3 後才考慮的 optional sanity；SciTLDR 不列入 v1 排程
- [x] 寫下 **Go/No-Go 條件**（研究主計畫 §9 與 `ARCHITECTURE.md` freeze gate）；最終 configuration 仍須 validation 後簽字凍結
- [x] 寫下 **canonical method specification**：`ARCHITECTURE.md` 為技術規格來源；目前是 Target Architecture v1，尚未 freeze

### 凍結

- [x] 對已核對的 legacy commit `1b9fe6f` 建立 annotated tag `legacy_ict_express`；不可把目前未提交修正誤標成 legacy
- [x] 在 `runs/README.md` 寫明「以下結果為 test-set 污染，不得用於新論文」

**Gate −1：✅ 已通過。** 路線、主 benchmark、Go/No-Go、method spec 已寫入版本化文件；這不等於架構或超參數已完成 validation freeze。

---

## Phase 0：專案整理 ⏱️ 1 天

> 詳細指令見 `REPO_CLEANUP.md`。這階段純粹是降低後續的認知負擔。

- [x] 100 個 legacy configs、237 個 archived runs、35 個 archived scripts 已位於 archive 路徑並由 `.gitignore` 排除；本機內容保留，未做破壞性刪除
- [ ] 將疑似死碼 `src/pipeline/build_features.py` 移入 legacy archive；先做入口與歷史重現檢查，不直接刪除
- [ ] `frontend/`、`backend/`、`experimental/`、`notebooks/` 移出研究主線（另開 repo 或標明與論文無關）
- [~] runtime／CI requirements 已拆分且重複宣告已清除；正式 Python／套件 lockfile 與乾淨環境重製仍未完成
- [~] `pytest` 已納入依賴、完整 tests 與 GitHub Actions 可跑；仍待把 test tooling 從 runtime requirements 分離成明確 dev lock

**Gate 0**：`ls configs/` 與 `ls runs/` 一眼看得懂哪些是現行的。

---

## Phase 1：正確性重構 ⏱️ 1–2 週 🔴

> 目標：讓每個數字都可被獨立驗證。這階段不追求分數。

### 1a. Patch 與核心 regression 已完成，多資料集／外部協定驗收仍待補

- [~] `src/eval/rouge.py` → 已改 ROUGE-Lsum、同一 reference 由最高 R1 選定、長度 mismatch fail，內部 golden/regression 已通過；published-protocol parity 仍待驗證
- [x] `src/eval/oracle.py` → canonical `documents` 已正確展平，舊 schema 不符會
      fail loud；ROUGE-1／ROUGE-2／ROUGE-Lsum 各自獨立最佳化並保存 selections。
      搜尋每一步與最終輸出都採 source order。名稱固定為 greedy reference，明記
      `exact_upper_bound=false`；舊 `greedy_oracle_*` 名稱僅留相容 wrapper。SciTLDR
      official single-sentence conformance 因 v1 排除該資料集而不重開。
- [x] `src/features/graph.py` → dense input 不被 mutation、dangling mass、zero diagonal、sparse edge bound 均有 regression tests
- [x] `src/models/extractive/encoder_rank.py` → 模型快取、完整輸入 batch encode、revision/truncation artifact 已接線；pinned MiniLM CPU smoke 與 3-row canonical pipeline 通過（GPU 與正式成本屬後續 cost pilot）
- [x] `src/pipeline/optimizer_dispatch.py` → NSGA-II 參數接線與 no-fallback pytest regression 已通過
- [x] `src/pipeline/select_sentences.py`／`evaluate.py` → production prediction 不再攜帶 gold；評估另以 `--gold` 按 ID 嚴格對齊，並禁止 `candidates.recall_target`
- [x] `scripts/audit/` → 稽核診斷腳本已版本化（`lead_vs_system` / `selection_diagnostics` / `dataset_headroom` / `plm_timing`），
      由版本化位置重跑確認 F-0 與 61.7%／22.8% 數字完全一致；用法見 `scripts/audit/README.md`
      **仍是 diagnostic**：跑在 legacy artifact、內部 Lsum 協定、抽樣、未做 paired test

**1a 的共同驗收條件**（全部完成才能把上面的 `[~]` 改成 `[x]`）：

- [x] `pip install pytest` 並讓 `tests/` 能跑（2026-08-05：289 local passed，PR #15 Linux CI 綠燈）
- [~] Phase 1 canonical 主路徑的 patch 已有 golden／regression／10-document snapshot；TF-IDF/TF-ISF similarity parity、published-protocol parity 與尚未實作的多資料集路徑不在現有 289 tests 的完成範圍
- [x] v1 已排除 SciTLDR，因此 official single-sentence oracle conformance 不屬目前 Phase 1；evaluator 維持 fail-closed。若日後重新納入，須重開此 gate

### 1b. 資料層

- [x] 🔴 `preprocess_scitldr.py`：**停止串接 multi-reference**，`references` 存成 list（2026-07-26；含 canonical schema 與 golden test）
- [~] canonical Multi-News 與 GovReport 已改用 deterministic NLTK Punkt 並保存 char-span mapping；CNN-DM 只有 Gate 3 後納入時才須驗證
      （legacy 正則分句曾造成 358/37349 個「句子」超過 80 字，最長 855 字，該 flat artifact 不得進正式實驗）
- [x] Multi-News preprocessor 正確保留 `|||||` 分隔與換行 mapping；U+FFFD 預設 fail closed。正式 `multinews-validation-v1` 政策已在看 validation 分數前凍結：主分析保留 5,621 列且禁止修字，另以固定 72-row manifest 產生 5,549-row clean sensitivity；runner 強制核對 policy、dataset 與 manifests 的 SHA/fingerprint
- [~] 已實作從 pinned 作者資料重建 Multi-News，保存 boundary、source order、raw char span、hash 與 original-to-cleaned mapping；validation 的 5,621-row main 與 5,549-row clean sensitivity 已生成並受 frozen policy 守門，train/test 與各自 manifest 仍待生成。legacy 扁平 `sentences` 不得進正式實驗
- [x] GovReport 官方資料層已驗收：author archive SHA-256、validation ID counts、CC-BY-4.0、nested section／paragraph metadata、GAO Letter 規則與唯一空 reference 排除均版本化；canonical 973 rows、CRS 361／GAO 612、U+FFFD 0，詳見 `docs/research/evidence/a3_govreport_validation_data_audit.json`
- [x] `max_words / max_sentences / max_model_tokens / candidate_budget / compute_budget` 已拆成不同設定與 output artifact；`unit: words` 不再繞過 selector
- [ ] **條件式**：只有 Gate 3 通過且決定保留 CNN/DM optional sanity，才重建其 canonical validation／官方 **test 11,490**；不得使用舊 validation 結果冒充 test
- [~] 資料健檢器已實作：筆數、ID、split、文件／reference／每列句數分布、U+FFFD、debug subset、revision 與 checksum；兩個 primary 的 validation 已生成並保存證據，其他 split 不在 freeze 前讀取範圍
      （validation：5,622 raw → 5,621 canonical，1 列空來源排除；72 列／1,042 個 U+FFFD 依 frozen policy 保留於 main 並排除於 paired clean sensitivity；58 個 singleton clusters；412 列少於 20 句；最大 3,347 句，單句最長 2,638 words）
      （GovReport validation：974 official → 973 canonical；1 列空 reference 排除；每列平均 316.7 句、最大 2,889 句；reference 平均 570.2 words；section path 與 paragraph position 缺失皆 0）

### 1c. 候選生成重構 🔴 這是核心

- [x] 🔴 **lexical、semantic、sparse graph/structure 三路各自在完整輸入上獨立排名**，不可先被共同候選池截斷
- [x] 🔴 **候選池多來源聯集**：lexical／semantic sentence encoder／sparse graph 均先對完整輸入評分；`route_top_k` 保存 proposals、`min_per_route` 優先保留 route-exclusive evidence，再以 RRF 填 total cap；短文件以逐列 effective reservation 誠實降級並保存 requested/effective/shortfall，非法全域設定仍 fail loud
- [x] 🔴 position／document／section strata coverage guard 已可獨立設定；輸出明記 `guard:*` reason，且不把它們宣稱為第四個語意 route
- [x] candidate record 保存 `sentence_id / original_index / document_id / section_id / route raw score / rank / percentile / route agreement / fusion score / inclusion reason / model revision / deterministic cost facts`
- [x] K 在完整 rank 排序後截取；RRF 只能從 proposal union 與 explicit guards 填補，不得從全文引入任何 route top-K 外句子；最後才按原文位置輸出候選

### 1d. 路由與融合層

- [x] lexical TF-ISF v2 改用非負平滑 `log((N+1)/(sf+1))`；ubiquitous term 為 0 而非負證據，revision 明記 `smooth_nonnegative_sublinear_unigram`，v1 僅保留 legacy 重現
- [~] semantic route 已要求明確 sentence-encoder checkpoint/revision、一次載入、batch encode，並記錄 `max_model_tokens` 與截斷率；pinned MiniLM 真實 CPU 與 3-row Multi-News smoke 已通過，尚待正式 cold/warm cost pilot
- [x] graph candidate route 預設為有界 TF-IDF cosine sparse kNN；dense `N×N` 僅能以 `dense_legacy` 明確啟用
- [x] 候選融合採 normalized reciprocal-rank fusion、rank percentile 與 route agreement；MVP selector 實際接收 RRF salience，不再只使用 provenance membership
- [x] candidate route、已啟用 feature 與 similarity implementation 失敗會使 run fail；不再填 0、切換 NumPy 實作或靜默 fallback
- [~] 已建立 `compute_budget.mode: fixed` 與明確 enabled routes；adaptive allocator 尚未實作，若誤設為 adaptive 會 fail loud

### 1e. Objective 與 selector contract

- [~] task-profile factory 已使單句只啟用 salience、強制一個句子並拒絕 subset NSGA-II；document-group coverage 尚未實作，若提前宣告會 fail loud
- [~] canonical multi-sentence 已禁止 raw sum，僅允許 mean／length-normalized salience；shared evaluator 已固定 salience／full-source facility coverage／平均 pairwise redundancy 的方向與 aggregation，並把 `full source × candidates` coverage matrix 與 `candidates × candidates` redundancy matrix 分開；權重與跨文件尺度仍待 validation pilot
- [~] @chi 07-27 | 上一項的禁令只涵蓋「有 `task_profile` 且 `output_mode=multi_sentence`」的路徑（`factory.py:87-92`）；沒有 `task_profile` 的 legacy_unprofiled 路徑（`factory.py:44-57`）預設仍是 `sum`，不受此禁令限制，屬潛在缺陷、尚未確認實際造成污染，詳見 `CODE_AUDIT_IEEE_Access.md` F-14
- [x] `min_words / max_words / max_sentences / non-empty` 已由同一 feasibility contract 判斷；逐列以 exact source capacity 產生 requested/effective minimum 並保存 reason，僅允許 source-intrinsic shortfall 誠實調降。candidate pool 若不能達到 effective minimum、或 optimizer 在可行時失敗，仍 fail loud；不可選超長句不占 route/candidate quota 並留下 artifact
- [~] deterministic greedy、GRASP 與 NSGA-II 在**新 canonical pipeline** 已使用完全相同 candidates、objectives 與 constraints（注入同一 `SelectionObjective`），且保存 final evaluation；獨立 MMR baseline 尚待 Phase 2
- [ ] @chi 07-27 | **legacy config 仍違反此條**：`2_Fusion_NoNsga2.yaml`（`fast_fused`→`greedy_select`）與 `2_Fusion_ExpA/B/C`（`fast_nsga2`→`nsga2_select`）走不同呼叫鏈，差異不只 optimizer，因此不是 matched ablation，證據見 F-15
- [~] NSGA-II artifact 已保存完整可行 Pareto front 與 per-solution objectives；目前 weighted-sum Pareto policy 僅為 provisional，仍須在 validation 凍結 knee/reference-point policy
- [~] 3-row correctness smoke 證實未設下限時 mean-salience 會退化成 1 句（41–88 words）；MVP 因此暫設 requested 200–250 words 作 length-matched validation band。完整 validation 有 72/5,621 列的全文低於 200 words，精確 upper-bound feasibility audit 亦恰為這 72 列（沒有額外 fragmentation case），故逐列 effective minimum 誠實調降；數值與最終 objective 仍未 freeze，且不得依 test 調整
- [x] full-source coverage smoke 的 coverage universe 為 80／227／92 句（非 candidate pool 的 55／60／60）；三筆輸出為 246／240／248 words、全部 feasible，union/guard 越界與 RRF mismatch 皆為 0
- [x] non-negative smoothed TF-ISF v2 重跑同一 smoke，輸出長度與選句數維持 246／240／248 words、5／4／7 句，provenance revision 正確且所有 contract checks 仍通過

### 1f. 測試

- [x] GitHub Actions unit-test CI：push／PR 到 `master` 自動 compile + `python -m pytest -q`；正式 benchmark 不納入輕量 CI
- [x] TF-ISF v1/v2、length、position v1/v2 的手算 golden tests；測試明確記錄 legacy repetition/length/lead bias，不把現況誤認為已驗證的優良公式
- [x] `rougeL` vs `rougeLsum`、pred/ref 對稱分句、corpus guard 與 per-example mean alignment golden tests
- [x] SciTLDR max-ROUGE-1 選定同一 reference 的 aggregation rule 已有 regression；v1 不跑 SciTLDR，故不排程 local `rouge-score` 對官方 `files2rouge` 的數值 conformance。重新納入時才重開
- [x] Graph：diagonal、threshold、dangling node、sparse edge bound
- [x] 候選 top-K rank、union boundary、route reservation（含短文件 shortfall）、RRF selector handoff、route provenance 與 document guard 測試
- [x] canonical schema 與 production prediction 已保存 Multi-News document boundaries／selected sentence provenance；candidate route 與 enabled feature 均 fail loud
- [~] task-profile matrix 已測 single sentence 不建立 redundancy objective且拒絕 subset NSGA-II；multi-document group coverage 尚未完成
- [x] shared objective 手算 golden、min/max/non-empty feasibility、Greedy/GRASP/NSGA-II handoff、NSGA-II 參數傳遞、seed 跨重跑決定性與 **no-fallback** 已測
- [x] 10 篇 toy pipeline snapshot test；保存 route/proposal/reservation/guard/selector/objective/feasibility 決策軌跡，float 使用跨平台 tolerance

**Gate 1**：所有手算測試通過；同 seed 重跑得到相同 indices；故意移除 pymoo 時 run 必須 fail。

---

## Phase 2：Baseline 與 reality check ⏱️ 1 週 🔴

> **這階段的唯一目的：確認新架構真的比 Lead 好。沒過就不要往下走。**

- [x] **PR #10：baseline foundation + Lead 已進 master** —— `src/baselines/contract.py`、
      `lead.py`、`cli.py` 共用 canonical data preflight 與 upper-budget contract，保存
      ordering、requested/effective budget、source capacity、selected words 與 `min_words`
      不適用原因；217-test checkpoint 已涵蓋其 golden／CLI provenance。這只完成程式基礎，
      **不等於下方兩個 primary 的正式 Lead run 已完成**
- [x] **PR #11：Random baseline 已進 master**（235 tests）—— per-row SHA-256 seed 衍生、
      `--seed` 雙向 fail loud、去詞彙化的診斷 fixture、`scripts/audit/random_baseline_min_words.py`
- [x] **PR #14 + #15：TextRank／LexRank 程式與 offline tokenizer hotfix 已進 master** ——
      pinned `sumy==0.12.0`、shared baseline contract、`preserve_line=True` word-only adapter；
      Windows 289 tests 全過且 PR #15 Linux CI 綠燈。最終實作已在 frozen Multi-News
      validation 完成 5,621-row full-split rerun：TextRank `0.413845 / 0.128837 / 0.368487`，
      LexRank `0.430671 / 0.135995 / 0.389532`；完整 hashes 與 integrity 見
      `evidence/f19_centrality_final_pipeline.json`。**這只完成 Multi-News 的兩個 baseline，
      不代表 Gate 2 已完成**
- [x] **Multi-News validation 的 Lead governed artifact 已產出** ——
      `runs_v2/gate2_lead_document_order_validation/`，5,621 篇，`0.433204 / 0.146768 / 0.394039`
- [x] **第一次 validation pilot 已量測（diagnostic）** —— 見 `CODE_AUDIT_IEEE_Access.md` F-18

> 🔴 **2026-08-03 pilot 的結論：沒有配置在三項 ROUGE 全面贏過 Lead。** 後續 final
> `length_normalized` 在 R-1／R-Lsum 略高，但 R-2 低 0.011423，且未做 paired significance；
> 因此仍不可寫成「贏過 Lead」。
> 最佳配置 `greedy + length_normalized` 的 R-1／R-Lsum 領先**完全由多用 10.4 字解釋**
> （長度括弧兩側皆已量測）；R-2 在任何長度下都輸 0.011–0.024。
> `mean` 配置下系統 R-Lsum 甚至**低於 Random baseline**。
> 這不是 Gate 2 的最終裁決（MVP config、無 graph 軌、單 seed、未做 paired bootstrap），
> 但**足以說明「契約做完」離「方法有效」還很遠**。

**接下來的優先順序（依成本效益排序，2026-08-04）：**

- [x] ✅ **F-17 已選 option 1，契約與 full-split regression 完成** ——
      lower-bound document infeasibility 保留 selector 實際嘗試結果，逐列寫入
      `feasible=false`、`infeasible_code`、reason 與 violations；candidate capacity、
      Greedy、GRASP、NSGA-II、空來源／無 eligible sentence 均不得再中止 atomic batch。
      upper-bound、schema、config 與 route failure 仍 fail loud。正式主報 all rows，
      common-feasible intersection 只作 paired sensitivity，不得讓各方法各自刪列比較。
      2026-08-04 實測 governed Multi-News validation 5,621/5,621 rows 成功落地，
      5,620 feasible／1 recorded infeasible；primary all-rows R1/R2/Lsum =
      `0.423018 / 0.129178 / 0.372800`，selection time 2,146.53 秒（本機 CPU）。
- [ ] 🔴 **`objectives.importance_aggregation` 的正式選擇** ——
      pilot 顯示它的影響是 selector 的約 6 倍（+0.0232 vs +0.0039）。
      這個參數必須在 validation 上決定並凍結，不能沿用 MVP 的 `mean`。
- [ ] 🟠 **NSGA-II + `length_normalized`** —— 目前最佳 objective 配最佳 selector，
      **尚未跑過**。約 5.4 小時。
- [ ] 🟠 **開啟 graph 軌的配置** —— §5.4 刪除條件尚未被任何實測觸及。
- [ ] 🟡 **paired bootstrap** —— pilot 的所有差距（含 +0.0039 與 −0.0174）都還沒有顯著性。
      在此之前不得對任何一項宣稱勝負。

### 2.0 執行資料集矩陣 v1（2026-07-30 決定）

| 資料集／分析 | v1 決策 | Phase 2–3 | Phase 4 locked test | 是否阻塞主線 |
|---|---|---|---|---|
| **原版 Multi-News main** | **必跑 Primary B** | 5,621-row frozen validation | configuration freeze 後跑 canonical official test | ✅ 是 |
| **Multi-News clean sensitivity** | **必跑 paired sensitivity** | 同一 validation 排除 frozen U+FFFD 72-row manifest，5,549 rows | 只有在 test policy 於看分數前另行版本化後才跑 paired test | ✅ validation sensitivity 是 |
| **GovReport** | **必跑 Primary A** | canonical validation；不用 train 做 task-specific training | configuration freeze 後跑 official test | ✅ 是 |
| CNN/DailyMail | **延後、可選 sanity** | 不用於 Gate 2／3 調參或核心方法選擇 | 只有兩個 primary 過 Gate 3 且資源允許，才以 frozen method 跑 official test 11,490 | ❌ 否 |
| SciTLDR-AIC | **v1 排除，不排程** | 不跑 | 不跑 | ❌ 否；若重新納入，須先改本表並通過 official files2rouge／single-sentence conformance |
| Multi-News bad-retrieval-removed／Multi-News+ | **目前不跑** | 與現有 U+FFFD clean sensitivity 是不同分析，不得混稱 | 不跑 | ❌ 否 |
| PubMed／Multi-XScience | **reserve，目前不跑** | 不跑 | 不跑 | ❌ 否 |

> 「不用 train」只表示 proposed method 不做 task-specific training；若日後納入需要訓練的比較系統，必須另列 training regime，不能混入 no-task-training 主表。

- [~] **兩個 primary 的 Lead frozen-dev point estimate 已由 A1 同 pipeline 產生**；
      Multi-News `0.326291`、GovReport `0.399232`。仍須收斂進 Gate 2 governed matrix、
      paired significance 與共同 reporting artifact，未達完整 baseline DoD。
- [~] 在兩個 primary 跑 TextRank、LexRank（✅ Multi-News final-implementation full split；
      ⬜ GovReport，待資料政策與成本 preflight 凍結後執行）
- [ ] 在兩個 primary 跑 PacSum
- [ ] 在兩個 primary 跑 Sentence-BERT centroid + MMR
- [~] 兩個 primary 的 Random frozen-dev point estimate 已由 A1 固定 seed 產生：
      Multi-News `0.307098`、GovReport `0.408844`；仍須收斂進 Gate 2 governed matrix、
      paired significance 與共同 reporting artifact。
- [ ] 在兩個 primary 跑 exact extractive oracle（可行時）或明確標示的 greedy reference（不可稱 upper bound）
- [ ] Multi-News main／clean sensitivity 對共同 5,549 rows 報 paired 差異；不得把 clean 分數取代 5,621-row main 結果
- [x] SciTLDR 不屬 v1 Gate 2；不執行、不報新比較表。若日後重新納入，先修改本矩陣，再完成官方 `files2rouge`、單句限制、max-R1-reference 與 oracle R1 ≈ 52.4 conformance

**Gate 2**：兩個 primary benchmark 的 baseline 在各自明確 evaluator 下跑出合理數字。v1 沒有 SciTLDR gate。

---

## Phase 3：方法開發（只用 validation） ⏱️ 1–2 週 🔴

> 🚫 **這階段絕對不准碰 test set。**

### 3a. 先看兩個先行指標（比 ROUGE 更早給訊號）

- [ ] 🔴 **候選池對 validated oracle／greedy reference 的 recall@K** —— legacy 值為 **22.8%**，須以 validated oracle 重做
- [ ] 🔴 **選句位置分布與 Lead 的重疊率** —— legacy 值為 **61.7%**

量測工具已存在，改完架構後直接對新 run 重跑同一支腳本即可比較：

```bash
.venv/Scripts/python.exe -m scripts.audit.selection_diagnostics \
  --data <validation.jsonl> --pred runs_v2/<new_run>/predictions.jsonl --budget 245 --limit 200
```

> ### ⚠️ 中途檢查點（最重要的一個）
> 61.7% 與 22.8% 是 legacy baselines。新 validation pipeline 應降低非必要的 lead overlap、
> 提升 validated oracle-candidate recall；若兩者都沒有改善，視為**強烈 redesign 訊號**
> —— 但這是經驗判斷，不是「ROUGE 必然不改善」的數學定理。

### 3b. 方法設計

- [ ] Provenance-preserving fusion（用 route rank/score，不是只看有沒有進 union）
- [ ] Budget-aware adaptive routing（依文件特性決定要不要啟用 PLM）
- [ ] 重新設計 `position` 特徵（legacy greedy-reference 位置中位數探索值為 **0.46**；須在 validation 重做）
- [ ] Objective 正規化：`imp` 不再使用未正規化總和；平均或 length-normalized aggregation 由 validation pilot 決定
- [ ] 依 `ARCHITECTURE.md` 跑 task-profile/objective 啟用矩陣，不按 dataset 名稱偷換公式
- [ ] 明確的 Pareto 選解規則（knee point / reference point，權重只能在 validation 定）

### 3c. Validation 實驗

- [ ] K、graph threshold τ、fusion weights、population/generation 的 sensitivity
- [~] Selector isolation：Greedy / MMR / NSGA-II 的 frozen 200-row + NSGA 五 seed
      已完成並決定 MMR main；GRASP 若保留只作 optional comparator，不阻塞主線
- [ ] Ablation：No-statistical / No-graph / No-PLM / No-provenance / No-routing
- [ ] Route utility：各路 unique candidate recall、quality delta、latency 與 peak memory；無增量效果的 route 刪除
- [ ] 原版 Multi-News main 與 frozen 5,549-row U+FFFD clean sensitivity 作 paired validation 分析；bad-retrieval-removed／Multi-News+ 是未排程的另一種 retrieval-contamination 研究，不得混稱

**Gate 3** 🔴：
- 在 validation 上，至少一個主 benchmark明顯勝過強 no-task-training baseline；另一個至少 non-inferior 或形成預先定義的 cost Pareto 優勢
- candidate recall 與 lead-overlap 診斷可解釋，且至少一條非 lexical route 有可重現的獨立效益；不要求為了好看而機械式降低 lead overlap
- [x] NSGA-II 未在 matched pilot 提供穩定增益且選句不穩定，已移出標題與主方法；只作 comparator
- data schema、budget semantics、objective matrix、route set 與 output policy 全部通過 `ARCHITECTURE.md` freeze gate
- → 通過才 **freeze config**，解鎖 test

---

## Phase 4：正式 test ⏱️ 約 1 週計算

- [ ] 兩個 frozen primary datasets、全 seeds（≥5，建議 10）、一次性執行；只有在 Phase 2.0 已預先納入的 optional dataset 才能追加
- [ ] Paired bootstrap（≥10,000 resamples）、95% CI、Holm correction
- [ ] Runtime / memory：模型只載入一次，分開報 cold-start 與 warmed inference
- [ ] Quality–latency Pareto 圖（**用完整 pipeline 成本**，不是單一元件）
- [ ] 產生 immutable artifacts（config hash、commit、data fingerprint）

**Gate 4**：Go / No-Go 決策（研究主計畫 §9）。

---

## Phase 5：分析與寫作 ⏱️ 1–2 週

- [ ] Candidate analysis：各 route 的 recall@K、overlap、unique contribution
      → 這組實驗直接回答「三軌到底互不互補」
- [ ] Qualitative error analysis（成功/失敗各 ≥3 例，選例規則預先定義）
- [ ] BERTScore
- [ ] （加分）Human evaluation 50–100 篇 × 3 人
- [ ] 依研究主計畫 §11 的骨架重寫論文
- [ ] Reviewer response matrix：四位審稿人每一條意見逐項對應
- [ ] Conference extension table（ICACT → IEEE Access 新增了什麼）

---

## Phase 6：投稿前稽核 ⏱️ 2–3 天

- [ ] Equation ↔ code ↔ config ↔ result 全鏈可追溯
- [ ] 從乾淨環境一鍵重現
- [ ] IEEE Access 合規：引用 ICACT、similarity < 35%、AI 揭露、ORCID、biography
- [ ] 文法校對
- [ ] 對照研究主計畫 §16 的最終檢查表逐項打勾

---

## 論文主張紅線（寫作時隨時對照）

### 🚫 不可以寫

- meta-heuristics outperform LM-based methods
- significantly outperforms all extractive baselines（**目前實測不成立**）
- converge to the global optimum
- 3×–170× speedup 概括完整 pipeline
- primary innovation（graph 只是 thresholded TextRank）
- 任何**抄來的 baseline 數字**

### ✅ 可以寫

- under a fixed no-task-training protocol
- provides a statistically supported quality–cost trade-off
- graph centrality acts as a structural complementary signal
- 可審計的 candidate provenance（相對一般端到端 baseline 更直接，但不能宣稱 neural 方法做不到）
- 實證比較 deterministic MMR／Greedy 與 stochastic multi-objective search；
  不把未帶來品質增益的 Pareto front 包裝成主貢獻
- 以分句／稀疏圖／routing 避免單次全文 512-token 限制；仍須報 sentence encoder 截斷與實測 scaling

---

## 如果 Gate 3 沒過怎麼辦

**不要硬投。** 兩個選項：

1. **改寫成 empirical study / negative result**
   「在 news 領域，lead bias 使 unsupervised 選句方法難以超越 Lead」
   —— 配合 headroom 分析與 oracle gap，這是有價值的發現，但必須有普遍性結論
2. **換到 lead bias 弱的領域重來**（GovReport / PubMed / 法律 / 醫療長文件）

**最糟的選擇是：只修 evaluator、補幾個 baseline、改寫文字就投。**
那會更清楚地顯示輸給 Lead，而且這次是帶著公開程式碼被抓。

---

## 進度追蹤

| Phase | 狀態 | Gate 通過 | 備註 |
|---|---|---|---|
| −1 決策與凍結 | `[x]` | ✅ | 研究路線、primary benchmarks、Go/No-Go、Target Architecture v1、legacy tag 與 invalid-run 標記均已版本化；最終 configuration freeze 屬 Phase 3 |
| 0 專案整理 | `[~]` | | archive 已隔離、requirements/CI 已整理；死碼、非論文模組與 lockfile 仍待處理 |
| 1 正確性重構 | `[~]` | 核心內部 Gate 1 tests 已滿足 | 357 local tests（2026-08-08）、PR #15 Linux CI、10-document snapshot、shared objectives、document-aware position、exact batched Greedy additions、兩 primary validation policy/preflight、partition enforcement、GovReport data layer、A1/D1 runners 與 greedy-reference correctness 已完成；外部 evaluator parity、正式成本 pilot 與 validation-frozen output policy 仍待補；CNN/DM 是 Gate 3 後 optional |
| 2 Baseline | `[~]` | | Lead、Random、TextRank／LexRank／SBERT centroid／MMR 程式已接線；舊 Multi-News full-validation rerun 只保留為 historical diagnostic。PacSum、partitioned SBERT run、GovReport 方法 runs、完整 paired matrix 與 Gate 2 尚未完成 |
| 3 方法開發 | `[~]` | selector sub-gate ✅ | matched selector pilot 與 NSGA 五 seed stability 已完成；MMR main／Greedy reference／NSGA-II comparator。candidate-router 與 route utility gate 尚未完成 |
| 4 正式 test | `[ ]` | | |
| 5 分析寫作 | `[ ]` | | |
| 6 投稿稽核 | `[ ]` | | |
