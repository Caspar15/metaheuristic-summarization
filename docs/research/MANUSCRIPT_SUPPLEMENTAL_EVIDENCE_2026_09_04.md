# Manuscript supplemental evidence（2026-09-04）

## 狀態與用途

本文件補齊 IEEE Access 主文稽核發現的證據缺口：輸出長度、Multi-News
不可行列敏感度、development 配置比較量、固定規則 provenance 案例、route
reservation 單獨消融、lexical candidate route 單獨消融，以及固定候選池的
zero-lexical-weight 診斷。

這些工作**沒有重跑 official test prediction、沒有依 test 分數選新方法，也沒有更換
frozen final system**。輸出長度與敏感度只讀既有 frozen test artifacts；三項新增診斷
在看見各自新 variant 分數前預先登記，且只跑 frozen development partition。故它們是
投稿說明與事後機制證據，不是新的 confirmatory test。

機器可讀來源：

- `runs_v2/manuscript_supplemental_analysis_v1/analysis.json`
- `runs_v2/manuscript_supplemental_analysis_v1/provenance_case.json`
- `runs_v2/postfreeze_no_reservation_v1/analysis.json`
- `runs_v2/postfreeze_no_lexical_route_v1/analysis.json`
- `runs_v2/postfreeze_zero_lexical_weight_v1/analysis.json`
- `runs_v2/postfreeze_zero_lexical_weight_v1/implementation_snapshot.json`
- `configs/preregistrations/postfreeze_no_reservation_ablation_v1.json`
- `configs/preregistrations/postfreeze_no_lexical_route_ablation_v1.json`
- `configs/preregistrations/postfreeze_zero_lexical_weight_exact_pool_v1.json`

## 1. Route reservation：保存來源差異，但沒有品質增益證據

比較只把 `candidate_budget.min_per_route` 從 20 改成 0，其餘凍結。表中的差異是
Full − No-reservation；100,000 次 paired bootstrap，資料集內 Holm-4。

| Dataset | Full macro | No-reservation macro | Full − variant | 95% CI | Holm-4 p |
|---|---:|---:|---:|---:|---:|
| GovReport dev (681) | 0.457404 | 0.457404 | -0.000001 | [-0.000029,+0.000028] | 1.0000 |
| Multi-News dev (3,935) | 0.330417 | 0.330643 | -0.000226 | [-0.000432,-0.000019] | 0.0995 |

Reservation 仍略提高 route-exclusive candidate 比例：GovReport `0.6973 → 0.7115`，
Multi-News `0.3391 → 0.3434`；但沒有 multiplicity-corrected 品質增益。因此論文只能把
它描述為**候選來源平衡與可稽核紀錄機制**，不能列為 ROUGE 改善來源。

## 2. Lexical candidate route：事後單獨移除反而較好

比較只移除 lexical candidate-generation route，semantic、graph、RRF、reservation、
coverage guard、selector 與長度協定維持不變。Selector 仍使用 TF-IDF 特徵，因此這不是
「移除所有 lexical information」。表中差異是 Full − No-lexical-route；100,000 次
paired bootstrap，資料集內 Holm-4。

| Dataset | Full macro | No-lexical-route macro | Full − variant | 95% CI | Holm-4 p |
|---|---:|---:|---:|---:|---:|
| GovReport dev (681) | 0.457404 | 0.461872 | -0.004468 | [-0.006043,-0.002902] | 0.000080 |
| Multi-News dev (3,935) | 0.330417 | 0.334800 | -0.004383 | [-0.005171,-0.003595] | 0.000080 |

這是負結果：凍結架構中的 lexical candidate route 不是正向品質來源。由於這項分析在
final test 後才提出，不能據此重建另一個 final system 或補跑 test；主文應保留 final
system 的真實三路徑描述，但不得再聲稱「三路各自皆有正向品質貢獻」。可以寫的是：

- prespecified ablation 支持 semantic route、graph route 與 provenance-aware ranking；
- post-freeze analysis 顯示 lexical candidate route 可被簡化，須視為架構限制；
- lexical features／TF-IDF selector 的作用未被此消融否定。

## 2.1 Lexical 傷害來自排名投票還是候選成員？

第三項 post-freeze 診斷逐篇固定完整系統的候選 indices，三條 route 仍全部重算，
但只把 weighted RRF 的 lexical 權重設為 0。因候選 membership 完全相同，Full 與
Zero-weight 的差異只來自 lexical ranking vote；Zero-weight 與既有 No-lexical-route
的差異，則反映保留 lexical 所提出／保留的候選成員是否有額外影響。這項診斷在新
zero-weight 分數產生前登記，兩個資料集合計 16 endpoints，以全域 Holm-16 校正。

| Dataset | Comparison (left $-$ right) | Endpoint | Difference | 95% CI | Holm-16 p |
|---|---|---|---:|---:|---:|
| GovReport | Full $-$ exact-pool $w_{lex}=0$ | R-1 | -0.002594 | [-0.004074,-0.001115] | 0.002880 |
| GovReport | Full $-$ exact-pool $w_{lex}=0$ | R-2 | -0.006110 | [-0.008000,-0.004207] | 0.000320 |
| GovReport | Full $-$ exact-pool $w_{lex}=0$ | R-Lsum | -0.004681 | [-0.006254,-0.003114] | 0.000320 |
| GovReport | Full $-$ exact-pool $w_{lex}=0$ | Mean | -0.004462 | [-0.006049,-0.002903] | 0.000320 |
| GovReport | exact-pool $w_{lex}=0$ $-$ no lexical route | R-1 | +0.000027 | [-0.000087,+0.000150] | 1.000000 |
| GovReport | exact-pool $w_{lex}=0$ $-$ no lexical route | R-2 | -0.000027 | [-0.000110,+0.000050] | 1.000000 |
| GovReport | exact-pool $w_{lex}=0$ $-$ no lexical route | R-Lsum | -0.000019 | [-0.000141,+0.000109] | 1.000000 |
| GovReport | exact-pool $w_{lex}=0$ $-$ no lexical route | Mean | -0.000006 | [-0.000103,+0.000090] | 1.000000 |
| Multi-News | Full $-$ exact-pool $w_{lex}=0$ | R-1 | -0.001498 | [-0.002166,-0.000826] | 0.000320 |
| Multi-News | Full $-$ exact-pool $w_{lex}=0$ | R-2 | -0.001134 | [-0.001962,-0.000309] | 0.037700 |
| Multi-News | Full $-$ exact-pool $w_{lex}=0$ | R-Lsum | -0.002456 | [-0.003138,-0.001779] | 0.000320 |
| Multi-News | Full $-$ exact-pool $w_{lex}=0$ | Mean | -0.001696 | [-0.002369,-0.001025] | 0.000320 |
| Multi-News | exact-pool $w_{lex}=0$ $-$ no lexical route | R-1 | -0.003099 | [-0.003649,-0.002544] | 0.000320 |
| Multi-News | exact-pool $w_{lex}=0$ $-$ no lexical route | R-2 | -0.002137 | [-0.002715,-0.001563] | 0.000320 |
| Multi-News | exact-pool $w_{lex}=0$ $-$ no lexical route | R-Lsum | -0.002825 | [-0.003375,-0.002277] | 0.000320 |
| Multi-News | exact-pool $w_{lex}=0$ $-$ no lexical route | Mean | -0.002687 | [-0.003211,-0.002164] | 0.000320 |

Macro means 為 GovReport `Full 0.457404 / zero-weight 0.461865 / no-lexical
0.461872`，Multi-News `0.330417 / 0.332113 / 0.334801`。因此 GovReport 的 lexical
傷害幾乎完全來自排名投票，保留 lexical 提名的候選 membership 沒有可檢出的額外
影響；Multi-News 則同時存在排名投票與候選 membership 兩部分。這是對 frozen system
的事後機制拆解，不可用來把 no-lexical 變體升格為新的 final method，也沒有跑任何
official test prediction。

正式 pipeline 仍要求啟用 route 的 production 權重大於 0；只有傳入 exact frozen pool
的 audit-only API 允許某一 route 權重為 0。前兩次 GovReport 啟動在任何分數產生前
分別因 Windows worker pipe 權限與正權重 guard 中止，failed evidence 均保留；修正後
加入 regression test，確認 production path 仍拒絕零權重。

## 3. 所有系統的輸出長度

平均值為 words／sentences，括號為 population standard deviation。Random 合併 10 個
固定 seeds 的 document-seed observations；其餘均為每篇文件。

| Dataset | System | Words mean±SD | Sentences mean±SD | Infeasible |
|---|---|---:|---:|---:|
| GovReport | PAMR-ES | 647.08±8.61 | 16.73±3.66 | 0 |
| GovReport | Lead | 633.77±16.61 | 25.82±4.45 | 0 |
| GovReport | Random (10 seeds) | 649.30±8.21 | 28.37±5.09 | 0 |
| GovReport | TextRank | 649.43±8.18 | 16.04±3.42 | 0 |
| GovReport | LexRank | 649.36±8.19 | 21.80±4.40 | 0 |
| GovReport | PacSum-TFIDF | 649.24±8.26 | 20.93±4.90 | 0 |
| GovReport | PacSum-SBERT | 649.35±8.20 | 25.70±10.92 | 0 |
| GovReport | SBERT centroid | 649.44±8.18 | 21.79±6.04 | 0 |
| GovReport | SBERT+MMR | 649.43±8.17 | 21.62±5.41 | 0 |
| Multi-News | PAMR-ES | 244.10±15.35 | 15.05±4.21 | 12 |
| Multi-News | Lead | 233.61±19.34 | 10.70±3.24 | 0 |
| Multi-News | Random (10 seeds) | 246.75±14.89 | 13.14±3.23 | 0 |
| Multi-News | TextRank | 247.25±14.74 | 8.02±2.60 | 0 |
| Multi-News | LexRank | 246.93±14.97 | 10.97±3.01 | 0 |
| Multi-News | PacSum-TFIDF | 245.92±15.12 | 12.04±3.94 | 0 |
| Multi-News | PacSum-SBERT | 246.90±14.93 | 11.36±3.15 | 0 |
| Multi-News | SBERT centroid | 247.16±14.92 | 9.94±3.89 | 0 |
| Multi-News | SBERT+MMR | 247.16±14.92 | 10.11±3.05 | 7 |

GovReport 的 proposed 並沒有比所有強 baseline 填得更滿；Multi-News 也比多數 ranking
baseline 略短。因此 R-1／R-L 差異不能只用「PAMR-ES 更接近上限」解釋。

## 4. Multi-News 12 筆不可行列敏感度

這 12 筆不是來源全文不足：每筆 selection-eligible candidate capacity 均足以超過
200 words。原因是抽取系統不能拆句，Greedy／MMR 按排名逐句加入且不回溯；在
250-word 上限下，某些句子組合停在 200 words 以下。SBERT+MMR 有 7 筆相同類型的
shortfall（6 筆與 PAMR-ES 重疊），而 Lead／PacSum 的排名恰好找到可行組合。這是
selector packing limitation，不應誤寫為來源長度不足。所有輸出均保留實際內容並計分。

主分析仍保留全部 5,621 列。補充分析把 PAMR-ES 不可行的同一組 12 IDs 同時從
PAMR-ES 與 PacSum-TFIDF 移除，只重新彙整既有逐篇分數，不重做摘要。5,609 列結果：

| Endpoint | Proposed − PacSum-TFIDF | 95% CI | Adjusted p |
|---|---:|---:|---:|
| ROUGE-1 | +0.002319 | [+0.000713,+0.003924] | Holm-3 0.004980 |
| ROUGE-2 | -0.006815 | [-0.008725,-0.004921] | Holm-3 0.000060 |
| ROUGE-L | +0.004593 | [+0.002999,+0.006194] | Holm-3 0.000060 |
| Mean ROUGE | +0.000032 | [-0.001604,+0.001666] | 0.970830 |

結論不變：Mean 統計同級，ROUGE-1／ROUGE-L 較高，ROUGE-2 較低。這項 sensitivity
不取代 all-row primary result。

## 5. Development 配置比較量

這是 configuration selection，不是模型 fine-tuning。PAMR-ES 在 GovReport／Multi-News
development program 分別有 69／68 個 unique config hashes，且每個資料集只有 4 次
held-out dev-test score observations。可調 baseline 的候選數為 PacSum-TFIDF 21、
PacSum-SBERT 21、SBERT+MMR 5；Lead、TextRank、LexRank 與 SBERT centroid 各 1，Random
是 10 個事先固定 seeds。這揭露搜尋彈性不完全對稱；公開 search log 能排除 test tuning，
但不能宣稱所有方法得到完全相同的 configuration budget。

## 6. 固定規則 provenance 案例

案例是 GovReport frozen development prediction order 的第一列
`validation_crs_R44729`，不看 ROUGE、不挑人工認為好看的例子。摘要 647 words、13 句。
機器可讀檔列出每句的 route rank、route agreement、fusion rank、retention reason 與
final selection。前五個 reservation-only 未選候選亦完整保留，顯示「因哪條路徑進池」
與「最後被 selector 選中」是兩個不同事件。

此案例只證明 decision provenance 可追蹤，不證明 factuality、因果可解釋性或摘要品質。

## 7. NSGA-II selector 負結果（由主文移至補充證據）

PAMR-ES 的 final system 只使用 GovReport 的 MMR 與 Multi-News 的 Greedy；NSGA-II
不是最終架構元件，也不是本文方法貢獻。為避免主文產生與 provenance-aware fusion
無關的 metaheuristic 支線，主文的 selector table 與成本表只保留 Greedy／MMR 及實際
部署方法；既有 NSGA-II 結果不刪除，完整保留如下。

| Dataset | Greedy | MMR | NSGA-II |
|---|---:|---:|---:|
| GovReport development | 0.417862 | **0.446154** | 0.426844 |
| Multi-News development | **0.328077** | 0.321744 | 0.322615 |

上述為相同候選池、TF-IDF similarity 與長度限制下的 internal Mean ROUGE。成本量測同樣
使用固定 30-document CPU sample；數字是 30 篇總時間，而不是單篇時間：

| Dataset | Selector variant | Cold total (s) | Warm total (s) | Warm per document (s) | Cold/Warm RSS (MiB) |
|---|---|---:|---:|---:|---:|
| GovReport | PAMR-ES with MMR | 176.03 | 10.52 | 0.351 | 774.1 / 422.9 |
| GovReport | PAMR-ES with NSGA-II | 222.61 | 56.74 | 1.891 | 777.8 / 421.9 |
| Multi-News | PAMR-ES with Greedy | 47.85 | 9.23 | 0.308 | 594.6 / 398.9 |
| Multi-News | PAMR-ES with NSGA-II | 91.69 | 50.07 | 1.669 | 599.6 / 400.7 |

因此 NSGA-II 沒有在任一資料集成為品質最佳 selector，且 warm 執行時間明顯增加。這是
透明保留的負結果，不得被改寫為 final method 的一部分，也不因移出主文而從 repository
刪除。

## 8. 主文使用規則

- test 主表同時報 ROUGE-1、ROUGE-2、ROUGE-L 與預先定義的 Mean ROUGE。
- Mean ROUGE 保留為 frozen primary summary endpoint；不能在看到 test 後改成新的 endpoint。
- 三個分項必須並列，且以 Holm-3 報 paired inference，避免 Mean 掩蓋 Multi-News 的 R-2 負差。
- Post-freeze variants 只可標成 supplemental／mechanism checks，不可和 prespecified E3
  混稱為同一 confirmatory family。
- 不宣稱 route reservation 提高品質；不宣稱三條 candidate routes 各自皆有正貢獻。
- Zero-weight exact-pool 診斷只可解釋 lexical ranking vote 與 candidate membership，
  不可據此重選 final method 或補跑 test。
- 不宣稱「只調 selector」或「完全沒有調整」。正確說法是神經模型沒有 fine-tuning，
  route weights 與 selector 均以 development partition 選定並在 final test 前凍結。
- 主文不需要出現 NSGA-II；其 selector 與成本負結果由本文件及原始 evidence 保存。
