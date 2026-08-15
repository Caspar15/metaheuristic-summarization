# 計算環境與離線資源清單

> 對應 `paper_revision_plan_IEEE_Access.md` §14 的 reproducibility artifact。
> 本文件只記錄已驗證的環境契約；硬體時間不可跨機器直接比較。

> **2026-08-15 狀態**：預註冊 E2 已在 frozen GovReport dev 的 reference-blind
> 30-document sample 完成。九系統各有 cold/warm 三次 measured fresh subprocess、peak
> process-tree RSS 與 scaling；54/54 selected-index identities 一致，且沒有重疊 timed jobs。
> CPU 主表與 cache lifecycle 見 `E2_COST_SCALING_STATUS.md`；GPU 不取代主報告。

## TextRank／LexRank 的 tokenizer 契約

canonical dataset 已經凍結句界。`src/baselines/centrality.py` 直接以每個
canonical sentence 建立 `sumy.models.dom.Sentence`，禁止再經過 sumy 的
paragraph sentence splitter。

centrality scorer 真正需要的是 `Sentence.words`。目前的
`_WordOnlySumyTokenizer`：

- 保留 pinned `sumy==0.12.0` 的 `_is_word` 過濾契約；
- 使用 pinned `nltk==3.10.0` 的 `word_tokenize(..., preserve_line=True)`；
- 不載入、也不呼叫 NLTK Punkt sentence model；
- `to_sentences()` 一律 fail loud，避免未來 refactor 靜默改變 frozen 句界。

因此 CI 與離線叢集不需要下載或 vendoring `punkt`／`punkt_tab`。這也避免把
授權未釐清的 NLTK data package 放進 repository。測試會把任何 sentence
tokenizer 呼叫改成例外，確保 word scoring 只消費既有 canonical sentence。

PR #14 原本宣稱此 adapter 與正常 sumy English `to_words()` 全量 0 mismatch，
但乾淨 Linux CI 證明該 adapter 仍會透過 `nltk.word_tokenize` 的預設參數載入
`punkt_tab`。修成 `preserve_line=True` 後，5,621 篇、456,942 個 canonical
sentences 中有 **1,550 句（0.339%）** 與舊 sumy/Punkt token stream 不同，主要
來自句內縮寫。這是移除二次分句後的預期 scoring 語義變更，不得再稱 token
parity；舊 TextRank／LexRank ROUGE 維持 historical。PR #15 合併後已用最終實作完成
Multi-News validation full-split rerun；新結果與 artifact hashes 見
`docs/research/evidence/f19_centrality_final_pipeline.json`，完整 token hash 與首個差異見
`docs/research/evidence/f19_word_tokenizer_parity.json`。

## PR #15 後的 full-split timing 與結果

兩個 baseline 都在同一台 Windows 11 主機、Python 3.12.7、`sumy==0.12.0`、
`nltk==3.10.0` 上**分開、單程序**執行。選句計時停止後才跑 evaluator，因此不受
ROUGE 工作競爭 CPU 影響。這些 wall-clock 只能描述本機，不能作跨硬體速度宣稱。

| baseline | rows | selection | evaluation | R-1 | R-2 | R-Lsum |
|---|---:|---:|---:|---:|---:|---:|
| TextRank | 5,621 | 523.539 s | 122.895 s | 0.413845 | 0.128837 | 0.368487 |
| LexRank | 5,621 | 766.732 s | 124.673 s | 0.430671 | 0.135995 | 0.389532 |

兩份 predictions 都有 5,621 個唯一 ID、5,621 feasible rows、0 個 selected-index 或
summary reconstruction mismatch。LexRank 仍正確標記 `validation_1082` 與
`validation_2303` 兩個 scorer-degenerate rows；TextRank 為 0。相較 pre-hotfix
historical metrics，TextRank 最大絕對差為 0.000010，LexRank 為 0.000095，研究排序
沒有翻轉。bulk predictions 仍在本機 `runs_v2/`，未進 Git；投稿／外部 artifact review
前必須上傳 immutable artifact store 並依 evidence manifest 的 SHA-256 驗證。

`nltk==3.10.0` 仍精確 pin，原因是 reproducibility 與 sumy runtime code
dependency，不代表 repo 會散布任何 NLTK data package。若未來升版：

1. 跑完整 unit tests；
2. 在 frozen canonical inputs 上比較共用 sentence splitter 的輸出；
3. 比較 centrality word-token delta 與 baseline selected indices；
4. 重新記錄 dependency versions。

本次由 3.9.1 升至 3.10.0 已對全部 frozen source sentences 與 references
做跨版本掃描：462,563 段輸入在兩版皆輸出 513,195 句，stream SHA-256
完全相同。證據見
`docs/research/evidence/nltk_391_310_sentence_split_parity.json`；此結論只涵蓋
本專案的 code-only `PunktSentenceTokenizer`，不外推到下載式 Punkt models。

## LexRank 相似度矩陣的已知退化

sumy LexRank 使用：

```text
idf(term) = log(N / (1 + n_j))
```

對兩句且零共享詞彙的文件，每個詞皆有 `n_j=1`，所以所有 IDF 都是
`log(2/2)=0`。similarity matrix 因而全零，power iteration 正規化零向量時
產生 NaN。

Multi-News validation 全量掃描的診斷結果：

- 2 / 5,621 documents：`validation_1082`、`validation_2303`；
- 兩篇分別只有 59、77 words，所有 eligible sentences 都能放入 250-word
  budget；
- ranking 在這兩篇不影響輸出，因此保留全部 eligible sentences，並記錄
  `scorer_degenerate=true` 與原因；
- 這是 scorer 狀態，不是 length infeasibility，不可濫用 `infeasible_code`。

若 scorer 退化且只有部分 eligible sentences 能放入 budget，哪一句該被刪除
取決於不存在的 ranking signal；目前必須 fail loud，不得任意 fallback。

scoring degeneracy 與 selection feasibility 是正交欄位。即使所有 source
sentences 都因超過 active budget 而被排除，artifact 仍須同時記錄：

```text
feasible=false
infeasible_code=source_no_eligible_sentence
scorer_degenerate=true
```

## O(n²) 成本模型

sumy TextRank／LexRank 建立 dense sentence-similarity matrix，成本由文件句數
尾端分布主導，不可只看平均值。

Multi-News 本機 CPU diagnostic：

| calibration | TextRank wall time | 約略 c（seconds / sentence²） |
|---|---:|---:|
| `validation_2284`, n=3,347 | 32.21 s | 2.88e-6 |
| 30-document isolated batch | 2.21 s | 4.48e-6 |

LexRank 在同一批小型校準約為 TextRank 的 1.46 倍。這些數字只供 GovReport
preflight 排程，不能跨硬體當正式效率比較。下載並凍結 GovReport 後，正式
run 前必須先記錄 sentence-count 的 p50／p95／p99／max，並對尾端文件做
isolated timing。

## Timing 污染規則

TextRank 與第一次 LexRank 同時執行時，TextRank 的
`time_select_seconds.txt` 與 tqdm elapsed 相差超過 5 倍。該 measurement 已
標為 unusable。任何 timing evidence 必須：

- 一次只執行一個量測 process；
- 記錄 CPU／GPU、Python、NumPy／BLAS 與 dependency versions；
- 同時保存工具內 timing 與外部 wall-clock；
- 不得因數值忠實抄自 artifact 就假設 artifact 本身未受 contention 污染。

## CI coverage

PR #14 合併 head 的 GitHub Linux run 實際為 `15 failed, 270 passed, 4 skipped`；
15 個 failure 都源自乾淨 runner 找不到 `punkt_tab`，不能視為綠燈。PR #15 hotfix
改成 `preserve_line=True` 後，Windows 本機完整結果為 `289 passed`，GitHub Linux CI
亦已綠燈。其後 GovReport-centered 分支曾有本機完整結果 **459 passed**，PR #17
clean-clone Linux CI 為 **454 passed／5 skipped**。skips 仍須依 CI artifact 判讀；
E2／E3 已在實際實驗環境完成並各自保存 dependency evidence；未來 final test 前仍須
重新 pin exact execution environment，不能把 unit-test CI 當成正式實驗環境驗收。

跨平台驗收至少包含 Linux CI 與 Windows 本機 suite。路徑測試必須以
`pathlib.Path`／normalized path component 比較，不能硬編碼 `/`。

## 版本

| Dependency | Pin | 用途 |
|---|---|---|
| `sumy` | `==0.12.0` | TextRank／LexRank implementation；Apache-2.0 |
| `nltk` | `==3.10.0` | sumy runtime 與 project tokenizer code；不含 data package |
