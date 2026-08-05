# 計算環境與離線資源清單

> 對應 `paper_revision_plan_IEEE_Access.md` §14 的 reproducibility artifact。
> 本文件只記錄已驗證的環境契約；硬體時間不可跨機器直接比較。

## TextRank／LexRank 的 tokenizer 契約

canonical dataset 已經凍結句界。`src/baselines/centrality.py` 直接以每個
canonical sentence 建立 `sumy.models.dom.Sentence`，禁止再經過 sumy 的
paragraph sentence splitter。

centrality scorer 真正需要的是 `Sentence.words`。目前的
`_WordOnlySumyTokenizer`：

- 繼承 pinned `sumy==0.12.0` 的 `Tokenizer.to_words`；
- 使用 sumy 自己的 English `DefaultWordTokenizer`；
- 不載入、也不呼叫 NLTK Punkt sentence model；
- `to_sentences()` 一律 fail loud，避免未來 refactor 靜默改變 frozen 句界。

因此 CI 與離線叢集不需要下載或 vendoring `punkt`／`punkt_tab`。這也避免把
授權未釐清的 NLTK data package 放進 repository。word-only adapter 與正常
sumy English `to_words()` 的 parity 由 `tests/test_baselines_centrality.py`
固定。另以全部 5,621 篇、456,942 個 canonical sentences 掃描，結果為
**0 mismatch**；資料 SHA、程式 SHA 與 token-stream SHA 見
`docs/research/evidence/f19_word_tokenizer_parity.json`。

`nltk==3.10.0` 仍精確 pin，原因是 reproducibility 與 sumy runtime code
dependency，不代表 repo 會散布任何 NLTK data package。若未來升版：

1. 跑完整 unit tests；
2. 在 frozen canonical inputs 上比較共用 sentence splitter 的輸出；
3. 比較 centrality word-token parity 與 baseline selected indices；
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

PR #14 原 GitHub Linux run 的實際摘要是 `285 passed, 4 skipped`，不是
`289 passed`。四個 skips 來自 `pytest.importorskip("pymoo")`；因此 CI 尚未
覆蓋 NSGA-II。這是既存 coverage gap，與 centrality correctness 分開處理，
但在下一次正式 NSGA-II run 前應將 `pymoo` 納入 CI dependency。

移除 vendored Punkt、加入 word-only parity 與 scorer／feasibility 正交回歸後，
Windows 隔離環境的完整結果亦為 `285 passed, 4 skipped`。

跨平台驗收至少包含 Linux CI 與 Windows 本機 suite。路徑測試必須以
`pathlib.Path`／normalized path component 比較，不能硬編碼 `/`。

## 版本

| Dependency | Pin | 用途 |
|---|---|---|
| `sumy` | `==0.12.0` | TextRank／LexRank implementation；Apache-2.0 |
| `nltk` | `==3.10.0` | sumy runtime 與 project tokenizer code；不含 data package |
