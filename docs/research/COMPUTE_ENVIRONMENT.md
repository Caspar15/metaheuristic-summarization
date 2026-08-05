# 計算環境與離線資源清單

> 對應 `paper_revision_plan_IEEE_Access.md` §14「Reproducibility artifact」的
> hardware/software manifest 項目。這裡只記錄「跑得動」需要什麼，不記錄
> 研究結論——結論仍在 `CODE_AUDIT_IEEE_Access.md`。

## NLTK 資料快取（sumy baseline 專屬，見下方 §「範圍」）

### 需要什麼

`tokenizers/punkt_tab/english/` 目錄，**4 個檔案，244 KB**：

```
tokenizers/punkt_tab/english/sent_starters.txt
tokenizers/punkt_tab/english/abbrev_types.txt
tokenizers/punkt_tab/english/ortho_context.tab
tokenizers/punkt_tab/english/collocations.tab
```

實測（2026-08-05）：在 socket 層級完全封網、`nltk.data.path` 只指向這個
244 KB 目錄的條件下，`sumy.nlp.tokenizers.Tokenizer("english")` 建構成功，
`TextRankSummarizer.rate_sentences` 與 `LexRankSummarizer.__call__` 都正常
運作。移除 README 與其他 19 種語言後結果不變。

**已 vendored 進 repo**：`vendor/nltk_punkt_tab/tokenizers/punkt_tab/english/`
（見該目錄的 `README.md`）。CI 上第一次不帶這份資料跑時，GitHub Actions
的乾淨 runner 直接重現了國網會遇到的情況——沒有本機快取、沒有外網——
並印出下面這個真實 traceback，把「需要什麼」從實驗變成了程式碼本身可以
指認的證據：

```
sumy/nlp/tokenizers.py:203   nltk.data.load("tokenizers/punkt/english.pickle")
nltk/data.py:1126            switch_punkt(fil)
nltk/tokenize/punkt.py:1769  find("tokenizers/punkt_tab/english/")
```

這個 traceback 同時證實了三件事：(1) sumy 呼叫的是舊 `.pickle` 路徑，
`switch_punkt` 在 `nltk.data.load` 內部攔截並改向 `punkt_tab`——`.pickle`
從未被讀到，佐證上面 244 KB 的結論不是巧合；(2) `find()` 要找的是**目錄**
（資源名結尾有斜線），路徑層級必須完全對上 `tokenizers/punkt_tab/english/`，
少一層多一層都會失敗；(3) 這個 nltk 版本的 `find()` 對路徑做過安全檢查
（拒絕含 `..` 或非絕對路徑的項目），所以加進 `nltk.data.path` 的必須是
`Path(...).resolve()` 過的絕對路徑（見 `src/baselines/centrality.py` 的
註冊程式碼）。

### 不需要什麼

**古典 `punkt`（pickle 格式，49 MB）不需要，即使 sumy 的原始碼字面上在讀
`tokenizers/punkt/english.pickle`。** 這是因為本專案釘死的 nltk 3.10.0
在 `nltk/data.py` 有一個 `switch_punkt` shim：任何對舊路徑
（`tokenizers/punkt/<lang>.pickle`）的載入呼叫都會被攔截，改為要求
`tokenizers/punkt_tab/<lang>/`（新格式，NLTK 3.9 起導入，動機是安全性——
pickle 反序列化任意資料本身是供應鏈風險，`punkt_tab` 改用純文字/表格檔）。

實測驗證，不是只看 shim 程式碼推論：先只放 49 MB 的 `punkt` pickle、不放
`punkt_tab`，在封網條件下建構 `Tokenizer("english")` 立即 `LookupError`
要求 `punkt_tab`；换成只放 244 KB 的 `punkt_tab/english/`、完全不放
`punkt`，同一組操作（建構 Tokenizer、跑 TextRank rate_sentences、跑
LexRank `__call__`）全部成功。**古典 `punkt` pickle 從未被讀取。**

### 為什麼是 sumy 專屬，不是叢集前置條件（範圍判定，2026-08-05 實測，含 vendored 資料上線後的回歸驗證）

本專案自己的分句器 `src/data/sentence_split.py`（Multi-News canonical
前處理與 ROUGE-Lsum 評測共用，見該檔 docstring 與 F-12）**不需要任何
NLTK 資料檔**：它直接用 `PunktParameters()` + 手動維護的縮寫清單建構一個
**未訓練**的 `PunktSentenceTokenizer`，從不呼叫 `nltk.data.load`。這正是
該檔 docstring 說明的設計動機——避免依賴一個本 repo 未 vendor 的下載資源。

實測兩次，條件逐次收緊，同一天內完成：第一次 `nltk.data.path` 指向一個
不存在的空目錄；第二次（vendored 資料上線後的回歸驗證）改成
`nltk.data.path` **只含** `vendor/nltk_punkt_tab/`（也就是 sumy baseline
實際會用的那份資料本身），兩次都在 socket 層級完全封網。兩次
`build_sentence_tokenizer()`/`split_sentences()` 都正常運作（含縮寫測試句
`"Mr. Smith met U.S. officials..."` 正確切句），結果與是否存在 vendored
資料無關——不是「沒有資料所以繞過了檢查」，而是這條路徑本來就不查
`nltk.data`。

**結論：這份 244 KB 快取只給 sumy-based baseline（TextRank/LexRank）用。**
Multi-News canonical 前處理、GovReport 前處理（若沿用同一個共用
tokenizer，見 CLAUDE.md 第 3 節「不要再寫任何新的分句 regex」）、
ROUGE-Lsum 評測，都不受影響、不需要這份快取。國網計算節點若完全不裝
sumy baseline，就完全不需要這一節的任何東西。

## nltk 版本 pin（與上面的快取內容耦合）

`requirements.txt` 曾經是 `nltk>=3.8.1`——這是下限，不是 pin，而快取內容
隨版本改變：3.8.x 走舊 `punkt` pickle 路徑（不受 `switch_punkt` shim
影響，因為那是 3.9 才加入的），3.9+ 一律要求 `punkt_tab`。國網環境若用
下限解析出 3.8.x，帶上去的 244 KB `punkt_tab` 快取會不被讀取（3.8.x 甚至
不知道 `punkt_tab` 這個路徑），變成一份沒用的快取加上一個看似無關的
`LookupError`。

因此 `requirements.txt` 改為精確 pin `nltk==3.10.0`（本文件所有實測都在
這個版本上做的）。**這個 pin 不只是一般意義的可重現性，而是「離線快取
是否正確」的前提**——版本一變，這整節要不要用、要放哪個目錄都要重新
實測，不能假設沿用。

## 這份資料已 vendored 進 repo，不需要每個環境各自下載

**位置：`vendor/nltk_punkt_tab/tokenizers/punkt_tab/english/`**（詳見該目錄
`README.md`）。CI、本機、國網計算節點三者共用同一份檔案，不需要在 CI
另外加下載步驟——加下載步驟會讓 CI 依賴 NLTK 的伺服器（跟這份資料存在
的理由自相矛盾），對完全無外網的國網節點也毫無幫助。

**註冊方式：`src/baselines/centrality.py` 在 import 時直接把這個目錄的
絕對路徑插進 `nltk.data.path` 最前面**，不是環境變數。理由：環境變數
（`NLTK_DATA`）必須在 CI、本機、國網三個環境**各自設定一次**，而且會被
忘記；import-time 註冊則是任何 import 這個模組的呼叫者（pytest、
`python -m src.baselines.cli`、未來的稽核腳本）都自動拿到同一份資料，
不需要額外的環境設定步驟。用 `Path(__file__).resolve()` 而非相對路徑，
因為這個 nltk 版本的 `find()` 對路徑做過安全檢查（見上方 CI traceback
段落），拒絕含 `..` 或非絕對路徑的項目。插在最前面（`insert(0, ...)`
而非 append）是為了讓這份 vendored 資料永遠優先於機器上其他可能存在的
`punkt_tab`，三個環境的行為才會一致，不受各機器既有快取的搜尋順序影響。

驗證（不是只看測試通過）：

```python
import nltk
import src.baselines.centrality  # 觸發 import-time 路徑註冊
nltk.data.find("tokenizers/punkt_tab/english/")
# 必須成功，且回傳路徑在 vendor/nltk_punkt_tab/ 底下
```

若日後升級 `nltk` 版本，用 `vendor/nltk_punkt_tab/README.md` 裡的指令
重新產生這份資料，並重跑本文件與下方「乾淨環境驗證」——不能假設沿用。

## 乾淨環境驗證（2026-08-05，這才算修好，本機通過不算）

本機有 `~/nltk_data`，所以「本機測試通過」不能證明離線環境真的沒問題。
實測條件：`nltk.data.path` 清空後只加回 `vendor/nltk_punkt_tab/`
（不含任何其他路徑，包括預設的 home directory）、`socket.socket.connect`
在程式層級直接 monkeypatch 成擲錯（比只設代理更嚴格，連 DNS 前的連線
嘗試都擋下）、`HTTP_PROXY`/`HTTPS_PROXY` 指向不可路由位址。在這個條件下
跑完整的 287 個測試（不是 smoke test），結果：**287 passed**。

```python
import socket
def _blocked_connect(self, *a, **kw):
    raise OSError("network access blocked for clean-environment simulation")
socket.socket.connect = _blocked_connect

import nltk
nltk.data.path = []
import src.baselines.centrality  # 這一行本身必須把 nltk.data.path 填成
                                  # 只有 vendor/nltk_punkt_tab/ 這一項
assert nltk.data.path == [".../vendor/nltk_punkt_tab"]
nltk.data.find("tokenizers/punkt_tab/english/")  # 必須成功

import pytest
pytest.main(["-q", "tests/"])
```

## LexRank 相似度矩陣的已知退化（2026-08-05 實測，Multi-News validation）

`src/baselines/centrality.py` 對 LexRank／TextRank 分數做 `np.isfinite`
檢查，非有限值時 fail loud（不讓 `NaN` 靜默流進 `select_by_score` 的排序）。
根因**不是**「整句都是停用詞」（最初的錯誤猜測，已在程式碼與測試裡改正）：

真正機制是 sumy `LexRankSummarizer._compute_idf` 的
`idf(term) = log(N / (1 + n_j))`：當一篇文件只有 **2 句**、且兩句完全沒有
共同詞彙時，每個詞的 `n_j`（出現在幾句）必然是 1，`log(2/2) = 0`——
**不分該句內容是否豐富**，全文件每個詞的 idf 都變 0，導致整個相似度矩陣
（含對角線的自我相似度）全部塌陷成 0，`power_method` 逐次迭代做
L2-norm 正規化時除以零，得到 `NaN`（numpy 只警告，不拋例外）。

**全量掃描 `multi_news_validation_canonical.jsonl`（5,621 篇，456,942 句，
79.9 秒，用 `_compute_idf` 直接檢查「全文件每個詞 idf 是否都是 0」，
不需要建完整 `O(n²)` 矩陣）**：

- 命中：**2 / 5,621 篇（0.036%）**——`validation_1082`、`validation_2303`，
  兩篇都恰好是 2 句、兩句零共同詞彙（前者是網路爬蟲 boilerplate，
  後者是 YouTube 訂閱樣板文字，皆屬已知的資料品質雜訊，與 F-12 同類）。
- 誤導性強的次要訊號：「單句 stopword 過濾後有效詞為零」的句子有
  **10,364 句、分布在 2,818 篇文件（50%）**——但這個數字**不能**當作
  退化風險的代理指標：絕大多數這種句子與文件內其他正常句子共存，
  文件整體不會塌陷。真正決定塌陷與否的是「全文件」層級的 idf，
  不是任何單一句子的狀態。

**GovReport 風險判斷（設計問題，只回報判斷，不實作）**：
GovReport 文件本身通常很長（非 2 句的短文件），所以「整篇 2 句零共同
詞彙」這個**具體**觸發條件在 GovReport 上大概率是罕見的——但這不代表
GovReport 沒有風險：GovReport 已知含大量標題／頁碼／表格殘留這類雜訊句，
若某篇文件被前處理誤判成極短（例如解析錯誤只留下 2-3 句雜訊），
仍可能踩到同一個退化。**建議：改用 F-17 的 reason code 路線，而非
fail-loud 中止整批**，理由：
1. 這個失敗不是 config/schema/programming error（CLAUDE.md 對這三類仍要求
   fail loud 中止），而是**特定文件的資料特性**——與
   `source_no_eligible_sentence`／`candidate_capacity_shortfall` 等既有
   F-17 reason code 屬於同一類。
2. F-17（PR #12）存在的唯一理由就是「不要讓一篇文件的失敗，作廢前面已經
   算完的 N-1 篇」。本次 LexRank 全量 run 在第 1082 篇（跑了 36 分鐘）
   撞到目前的 fail-loud 設計後，**整批作廢、沒有任何 predictions.jsonl**——
   與 F-17 修復前 greedy 在 `validation_4066` 撞見的失敗形狀一模一樣，
   只是換了一個新的程式碼路徑（baseline）重蹈覆轍。
3. Multi-News 上兩篇撞見的文件本身就是已知的資料雜訊——把它們記成
   `infeasible_code`（例如 `lexrank_similarity_graph_degenerate`）、
   分數記 0 或直接排除該篇繼續跑，比中止整批更能保留「這個方法在雜訊
   文件上會退化」這個研究事實本身。

**已實作（2026-08-05）**：先驗證了一件事，讓輸出問題本身消失——
`validation_1082`（59 字）與 `validation_2303`（77 字）**兩篇在 250 字
上界下兩句都放得進去**。兩句都放得進代表任何 scorer（不管怎麼排序）
都會給出同一個答案（兩句全取），`NaN` 只是程式炸了，輸出本身沒有歧義。
因此最終沒有走 `infeasible_code`（那個欄位的不變量是「`feasible=True`
時必為 `None`」，而這兩篇的 length/sentence-count 限制都完全滿足，並非
真的不可行）——改為一對新欄位，行為與既有的
`min_words_relaxed`／`relaxation_reason` 同一種「永遠記錄有沒有發生」慣例：

- `scorer_degenerate`（bool，一般文件為 `False`）
- `scorer_degenerate_reason`（`str` 或 `None`）

行為分兩支，都不靜默：
- **兩句都放得進**（Multi-News 兩個實例都屬此類）：以 placeholder 均一分數
  跑 `select_by_score`，退化成「按文件順序、skip-tolerant 全取」；
  若真的全取成功，代表排序從未影響過結果，`scorer_degenerate=True`，
  繼續（不中止整批）。
- **放不進**（Multi-News 尚未觀察到，但程式碼保留這個分支）：哪些句子該
  被丟掉真的取決於一個這個方法產生不出來的排序，**仍然 fail loud**——
  這是懸而未決的政策問題，不用猜的。

程式碼位置：`src/baselines/centrality.py`
`summarize_one_centrality`／`_score_document_by_original_index`；測試見
`tests/test_baselines_centrality.py` 的四個對應案例（兩句都放得進、
不放得進、一般文件 `scorer_degenerate=False`、N=1 邊界，見下）。

**N=1 代數邊界**：`idf(term) = log(1/(1+1)) = log(0.5)`，是**負值**，
不是 0——與塌陷完全不同的病理。因為 `cosine_similarity` 只用到
`idf²`，正負號不影響結果，單句文件的自我相似度恆為 `1.0`，正常給分、
不觸發 `scorer_degenerate`（已用合成單句文件實測確認）。**canonical
Multi-News validation 掃描確認句數下限是 2，沒有任何單句文件**——
這個邊界目前對 Multi-News 是純理論，留給 GovReport 或其他資料集
的前置檢查參考。

## O(n²) 成本模型（TextRank/LexRank，2026-08-05 實測）

`_create_matrix`（TextRank 與 LexRank 共用這個複雜度特徵）是逐對句子的
純 Python 迴圈，成本隨句數 `n` 大約呈 `O(n²)`。**兩個獨立、單一 process
（無並發）量到的校準點**：

| n（句數） | 量測時間（TextRank，隔離單一 process） | 隱含常數 c（秒／句²） |
|---|---|---|
| 3,347（`validation_2284`，全量最大文件） | 32.21 秒 | 2.88e-6 |
| 30 篇混合批次（句數 17–439，Σn²=492,930） | 2.21 秒 | 4.48e-6 |

兩個常數差約 1.56 倍（小文件的固定 overhead——`Sentence`/`Paragraph`
建構、tokenizer 呼叫——在小 n 時佔比更高，拉高隱含常數），粗估
`c ≈ 3–4.5e-6` 秒／句²，即 `wall_time ≈ c · n²`。LexRank 額外做
TF/IDF 字典運算，30 篇批次量到比 TextRank 慢 **1.46 倍**
（107.9ms vs 73.8ms／篇平均）。

**Multi-News validation 的句數分布**（5,621 篇）：mean 81.3、p50 61、
p90 150、**p95 214**、p99 385、**max 3,347**。

**外推規則**：句數變 3 倍 → 成本變約 9 倍（`O(n²)`）。GovReport 文件
通常比 Multi-News 長得多；下載後應**先跑句數分布**，用上表常數直接
估算最大/p95 文件的單篇秒數，再決定要不要對 outlier 文件加時間上限或
另外排程，而不是跑到一半才發現卡在某一篇。

**風險在 max，不在 mean**：Multi-News 平均文件（81.3 句）成本約
`4.5e-6 × 81.3² ≈ 0.030 秒`，可忽略；32.21 秒全部來自那篇 3,347 句的
單一文件。這代表**用平均句數估算總成本會嚴重低估**——真正決定單篇
是否會拖垮一次 run 的是分布的尾端（p95／p99／max），不是 mean。

### GovReport 下載後的第一個檢查（可執行，不是等跑到一半才發現）

```python
import json, statistics

counts = []
with open("<govreport_processed_path>.jsonl", encoding="utf-8") as f:
    for line in f:
        d = json.loads(line)
        n = sum(len(sec["sentences"]) for doc in d["documents"] for sec in doc["sections"])
        counts.append(n)

counts.sort()
def pct(s, p):
    return s[min(len(s) - 1, int(round(p / 100 * (len(s) - 1))))]

print("n docs:", len(counts))
print("mean:", statistics.mean(counts))
print("p50:", pct(counts, 50), "p95:", pct(counts, 95), "p99:", pct(counts, 99), "max:", counts[-1])

C_LOW, C_HIGH = 2.88e-6, 4.48e-6 * 1.46  # TextRank low end .. LexRank high end
for label, n in [("p95", pct(counts, 95)), ("p99", pct(counts, 99)), ("max", counts[-1])]:
    print(f"{label} (n={n}): estimated {C_LOW*n*n:.1f}s - {C_HIGH*n*n:.1f}s per document")
```

**建議門檻：單篇估算成本超過 60 秒，觸發方法層面的決定，而非在實作時
順手處理**。理由：

- 這不是隨便選的整數——**Multi-News 現有的單一最大文件（3,347 句）本身
  的實測成本（32.21 秒）已經逼近這個門檻的一半**，代表這個閾值是貼著
  「本專案已經實際運行過、且僅發生一次」的邊界設的，不是憑空拍的數字。
- 門檻定得更低（例如 5 秒，約對應 n≈1,050–1,300）會讓 Multi-News 自己
  p95（214 句）以上、目前已知運作正常的一大段文件都被標記，訊噪比太差。
- 門檻定得更高（例如 10 分鐘）會讓一篇文件在完全不示警的情況下拖住
  一整批 run 長達數分鐘——如果 GovReport 裡這種文件不只一篇（已知它
  比 Multi-News 長得多），未示警的總拖累可能是數十分鐘到數小時，
  而且是在 run 途中才發現。
- 60 秒是「單篇本身還不到需要重新設計 pipeline 的地步，但已經足以要求
  一個**方法層面**、寫進論文/文件的決定，而不是實作時的權宜」的量級。

**觸發後的決定必須是方法層面、明寫進文件的，不是實作時的權宜**（三選一，
或組合）：
1. **稀疏化**：改用 `src/features/graph.py` 已有的
   `build_sparse_tfidf_knn_graph`（有界 kNN，非 dense `O(n²)`）作為
   TextRank/LexRank 的相似度圖，而非 sumy 內建的 dense 矩陣——但這樣做
   會偏離「用第三方 sumy 實作、避免自己重新造輪子」的整個理由（見
   `src/baselines/centrality.py` 模組 docstring 開頭），需要重新評估
   reviewer-credibility 的權衡是否仍然成立。
2. **句數上限**：對超過某個句數的文件，只對前 N 句（或以某種抽樣）跑
   centrality，但這會讓 TextRank/LexRank 不再是文獻定義的「對全文件
   排序」，必須明寫這個偏離並說明理由。
3. **接受成本**：如果 outlier 文件數量少（如 Multi-News 只有 1 篇撞到
   32 秒等級），直接接受，只需要把預期總 wall-time 寫進文件（如本文件
   對 Multi-News 做的），不需要改方法。

三選一的判斷必須根據 GovReport 實際的句數分布（尤其是 p99/max 的頻率，
不是只看單一 max）決定，屬於下載資料後、動手實作前的方法決策，本文件
只給檢查腳本與門檻，不預先替 GovReport 做決定。

⚠️ **方法學警告，不是另一個校準點**：本文件曾經同時背景平行跑
TextRank 與 LexRank 兩個全量 5,621 篇的 process，兩者搶同一批 CPU
核心。那次跑出的 `time_select_seconds.txt`（475 秒）與同一次 run 的
`tqdm` 累計時間、檔案時間戳記（換算約 40.5 分鐘）**互相矛盾超過 5 倍**，
且矛盾方向（並發本應使 wall-clock *拉長*而非縮短）無法用單純 CPU
contention 解釋清楚。**這個數字不可信、不可用於外推**，本節的常數
只採用上方兩個隔離、單一 process 的乾淨量測。若要精確的全量 wall-clock
基準，必須重新用單一 process 個別重跑（本文件尚未做這件事）。

## CI 綠不等於全部測試執行（2026-08-05 查證）

`pytest -q` 在 CI 與本機都回報 `269 passed, 4 skipped`（加上 14 個原本因
vendored 資料缺失而 failed、修好後轉為 passed，合計 287）。**查證結果：
那 4 個 skip 與 canonical 資料無關**，是 `tests/test_pipeline_integration.py`
與 `tests/test_objective_evaluator.py` 裡 4 處 `pytest.importorskip("pymoo")`——
`requirements-ci.txt`（CI 用的輕量依賴集）刻意不含 `pymoo`（該檔自己的
註解說明：「Full research runs still require requirements.txt (including
torch/transformers/pymoo)」），本機因為裝了完整 `requirements.txt` 才會
全部執行、0 skip。這 4 個測試本身只用內建的合成 fixture（`sample_doc`／
手造 `evaluator`），跟是否有 canonical 資料完全無關，是既有的、與本
PR 無關的 CI/本機依賴集差異。

**結論記錄下來，供之後查閱**：CI 綠不代表 287 個測試都真的執行過——
在只裝 `requirements-ci.txt` 的環境（如 CI）跑，NSGA-II 相關的 4 個測試
會被跳過而非執行。若之後任何人改動 `nsga2.py`／pymoo 介接，CI 綠也
不能當作那 4 個測試通過的證據，必須用完整 `requirements.txt` 另外驗證。

## 版本

| 套件 | Pin | 備註 |
|---|---|---|
| `nltk` | `==3.10.0` | 見上方「nltk 版本 pin」；升版前必須重跑本文件的三個實測 |
| `sumy` | `==0.12.0` | Apache-2.0；依賴與授權見 repo 根目錄 `README.md`「授權」一節 |
