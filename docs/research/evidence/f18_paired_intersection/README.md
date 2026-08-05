# F-18 paired feasible intersection

這個目錄是 `scripts.audit.paired_run_intersection` 在 2026-08-05 產生的
versioned diagnostic output。它比較目前的兩份完整 Multi-News validation
predictions：

- greedy + `mean`：SHA-256
  `669f46c5859aee318d4ec2527c17296d12b30b80a2ea5c470195752a61e20bb8`
- greedy + `length_normalized`：SHA-256
  `1a254eb149ab6929381389e8150762b1a5103d7644fcee5e6a7a1be43ec02b33`

兩份 bulk predictions 分別約 495 MB、529 MB，不進 Git；
`intersection_report.json` 的相對 path 是原始本機執行位置，identity 以 SHA-256
為準。共同 feasible intersection 為 5,613 篇；兩個 `per_example.jsonl` 保存
逐篇 ROUGE，可作 paired bootstrap／Wilcoxon 的輸入。

目前**尚未執行顯著性檢定**。這些檔案證明 matched-denominator 平均值可重算，
不等於差異已達統計顯著。

產生器固定寫 LF，目錄內生成檔也由 `.gitattributes` 標記為 `text eol=lf`，
避免 Windows／Linux checkout 改寫換行而使 manifest 記錄的 raw-byte SHA-256
失效；README 本身不納入那些 artifact hashes。
