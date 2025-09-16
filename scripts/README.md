# scripts 目錄說明（已整理）

保留（現行使用）
- benchmark_small.py：小樣本基準（多個選句器）
- build_union_stage2.py：將 Stage1 Base 與 LLM 的 predictions.jsonl 建成聯集 U（支援 cap 與可選去重）
- grid_stage2_fast.py：Stage2 fast 系列（不依賴 BERT）的網格搜尋
- jsonl_to_csv.py：JSONL 轉 CSV
- organize_runs.py：整理 runs/ 成結構化目錄（支援 tune1/tune2/fast2 命名）
- run_7_2_1.py：一次性跑三個 split 的前處理→選句→評估（通用工具）
- split_dataset.py：切分單一 CSV 為 train/validation/test
- summarize_runs.py：彙總多份 tune_summary_*.csv
- tune_union_fusion.py：二階段調參與彙總（現行 Stage2 僅允許 fast|fast_grasp|fast_nsga2）
- run_two_stage_timed.py：新增，二階段整流程執行＋計時彙總（Stage1 Base/LLM 與 Stage2）

歸檔（歷史/不常用）
- _archive/time_validation.ps1：驗證耗時的舊 PowerShell 腳本（留存參考）

注意
- 已移除 Stage2 的 bert/fused 實作；調參與文件已更新為 Stage2 僅 fast 系列。
- organize_runs.py 能整理舊有 bert/fused 的 tune2 命名，也支援 fast2 命名。
