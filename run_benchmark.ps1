# Multi-News Full Benchmark Script
# Runs the 3-Way Fusion Pipeline on the full test set.

$Dataset = "data/processed/multi_news_test.jsonl"
$RunDir = "runs/full_benchmark_result"

# Ensure Run Directory Exists
New-Item -ItemType Directory -Force -Path $RunDir

Write-Host "=========================================="
Write-Host "Starting Full Benchmark on Multi-News (Test Set)"
Write-Host "Dataset: $Dataset"
Write-Host "Output: $RunDir"
Write-Host "=========================================="

# 1. Run Stage 1: Base (NSGA-II)
Write-Host "`n[1/6] Running Stage 1: Base (NSGA-II)..."
python -m src.pipeline.select_sentences --config configs/1_Base_NSGA2.yaml --split test --input $Dataset --run_dir $RunDir --stamp base --optimizer nsga2
if ($LASTEXITCODE -ne 0) { Write-Error "Base run failed"; exit 1 }

# 2. Run Stage 1: LLM (BERT)
Write-Host "`n[2/6] Running Stage 1: LLM (BERT)..."
python -m src.pipeline.select_sentences --config configs/1_LLM_BERT.yaml --split test --input $Dataset --run_dir $RunDir --stamp llm --optimizer bert
if ($LASTEXITCODE -ne 0) { Write-Error "LLM run failed"; exit 1 }

# 3. Run Stage 1: Graph (TextRank)
Write-Host "`n[3/6] Running Stage 1: Graph (TextRank)..."
python -m src.pipeline.select_sentences --config configs/1_Graph_TextRank.yaml --split test --input $Dataset --run_dir $RunDir --stamp graph --optimizer greedy
if ($LASTEXITCODE -ne 0) { Write-Error "Graph run failed"; exit 1 }

# 4. Build Union
Write-Host "`n[4/6] Building Union Candidate Pool..."
python scripts/utils_fusion.py `
    --input $Dataset `
    --base_pred "$RunDir/base/predictions.jsonl" `
    --bert_pred "$RunDir/llm/predictions.jsonl" `
    --graph_pred "$RunDir/graph/predictions.jsonl" `
    --out "$RunDir/union.jsonl" `
    --cap 100
if ($LASTEXITCODE -ne 0) { Write-Error "Union build failed"; exit 1 }

# 5. Run Stage 2: Fusion (Fast NSGA-II)
Write-Host "`n[5/6] Running Stage 2: Fusion Optimization..."
python -m src.pipeline.select_sentences --config configs/2_Fusion_Final.yaml --split test --input "$RunDir/union.jsonl" --run_dir $RunDir --stamp final_summary --optimizer fast_nsga2
if ($LASTEXITCODE -ne 0) { Write-Error "Stage 2 failed"; exit 1 }

# 6. Evaluate
Write-Host "`n[6/6] Evaluating ROUGE Scores..."
python -m src.pipeline.evaluate --pred "$RunDir/final_summary/predictions.jsonl" --out "$RunDir/metrics.csv"

Write-Host "`n=========================================="
Write-Host "Benchmark Completed Successfully!"
Write-Host "Results saved to: $RunDir/metrics.csv"
Get-Content "$RunDir/metrics.csv"
Write-Host "=========================================="
