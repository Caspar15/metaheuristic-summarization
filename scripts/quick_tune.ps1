$ErrorActionPreference = "Stop"
$Dataset = "data/processed/multi_news_test.jsonl"
# Use the EXISTING union file from the previous full run
$UnionInput = "runs/full_benchmark_result/union.jsonl"
$RunDir = "runs/tuning_experiments"

# Check if union exists
if (-not (Test-Path $UnionInput)) {
    Write-Error "Cannot find union.jsonl at run/full_benchmark_result. Please run benchmark first."
    exit 1
}

New-Item -ItemType Directory -Force -Path $RunDir | Out-Null

$Tests = @(
    @{ Name="ExpA_BERT_Heavy"; Config="configs/2_Fusion_ExpA.yaml" },
    @{ Name="ExpB_Max_Coverage"; Config="configs/2_Fusion_ExpB.yaml" },
    @{ Name="ExpC_High_Precision"; Config="configs/2_Fusion_ExpC.yaml" }
)

Write-Host "Starting Quick Tuning on $($Tests.Count) Experiments..." -ForegroundColor Cyan

foreach ($t in $Tests) {
    $Name = $t.Name
    $Cfg = $t.Config
    Write-Host "`n[Running] $Name..." -ForegroundColor Yellow
    
    $PredFile = "$RunDir/$Name/predictions.jsonl"
    if (-not (Test-Path $PredFile)) {
        python -m src.pipeline.select_sentences `
            --config $Cfg `
            --split test `
            --input $UnionInput `
            --run_dir $RunDir `
            --stamp $Name `
            --optimizer fast_nsga2
    } else {
        Write-Host "  > Skipping Prediction (Exists)"
    }

    # Verify output exists
    $PredFile = "$RunDir/$Name/predictions.jsonl"
    if (Test-Path $PredFile) {
        Write-Host "  > Evaluating..."
        python -m src.pipeline.evaluate `
            --pred $PredFile `
            --out "$RunDir/$Name/metrics.csv"
            
        $Res = Get-Content "$RunDir/$Name/metrics.csv" | Select-Object -Skip 1 | Select-Object -First 3
        Write-Host "  > Result: $Res" -ForegroundColor Green
    } else {
        Write-Error "Prediction failed for $Name"
    }
}

Write-Host "`nAll experiments done. Check $RunDir for results." -ForegroundColor Cyan
