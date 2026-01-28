$ErrorActionPreference = "Stop"
$Dataset = "data/processed/multi_news_test.jsonl"
$UnionInput = "runs/tuning_experiments/union_k20.jsonl"
$RunDir = "runs/tuning_experiments"

# Check if union exists
if (-not (Test-Path $UnionInput)) {
    Write-Error "Cannot find union_k20.jsonl. Please run utils_fusion first."
    exit 1
}

$Tests = @(
    @{ Name="ExpA_K20_BERT_Heavy"; Config="configs/2_Fusion_ExpA.yaml" },
    @{ Name="ExpB_K20_Max_Coverage"; Config="configs/2_Fusion_ExpB.yaml" },
    @{ Name="ExpC_K20_High_Precision"; Config="configs/2_Fusion_ExpC.yaml" }
)

Write-Host "Starting Rapid Tuning (K=20) on $($Tests.Count) Experiments..." -ForegroundColor Cyan

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

Write-Host "`nAll K=20 experiments done." -ForegroundColor Cyan
