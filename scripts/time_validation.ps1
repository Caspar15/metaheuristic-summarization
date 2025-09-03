$ErrorActionPreference = 'Stop'

# Resolve Python (prefer project venv)
$pyPath = Join-Path (Get-Location) '.venv/Scripts/python.exe'
if (-Not (Test-Path $pyPath)) { $pyPath = 'python' }

$inputCsv = 'data/raw/cnn_dailymail/validation.csv'
$outJsonl = 'data/processed/validation.full.jsonl'
New-Item -ItemType Directory -Force -Path (Split-Path $outJsonl) | Out-Null

# Preprocess full validation (cap per-doc sentences for efficiency)
$tPre = Measure-Command {
  & $pyPath -m src.data.preprocess --input $inputCsv --split validation_full --out $outJsonl --max_sentences 25
}
Write-Host ('preprocess_seconds=' + [math]::Round($tPre.TotalSeconds, 2))

# Prepare stamps
$stampBase = Get-Date -Format 'yyyyMMdd-HHmmss'
$opts = @('greedy','grasp','nsga2')
$stamps = @{}
foreach ($o in $opts) { $stamps[$o] = ($stampBase + '-' + $o + '-validation_full') }

$summary = @()

foreach ($o in $opts) {
  $stamp = $stamps[$o]
  $tSel = Measure-Command {
    & $pyPath -m src.pipeline.select_sentences --config 'configs/features_3sent.yaml' --split 'validation' --input $outJsonl --run_dir 'runs' --stamp $stamp --optimizer $o
  }
  Write-Host ("select_${o}_seconds=" + [math]::Round($tSel.TotalSeconds, 2))

  $pred = Join-Path (Join-Path 'runs' $stamp) 'predictions.jsonl'
  $outm = Join-Path (Join-Path 'runs' $stamp) 'metrics.csv'
  $tEval = Measure-Command {
    & $pyPath -m src.pipeline.evaluate --pred $pred --out $outm
  }
  Write-Host ("evaluate_${o}_seconds=" + [math]::Round($tEval.TotalSeconds, 2))

  $summary += [PSCustomObject]@{ phase='preprocess'; optimizer=''; seconds=[math]::Round($tPre.TotalSeconds,2) }
  $summary += [PSCustomObject]@{ phase='select'; optimizer=$o; seconds=[math]::Round($tSel.TotalSeconds,2) }
  $summary += [PSCustomObject]@{ phase='evaluate'; optimizer=$o; seconds=[math]::Round($tEval.TotalSeconds,2) }
}

# Write CSV summary
New-Item -ItemType Directory -Force -Path 'runs' | Out-Null
$summaryPath = Join-Path 'runs' ($stampBase + '-time-summary.csv')
"phase,optimizer,seconds" | Out-File -Encoding utf8 $summaryPath
foreach ($row in $summary) { ("{0},{1},{2}" -f $row.phase, $row.optimizer, $row.seconds) | Add-Content -Encoding utf8 $summaryPath }
Write-Host ('summary_csv=' + $summaryPath)

