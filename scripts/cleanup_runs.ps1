param(
  [string]$RunsDir = "runs",
  [string]$ArchiveDir = "runs/archive"
)

Write-Host "Scanning $RunsDir for run folders..."
$dirs = Get-ChildItem -Path $RunsDir -Directory | Where-Object { $_.Name -ne 'archive' }
if ($dirs.Count -eq 0) { Write-Host "No run directories found."; exit }

Write-Host "Select runs to archive (comma-separated indices):"
for ($i=0; $i -lt $dirs.Count; $i++) { Write-Host ("[{0}] {1}" -f $i, $dirs[$i].Name) }
$sel = Read-Host "Indices"
$idx = $sel -split "," | ForEach-Object { $_.Trim() } | Where-Object { $_ -match '^[0-9]+$' } | ForEach-Object { [int]$_ }
if ($idx.Count -eq 0) { Write-Host "No selection."; exit }

New-Item -ItemType Directory -Force -Path $ArchiveDir | Out-Null
foreach ($i in $idx) {
  if ($i -ge 0 -and $i -lt $dirs.Count) {
    $src = $dirs[$i].FullName
    $dst = Join-Path $ArchiveDir $dirs[$i].Name
    Write-Host "Moving $src -> $dst"
    Move-Item -Path $src -Destination $dst -Force
  }
}
Write-Host "Done."

