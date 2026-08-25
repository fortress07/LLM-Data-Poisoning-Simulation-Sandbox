$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $PSScriptRoot
Set-Location $root

python -m pip install -e . | Out-Null
Write-Host "installed poisonlab in editable mode"

python -m poisonlab doctor

if (Get-Command node -ErrorAction SilentlyContinue) {
  Write-Host "node found, the html viewer is available"
} else {
  Write-Host "node not found, reports stay in json and markdown"
}

Write-Host ""
Write-Host "next: poisonlab run configs/backdoor.toml --html"
