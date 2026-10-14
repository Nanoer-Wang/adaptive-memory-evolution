$ErrorActionPreference = 'Stop'
$ProjectRoot = Split-Path -Parent $PSScriptRoot
Set-Location -LiteralPath $ProjectRoot
$TestParent = Join-Path $ProjectRoot 'artifacts/test-tmp'
New-Item -ItemType Directory -Path $TestParent -Force | Out-Null
$TestRoot = Join-Path $TestParent ([guid]::NewGuid().ToString('N').Substring(0, 6))
& ./.venv/Scripts/python.exe -m pytest -q --tb=short --basetemp $TestRoot
exit $LASTEXITCODE
