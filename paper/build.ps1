$ErrorActionPreference = 'Stop'
Push-Location $PSScriptRoot
try {
    New-Item -ItemType Directory -Force -Path 'build' | Out-Null
    $paperTectonic = Join-Path $PSScriptRoot 'tools/tectonic/tectonic.exe'
    if (Test-Path -LiteralPath $paperTectonic) {
        & $paperTectonic --keep-logs --keep-intermediates main.tex -o build
    } elseif (Get-Command tectonic -ErrorAction SilentlyContinue) {
        & tectonic --keep-logs --keep-intermediates main.tex -o build
    } elseif (Get-Command latexmk -ErrorAction SilentlyContinue) {
        & latexmk -pdf -interaction=nonstopmode -halt-on-error -outdir=build main.tex
    } else {
        throw 'Install Tectonic or a TeX distribution with latexmk, or compile main.tex on Overleaf.'
    }
    if ($LASTEXITCODE -ne 0) { throw "LaTeX compilation failed (exit $LASTEXITCODE)." }
    Copy-Item -LiteralPath 'build/main.pdf' -Destination 'main.pdf' -Force
    Write-Output 'Compiled manuscript: main.pdf'
} finally {
    Pop-Location
}
