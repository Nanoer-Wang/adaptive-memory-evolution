param(
    [int]$Epochs = 5,
    [int]$Limit = 5,
    [switch]$Online,
    [switch]$Judge
)
$ErrorActionPreference = 'Stop'
$ProjectRoot = Split-Path -Parent $PSScriptRoot
Set-Location -LiteralPath $ProjectRoot
$PythonExe = Join-Path $ProjectRoot '.venv/Scripts/python.exe'
if (-not (Test-Path -LiteralPath $PythonExe)) { throw 'Create .venv and install requirements first; see README.md.' }
function Run-Lcm {
    & $PythonExe -m src.cli @args
    if ($LASTEXITCODE -ne 0) { throw "LCM stopped (exit $LASTEXITCODE). Completed work is saved." }
}
Run-Lcm download
Run-Lcm prepare
foreach ($Method in @('lcm', 'gru', 'transformer')) {
    $TrainingArgs = @('train', '--method', $Method, '--epochs', "$Epochs")
    if (Test-Path -LiteralPath "artifacts/checkpoints/$Method/last.pt") { $TrainingArgs += '--resume' }
    Run-Lcm @TrainingArgs
}
foreach ($Method in @('no_memory', 'recent', 'tfidf', 'hash_rag', 'lcm', 'gru', 'transformer')) {
    Run-Lcm evaluate --method $Method --retrieval-only
}
if ($Online) {
    Run-Lcm doctor --online
    foreach ($Method in @('no_memory', 'tfidf', 'lcm')) {
        $EvaluationArgs = @('evaluate', '--method', $Method, '--split', 'val', '--limit', "$Limit")
        if ($Judge) { $EvaluationArgs += '--judge' }
        Run-Lcm @EvaluationArgs
    }
}
