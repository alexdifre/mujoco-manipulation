param(
    [switch]$Headless
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

$root = $PSScriptRoot
$workdir = Join-Path $root "catkin_ws"
$conda = "C:\conda-forge\condabin\conda.bat"

if (-not (Test-Path -LiteralPath $conda)) {
    $cmd = Get-Command conda -ErrorAction SilentlyContinue
    if ($null -eq $cmd) {
        throw "Conda not found. Expected C:\conda-forge\condabin\conda.bat or conda in PATH."
    }
    $conda = $cmd.Source
}

$env:PYTHONNOUSERSITE = "1"
$numbaCache = Join-Path $root ".numba_cache"
New-Item -ItemType Directory -Force -Path $numbaCache | Out-Null
$env:NUMBA_CACHE_DIR = $numbaCache

$outDir = Join-Path $root "experiments\robosuite_pickplace_grasp"
New-Item -ItemType Directory -Force -Path $outDir | Out-Null
$stamp = Get-Date -Format "yyyyMMdd_HHmmss"
$logPath = Join-Path $outDir "launch_robosuite_pickplace_$stamp.log"

Push-Location $workdir
try {
    $graspArgs = @(
        "run_robosuite_pickplace_grasp.py",
        "--robot", "Panda",
        "--controller", "BASIC",
        "--arm-controller", "OSC_POSE",
        "--object-type", "can",
        "--max-steps", "1600",
        "--control-freq", "20",
        "--open-steps", "40",
        "--close-steps", "40",
        "--approach-clearance", "0.12",
        "--grasp-z-offset", "0.015",
        "--grasp-forward-offset", "0.010",
        "--lift-z-offset", "0.22",
        "--success-lift", "0.08",
        "--reach-tol", "0.025",
        "--grasp-tol", "0.020",
        "--grasp-latch-distance", "0.085",
        "--position-gain", "8.0",
        "--max-pos-action", "0.20",
        "--open-command", "-1.0",
        "--close-command", "1.0",
        "--out-dir", "..\experiments\robosuite_pickplace_grasp"
    )
    if (-not $Headless) {
        $graspArgs += "--viewer"
    }
    $output = & $conda run --no-capture-output -n robot_sim python @graspArgs 2>&1
    $exitCode = $LASTEXITCODE
    $output | Tee-Object -FilePath $logPath
    if ($exitCode -ne 0) {
        exit $exitCode
    }
}
finally {
    Pop-Location
}
