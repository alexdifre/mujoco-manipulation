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
$outDir = Join-Path $root "experiments\ur10e_acados_grasp"
New-Item -ItemType Directory -Force -Path $outDir | Out-Null
$stamp = Get-Date -Format "yyyyMMdd_HHmmss"
$logPath = Join-Path $outDir "launch_ur10e_$stamp.log"

Push-Location $workdir
try {
    $graspArgs = @(
        "run_ur10e_acados_grasp.py",
        "--task-mode", "lift",
        "--start-above-cube",
        "--max-steps", "8000",
        "--horizon", "10",
        "--mpc-dt", "0.04",
        "--open-steps", "120",
        "--approach-target-speed", "0.0",
        "--descend-target-speed", "0.08",
        "--lift-target-speed", "0.10",
        "--approach-clearance", "0.003",
        "--grasp-z-offset", "0.025",
        "--lift-z-offset", "0.22",
        "--ee-pos-weight", "600",
        "--ee-z-weight", "700",
        "--ee-terminal-weight", "1300",
        "--ee-terminal-z-weight", "1500",
        "--q-weight", "0.05",
        "--qv-weight", "0.03",
        "--qf-weight", "30",
        "--qvf-weight", "0.1",
        "--delta-q-max", "0.08",
        "--delta-dq-max", "0.35",
        "--delta-tau-max", "32",
        "--tau-slew-rate", "400",
        "--delta-tau-cost", "0.025",
        "--ee-upright-weight", "0",
        "--ee-terminal-upright-weight", "0",
        "--reach-tol", "0.035",
        "--grasp-tol", "0.030",
        "--close-steps", "450",
        "--close-ramp-steps", "400",
        "--grasp-aperture-threshold", "0.133",
        "--latch-aperture-threshold", "0.135",
        "--grasp-latch-distance", "0.080",
        "--grasp-hold-extra-fraction", "0.04",
        "--regularization", "1e-5",
        "--target-y-offset", "0.0",
        "--target-x-offset", "0.0",
        "--out-dir", "..\experiments\ur10e_acados_grasp"
    )
    if (-not $Headless) {
        $graspArgs += "--viewer"
    }
    $output = & $conda run --no-capture-output -n robot_sim python -I @graspArgs 2>&1
    $exitCode = $LASTEXITCODE
    $output | Tee-Object -FilePath $logPath
    if ($exitCode -ne 0) {
        exit $exitCode
    }
}
finally {
    Pop-Location
}
