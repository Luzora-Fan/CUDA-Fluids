<#
.SYNOPSIS
Configures and builds CUDA Fluids with MSVC, Ninja, CMake, and NVCC.

.DESCRIPTION
CUDA on Windows needs a host C++ compiler that NVCC can find. This script uses
vswhere to locate Visual Studio, enters the x64 Visual Studio developer
environment, and then runs the CMake configure/build steps from that environment.

.PARAMETER BuildDir
Build output directory relative to the repository root unless an absolute path is
provided. Defaults to "build".

.PARAMETER CudaArchitectures
Optional CUDA architecture value passed to CMAKE_CUDA_ARCHITECTURES, such as
"75", "86", "89", "100", or "120".

.PARAMETER Clean
Deletes the selected build directory before configuring. Use this when CMake was
previously run from the wrong shell or with the wrong compiler.

.PARAMETER ConfigureOnly
Runs CMake configure/generate but skips the build step.

.EXAMPLE
.\scripts\build-msvc.ps1 -BuildDir build-live-clean -Clean

.EXAMPLE
.\scripts\build-msvc.ps1 -BuildDir build -CudaArchitectures 120
#>

param(
    [string]$BuildDir = "build",
    [string]$CudaArchitectures = "",
    [switch]$Clean,
    [switch]$ConfigureOnly
)

$ErrorActionPreference = "Stop"

$repoRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$buildPath = Join-Path $repoRoot $BuildDir

# vswhere is the supported way to find the latest installed Visual Studio from
# scripts without hard-coding Community/Professional/Enterprise paths.
$vswhere = Join-Path ${env:ProgramFiles(x86)} "Microsoft Visual Studio\Installer\vswhere.exe"
if (!(Test-Path $vswhere)) {
    throw "Could not find vswhere.exe. Install Visual Studio 2022 with the Desktop development with C++ workload."
}

$vsInstall = & $vswhere -latest -products * -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 -property installationPath
if (!$vsInstall) {
    throw "Could not find Visual Studio C++ tools. Install the Desktop development with C++ workload."
}

$devCmd = Join-Path $vsInstall "Common7\Tools\VsDevCmd.bat"
if (!(Test-Path $devCmd)) {
    throw "Could not find VsDevCmd.bat at $devCmd."
}

# Removing the build tree is safer than asking CMake to recover from a generator
# or compiler mismatch.
if ($Clean -and (Test-Path $buildPath)) {
    Remove-Item -LiteralPath $buildPath -Recurse -Force
}

# Build the configure command as a batch command so all work happens inside the
# Visual Studio developer environment initialized below.
$cmakeArgs = @("-S", "`"$repoRoot`"", "-B", "`"$buildPath`"", "-G", "Ninja")
if ($CudaArchitectures) {
    $cmakeArgs += "-DCMAKE_CUDA_ARCHITECTURES=$CudaArchitectures"
}

$configureCommand = "cmake $($cmakeArgs -join ' ')"
$buildCommand = "cmake --build `"$buildPath`""

# A temporary .cmd file avoids fragile nested quoting between PowerShell, cmd,
# VsDevCmd.bat, and paths containing spaces.
$batchPath = Join-Path ([System.IO.Path]::GetTempPath()) "cuda-fluids-build-$([System.Guid]::NewGuid()).cmd"
$batchLines = @(
    "@echo off",
    "call `"$devCmd`" -arch=x64",
    "if errorlevel 1 exit /b %errorlevel%",
    $configureCommand,
    "if errorlevel 1 exit /b %errorlevel%"
)

# Configure-only mode is useful when inspecting generated CMake cache values or
# IDE project metadata before compiling.
if (!$ConfigureOnly) {
    $batchLines += $buildCommand
    $batchLines += "if errorlevel 1 exit /b %errorlevel%"
}

try {
    Set-Content -LiteralPath $batchPath -Value $batchLines -Encoding ASCII
    & cmd.exe /d /c $batchPath
    if ($LASTEXITCODE -ne 0) {
        exit $LASTEXITCODE
    }
} finally {
    # Leave no generated helper files behind after the build completes or fails.
    if (Test-Path $batchPath) {
        Remove-Item -LiteralPath $batchPath -Force
    }
}
