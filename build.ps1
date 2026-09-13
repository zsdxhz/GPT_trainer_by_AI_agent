# =============================================================================
#  gpt_trainer one-click build script (Windows + VS2026 MSVC 14.42 toolset)
#  - Torch: bundled libtorch-win-shared-with-deps-2.14.0+cu130 (CUDA 13.0)
#  - Qt:    qt/6.9.3/msvc2022_64 (installed by aqtinstall)
#  - CUDA:  no nvcc calls at all - the cu130 runtime is bundled with libtorch
#  Usage:  powershell -ExecutionPolicy Bypass -File build.ps1 [-Fast]
#  NOTE: keep this file ASCII-only; PS 5.1 reads UTF-8 without BOM as GBK.
# =============================================================================
param([switch]$Fast)
$ErrorActionPreference = 'Stop'

$root = Split-Path -Parent $MyInvocation.MyCommand.Path

# ---- Locate toolchain (all absolute paths, see RTX build.ps1 notes) ----
# Use MSVC 14.44: it is the newest toolset that CUDA 13.0 officially supports
# (_MSC_VER 1944 < 1950), and its CRT already provides the ABI symbols
# (__std_search_1 etc.) that sentencepiece/abseil (built with the default
# 14.51 toolset) require. We compile no CUDA code - nvcc only runs CMake's
# compiler-id probe.
$msvcVer = '14.44'
$vsCmake = 'D:\vs2026\Common7\IDE\CommonExtensions\Microsoft\CMake\CMake\bin\cmake.exe'
$vsNinja = 'D:\vs2026\Common7\IDE\CommonExtensions\Microsoft\CMake\Ninja\ninja.exe'
$vcvars  = 'D:\vs2026\VC\Auxiliary\Build\vcvars64.bat'
if (-not (Test-Path $vsCmake)) { throw "cmake not found: $vsCmake" }
if (-not (Test-Path $vsNinja)) { throw "ninja not found: $vsNinja" }
if (-not (Test-Path $vcvars))  { throw "vcvars64.bat not found: $vcvars" }

# ---- ASCII junctions for link inputs ----
# MSVC link.exe reads Ninja response files in the ANSI codepage, so the
# Chinese project paths must not appear in the link line.  Create junctions
# with ASCII paths (mklink /J needs no admin rights):
#   E:\cpptrn\libtorch -> E:\project\cpp训练器\libtorch
#   E:\cpptrn\qt       -> E:\project\cpp训练器\qt
#   E:\cpptrn\zlib     -> E:\project\cpp训练器\third_party\zlib
#   E:\cpptrn\sp       -> E:\project\cpp分词器\sentencepiece-0.2.2\build
$junctions = @{
  'E:\cpptrn\libtorch' = (Join-Path $root 'libtorch')
  'E:\cpptrn\qt'       = (Join-Path $root 'qt')
  'E:\cpptrn\zlib'     = (Join-Path $root 'third_party\zlib')
  'E:\cpptrn\sp'       = 'E:\project\cpp分词器\sentencepiece-0.2.2\build'
}
New-Item -ItemType Directory -Force 'E:\cpptrn' | Out-Null
foreach ($j in $junctions.Keys) {
  if (-not (Test-Path $j)) {
    if (-not (Test-Path $junctions[$j])) { throw "junction target missing: $($junctions[$j])" }
    cmd /c mklink /J `"$j`" `"$($junctions[$j])`" | Out-Null
    if (-not (Test-Path $j)) { throw "failed to create junction: $j" }
  }
}

$libtorch = 'E:\cpptrn\libtorch'
if (-not (Test-Path (Join-Path $libtorch 'lib\torch_cpu.dll'))) {
  throw "libtorch not extracted: $($junctions['E:\cpptrn\libtorch'])"
}
$qtDir = 'E:\cpptrn\qt\6.9.3\msvc2022_64'
if (-not (Test-Path (Join-Path $qtDir 'lib\cmake\Qt6\Qt6Config.cmake'))) {
  throw "Qt not installed: $($junctions['E:\cpptrn\qt'])"
}

Write-Host "== Configure (CMake) ==" -ForegroundColor Cyan
# CUDA: force the v13.0 toolkit by absolute path (v12.4 also installed; CMake
# would otherwise pick v12.4 first). The MSVC 14.51 toolset is newer than what
# CUDA 13.0 officially supports (_MSC_VER 1951 >= 1950), so pass
# -allow-unsupported-compiler for the CMake compiler-id probe; this project
# compiles no .cu code at all (cu130 runtime comes from libtorch).
$cudaRoot = 'C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v13.0'
$cudaArgs = "-DCUDAToolkit_ROOT=`"$cudaRoot`" -DCMAKE_CUDA_COMPILER=`"$cudaRoot/bin/nvcc.exe`" -DCMAKE_CUDA_FLAGS=-allow-unsupported-compiler"
$cfg = "call `"$vcvars`" -vcvars_ver=$msvcVer >nul && `"$vsCmake`" -S `"$root`" -B `"$root\build`" -G Ninja -DCMAKE_BUILD_TYPE=Release -DCMAKE_MAKE_PROGRAM=`"$vsNinja`" $cudaArgs"
cmd /c $cfg
if ($LASTEXITCODE -ne 0) { throw "CMake configure failed" }

Write-Host "== Build ==" -ForegroundColor Cyan
# Chinese-locale cl.exe prints /showIncludes with a Chinese prefix that Ninja's
# msvc dependency parser cannot understand, so header-only edits are not
# tracked. Delete the target dir for a full rebuild (use -Fast to skip).
if (-not $Fast) {
  Remove-Item "$root\build\CMakeFiles\gpt_trainer.dir" -Recurse -Force -ErrorAction SilentlyContinue
}
cmd /c "call `"$vcvars`" -vcvars_ver=$msvcVer >nul && `"$vsCmake`" --build `"$root\build`""
if ($LASTEXITCODE -ne 0) { throw "Build failed" }

Write-Host "Build OK: $root\build\gpt_trainer.exe" -ForegroundColor Green
Write-Host "Run: .\run.bat  (GUI)   |   .\run.bat --cli  (CLI)" -ForegroundColor Green
