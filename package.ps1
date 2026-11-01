# =============================================================================
#  package.ps1 - build a self-contained, out-of-the-box Release folder
#
#  Output: <root>\Release\
#    gpt_trainer.exe          main app (WIN32 subsystem, no console window)
#    Qt6*.dll / z.dll         Qt and zlib runtime
#    platforms\ styles\       Qt platform/style plugins
#    *.dll (torch/cuda)       libtorch cu130 runtime DLLs (training + builtin engine)
#    msvcp140*.dll / vcruntime*.dll / concrt140.dll   VC++ runtime (no redist install)
#    llama.cpp\llama-server.exe   llama.cpp inference engine (static CPU build)
#    models\gguf\*.gguf       sample GGUF model (unless -NoSampleModel)
#    models\spm.model         sentencepiece tokenizer (for models trained here)
#    gpt_root.txt             portable marker ("." => data dirs live next to exe)
#    usage note (Chinese name, built from codepoints)
#
#  Usage:  powershell -ExecutionPolicy Bypass -File package.ps1 [-SkipBuild]
#          [-NoSampleModel] [-LlamaServer <path>] [-SampleModel <path>]
#  NOTE: keep this file ASCII-only; PS 5.1 reads UTF-8 without BOM as GBK and
#        non-ASCII bytes inside code/param blocks corrupt the token stream.
# =============================================================================
param(
  [switch]$SkipBuild,
  [switch]$NoSampleModel,
  [string]$LlamaServer = 'E:\project\llama.cpp-0.7.0\build\bin\llama-server.exe',
  [string]$LlamaServerCuda = 'E:\project\llama.cpp-0.7.0\build-cuda\bin\llama-server.exe',
  [string]$SampleModel = '',
  [string]$SampleSpm = ''
)
$ErrorActionPreference = 'Stop'

$root = Split-Path -Parent $MyInvocation.MyCommand.Path

# ---- 1. build ----
if (-not $SkipBuild) {
  & (Join-Path $root 'build.ps1')
  if ($LASTEXITCODE -ne 0) { throw "build failed" }
}

$exe = Join-Path $root 'build\gpt_trainer.exe'
if (-not (Test-Path $exe)) { throw "exe not found: $exe" }

# ---- 2. prepare Release ----
$release = Join-Path $root 'Release'
if (Test-Path $release) { Remove-Item $release -Recurse -Force }
New-Item -ItemType Directory -Force $release | Out-Null

Write-Host "== Copy app + Qt runtime ==" -ForegroundColor Cyan
Copy-Item $exe $release -Force
Copy-Item (Join-Path $root 'build\Qt6Core.dll'),
          (Join-Path $root 'build\Qt6Gui.dll'),
          (Join-Path $root 'build\Qt6Widgets.dll'),
          (Join-Path $root 'build\z.dll') $release -Force
Copy-Item (Join-Path $root 'build\platforms') $release -Recurse -Force
Copy-Item (Join-Path $root 'build\styles') $release -Recurse -Force

Write-Host "== Copy libtorch CUDA runtime DLLs (large) ==" -ForegroundColor Cyan
Copy-Item (Join-Path $root 'libtorch\lib\*.dll') $release -Force

# ---- 3. VC++ runtime (app-local, no redist install needed) ----
Write-Host "== Copy VC++ runtime DLLs ==" -ForegroundColor Cyan
$crtCandidates = Get-ChildItem 'D:\vs2026\VC\Redist\MSVC' -Directory -ErrorAction SilentlyContinue |
                 Where-Object { $_.Name -match '^\d+\.\d+' } | Sort-Object Name -Descending
$crtDir = $null
foreach ($c in $crtCandidates) {
  $d = Join-Path $c.FullName 'x64\Microsoft.VC143.CRT'
  if (Test-Path (Join-Path $d 'vcruntime140.dll')) { $crtDir = $d; break }
}
if ($crtDir) {
  $crtFiles = @('msvcp140.dll','msvcp140_1.dll','msvcp140_2.dll','msvcp140_atomic_wait.dll',
                'msvcp140_codecvt_ids.dll','vcruntime140.dll','vcruntime140_1.dll',
                'vcruntime140_threads.dll','concrt140.dll')
  foreach ($f in $crtFiles) {
    $src = Join-Path $crtDir $f
    if (Test-Path $src) { Copy-Item $src $release -Force }
  }
  Write-Host "   from: $crtDir" -ForegroundColor DarkGray
} else {
  Write-Warning "VC++ runtime redist not found; target machines may need MSVC redist installed"
}

# ---- 4. llama.cpp engines ----
#   llama.cpp\llama-server.exe  : CPU build (fallback, works on any machine)
#   llama-server.exe (root)     : CUDA build (preferred; finds cublas/cudart DLLs
#                                 that libtorch already ships in this folder)
Write-Host "== Copy llama.cpp engines ==" -ForegroundColor Cyan
if (Test-Path $LlamaServerCuda) {
  Copy-Item $LlamaServerCuda (Join-Path $release 'llama-server.exe') -Force
  $cudaDlls = Get-ChildItem -Path (Split-Path $LlamaServerCuda -Parent) -Filter '*.dll' -File -ErrorAction SilentlyContinue
  if ($cudaDlls) { foreach ($d in $cudaDlls) { Copy-Item $d.FullName $release -Force } }
  Write-Host ("   CUDA engine: {0:N1} MB (root, GPU inference)" -f ((Get-Item $LlamaServerCuda).Length / 1MB)) -ForegroundColor DarkGray
} else {
  Write-Warning "CUDA llama-server not found at $LlamaServerCuda; package will use CPU engine only"
}
if (Test-Path $LlamaServer) {
  $llamaDir = Join-Path $release 'llama.cpp'
  New-Item -ItemType Directory -Force $llamaDir | Out-Null
  Copy-Item $LlamaServer $llamaDir -Force
  $llamaDlls = Get-ChildItem -Path (Split-Path $LlamaServer -Parent) -Filter '*.dll' -File -ErrorAction SilentlyContinue
  if ($llamaDlls) {
    foreach ($d in $llamaDlls) { Copy-Item $d.FullName $llamaDir -Force }
  }
  $prevEap = $ErrorActionPreference
  $ErrorActionPreference = 'Continue'
  try { $verLine = (& $LlamaServer --version 2>&1 | Select-Object -First 1) } catch { $verLine = '' }
  $ErrorActionPreference = $prevEap
  Write-Host "   CPU engine:  $verLine (llama.cpp\)" -ForegroundColor DarkGray
} else {
  Write-Warning "CPU llama-server.exe not found at $LlamaServer"
}

# ---- 5. sample GGUF model + matching sentencepiece tokenizer ----
# -NoSampleModel: skip bundling a model (ship it as a separate asset -> main zip < 2GB)
if (-not $NoSampleModel) {
  if ([string]::IsNullOrWhiteSpace($SampleModel)) {
    $SampleModel = Join-Path $root 'models\gguf\model3.gguf'
  }
  if (Test-Path $SampleModel) {
    $ggufDir = Join-Path $release 'models\gguf'
    New-Item -ItemType Directory -Force $ggufDir | Out-Null
    Copy-Item $SampleModel $ggufDir -Force
    Write-Host ("== Copy sample GGUF: {0:N1} MB" -f ((Get-Item $SampleModel).Length / 1MB)) -ForegroundColor Cyan
  } else {
    Write-Host "== no sample GGUF found (users can export their own) ==" -ForegroundColor Yellow
  }
} else {
  Write-Host "== -NoSampleModel: model not bundled (ship it separately) ==" -ForegroundColor Yellow
}

# spm.model: needed to encode Chinese into token ids for models trained by this
# project; external models (e.g. Qwen BPE vocab) use llama.cpp's own tokenizer.
if ([string]::IsNullOrWhiteSpace($SampleSpm)) {
  $SampleSpm = Join-Path $root 'data\models\spm.model'
}
if (Test-Path $SampleSpm) {
  $modelsDir = Join-Path $release 'models'
  New-Item -ItemType Directory -Force $modelsDir | Out-Null
  Copy-Item $SampleSpm (Join-Path $modelsDir 'spm.model') -Force
  Write-Host ("== Copy matching spm.model: {0:N0} KB" -f ((Get-Item $SampleSpm).Length / 1KB)) -ForegroundColor Cyan
} else {
  Write-Host "== no spm.model found: llama.cpp chat needs one ==" -ForegroundColor Yellow
}

# ---- 6. portable marker: data dirs live next to the exe ----
# Content "." => project root == exe directory (models/logs/saves/data inside this folder)
Set-Content -Path (Join-Path $release 'gpt_root.txt') -Value '.' -Encoding ascii -NoNewline

# ---- 7. usage note (UTF-8 template kept in a separate file) ----
# Chinese filename built from codepoints so this script stays ASCII-only.
$noteName = (-join [char[]](0x4F7F, 0x7528, 0x8BF4, 0x660E)) + '.txt'
Copy-Item (Join-Path $root 'release_note.txt') (Join-Path $release $noteName) -Force

$size = (Get-ChildItem $release -Recurse -File | Measure-Object Length -Sum).Sum
Write-Host ""
Write-Host ("Release OK: {0}  ({1:N2} GB)" -f $release, ($size / 1GB)) -ForegroundColor Green
Write-Host "Double-click Release\gpt_trainer.exe to start (no console window, no extra install)" -ForegroundColor Green
