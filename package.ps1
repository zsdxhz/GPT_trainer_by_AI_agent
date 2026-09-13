# =============================================================================
#  package.ps1 - 打包为自包含的 Release 目录（免安装、免控制台窗口）
#
#  产物: <root>\Release\
#    gpt_trainer.exe          主程序（WIN32 子系统，双击运行不弹黑框）
#    Qt6*.dll / z.dll         Qt 与 zlib 运行时
#    platforms\ styles\       Qt 平台/样式插件
#    *.dll (torch/cuda)       libtorch cu130 全部运行时 DLL（与 exe 同目录，
#                             无需 PATH，双击即用）
#    使用说明.txt
#
#  用法:  powershell -ExecutionPolicy Bypass -File package.ps1 [-SkipBuild]
#  NOTE: keep this file ASCII-only; PS 5.1 reads UTF-8 without BOM as GBK.
# =============================================================================
param([switch]$SkipBuild)
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

# ---- 3. usage note (UTF-8 template kept in a separate file) ----
# NOTE: this script must stay ASCII-only (PS 5.1 reads UTF-8 w/o BOM as GBK),
# so the Chinese filename below is built from codepoints.
$noteName = (-join [char[]](0x4F7F, 0x7528, 0x8BF4, 0x660E)) + '.txt'
Copy-Item (Join-Path $root 'release_note.txt') (Join-Path $release $noteName) -Force

$size = (Get-ChildItem $release -Recurse -File | Measure-Object Length -Sum).Sum
Write-Host ""
Write-Host ("Release OK: {0}  ({1:N2} GB)" -f $release, ($size / 1GB)) -ForegroundColor Green
Write-Host "Double-click Release\gpt_trainer.exe to start (no console window)" -ForegroundColor Green
