@echo off
chcp 65001 >nul
cd /d "%~dp0build"
rem libtorch DLLs (incl. CUDA 13 runtime) and Qt DLLs must be on PATH
set "PATH=%~dp0libtorch\lib;%~dp0qt\6.9.3\msvc2022_64\bin;%PATH%"
if "%~1"=="--cli" (
  gpt_trainer.exe --cli
) else (
  gpt_trainer.exe
)
pause
