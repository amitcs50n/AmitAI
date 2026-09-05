@echo off
setlocal
pushd "%~dp0..\.." || exit /b 1
if not exist ".venv\Scripts\python.exe" (
    echo The existing Aevon .venv is required. Complete the V1 Windows installation first.
    popd
    exit /b 1
)
".venv\Scripts\python.exe" -m scripts.startup windows %*
set "aevon_exit=%errorlevel%"
popd
exit /b %aevon_exit%
