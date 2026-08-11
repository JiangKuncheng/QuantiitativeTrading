@echo off
rem ============================================================
rem  One-click: smoke tests (data -^> strategy -^> backtest -^> output)
rem  NOTE: keep this file ASCII-only (cmd uses GBK codepage on zh-CN)
rem ============================================================

set "CONDA_BAT=F:\tool\anaconda\Scripts\activate.bat"
if not exist "%CONDA_BAT%" (
    echo [ERROR] Anaconda activate.bat not found. Please run manually:
    echo   conda activate QuantitativeTrading ^&^& python -m unittest discover -s tests -v
    pause
    exit /b 1
)

call "%CONDA_BAT%" QuantitativeTrading
set PYTHONIOENCODING=utf-8
python -m unittest discover -s tests -v
pause
