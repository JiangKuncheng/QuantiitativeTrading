@echo off
rem ============================================================
rem  One-click: real-data backtest (default: 000001 Ping An Bank)
rem  Usage: double-click, or run_demo.bat --symbol 600519 --fast 10 --slow 30
rem  NOTE: keep this file ASCII-only (cmd uses GBK codepage on zh-CN)
rem ============================================================

set "CONDA_BAT=F:\tool\anaconda\Scripts\activate.bat"
if not exist "%CONDA_BAT%" (
    echo [ERROR] Anaconda activate.bat not found. Please run manually:
    echo   conda activate QuantitativeTrading ^&^& python run_demo.py
    pause
    exit /b 1
)

call "%CONDA_BAT%" QuantitativeTrading
set PYTHONIOENCODING=utf-8
python run_demo.py %*
pause
