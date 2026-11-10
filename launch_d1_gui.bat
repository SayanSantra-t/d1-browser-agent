@echo off
title D1-Omni System-1 Browser Agent
cd /d "%~dp0"
echo =======================================================
echo Starting D1-Omni System-1 Browser Agent GUI
echo Model: Liquid AI d1-omni-600M (NVIDIA CUDA)
echo Connected Browser: BrowserOS neo (CDP 127.0.0.1:9110)
echo =======================================================
python d1_browser_gui.py
if %ERRORLEVEL% NEQ 0 (
    echo.
    echo [ERROR] Application exited with code %ERRORLEVEL%.
    pause
)
