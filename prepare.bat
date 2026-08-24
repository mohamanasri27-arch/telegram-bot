@echo off
cd /d "%~dp0"
title Preparing the Reels Editor

if not exist venv (
    echo ERROR: Setup has not been run yet.
    echo Please double-click setup.bat first.
    pause
    exit /b 1
)

echo ============================================
echo   Preparing everything, once.
echo.
echo   This downloads the speech models (a few GB)
echo   and the Persian font, then edits a test
echo   video to prove the whole chain works.
echo.
echo   Keep your VPN on if you need one.
echo   If a download drops, just run this again -
echo   it resumes where it stopped.
echo ============================================
echo.

call venv\Scripts\activate.bat
python prepare.py

echo.
pause
