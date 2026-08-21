@echo off
cd /d "%~dp0"
title Instagram Reels Editor

if not exist venv (
    echo ERROR: Setup has not been run yet.
    echo Please double-click setup.bat first.
    pause
    exit /b 1
)

where ffmpeg >nul 2>nul
if errorlevel 1 (
    echo ============================================
    echo   ffmpeg is not installed.
    echo.
    echo   The editor cannot do anything without it.
    echo   Open PowerShell and run this one line:
    echo.
    echo       winget install Gyan.FFmpeg
    echo.
    echo   Then CLOSE this window, open it again, and
    echo   double-click reels-editor.bat once more.
    echo ============================================
    pause
    exit /b 1
)

if not exist videos-in mkdir videos-in
if not exist videos-out mkdir videos-out
if not exist assets mkdir assets

echo ============================================
echo   INSTAGRAM REELS EDITOR
echo.
echo   Put videos into the  videos-in  folder.
echo   Edited videos and Reels clips appear
echo   in the  videos-out  folder.
echo.
echo   Keep this window open while it works.
echo   Press Ctrl+C to stop.
echo ============================================
echo.

call venv\Scripts\activate.bat
python watch_videos.py

echo.
echo The editor stopped.
pause
