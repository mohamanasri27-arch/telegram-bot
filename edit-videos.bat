@echo off
cd /d "%~dp0"
title Automatic Video Editor

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
    echo   The video editor cannot do anything without it.
    echo   Open PowerShell and run this one line:
    echo.
    echo       winget install Gyan.FFmpeg
    echo.
    echo   Then CLOSE this window, open it again, and
    echo   double-click edit-videos.bat once more.
    echo ============================================
    pause
    exit /b 1
)

if not exist videos-in mkdir videos-in
if not exist videos-out mkdir videos-out
if not exist assets mkdir assets

echo ============================================
echo   Automatic Video Editor
echo.
echo   Put videos into the  videos-in  folder.
echo   Edited videos appear in  videos-out.
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
