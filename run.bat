@echo off
cd /d "%~dp0"
title Telegram Translator Bot

if not exist venv (
    echo ERROR: Setup has not been run yet.
    echo Please double-click setup.bat first.
    pause
    exit /b 1
)

if not exist .env (
    echo ERROR: No .env file found.
    echo Please double-click setup.bat first to save your bot token.
    pause
    exit /b 1
)

echo ============================================
echo   Telegram Translator Bot
echo   Keep this window open while using the bot.
echo   Press Ctrl+C twice to stop it for good.
echo ============================================
echo.

call venv\Scripts\activate.bat

:restart
python bot.py

echo.
echo ============================================
echo   The bot stopped. Restarting in 10 seconds...
echo   Close this window now if you want it to stay stopped.
echo ============================================
timeout /t 10 /nobreak >nul
goto restart
