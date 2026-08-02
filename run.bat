@echo off
cd /d "%~dp0"

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
echo   Starting the bot...
echo   Keep this window open while using the bot.
echo   Press Ctrl+C to stop.
echo ============================================
echo.

call venv\Scripts\activate.bat
python bot.py

echo.
echo The bot has stopped.
pause
