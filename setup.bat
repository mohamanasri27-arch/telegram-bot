@echo off
cd /d "%~dp0"

echo ============================================
echo   Telegram Translator Bot - Setup
echo ============================================
echo.

echo [1/3] Creating virtual environment...
python -m venv venv
if errorlevel 1 (
    echo.
    echo ERROR: Could not create the virtual environment.
    echo Make sure Python is installed and "Add python.exe to PATH" was checked.
    pause
    exit /b 1
)

echo.
echo [2/3] Installing dependencies. This takes a few minutes, please wait...
call venv\Scripts\activate.bat
python -m pip install --upgrade pip
pip install -r requirements.txt
if errorlevel 1 (
    echo.
    echo ERROR: Could not install dependencies. Check your internet connection.
    pause
    exit /b 1
)

echo.
echo [3/3] Configuring your bot token...
if exist .env goto :token_exists

set /p TOKEN="Paste your Telegram bot token from BotFather and press Enter: "
(echo TELEGRAM_BOT_TOKEN=%TOKEN%)> .env
echo Token saved to .env
goto :done

:token_exists
echo A .env file already exists - keeping the current token.

:done
echo.
echo ============================================
echo   Setup complete!
echo   Now double-click run.bat to start the bot.
echo ============================================
pause
