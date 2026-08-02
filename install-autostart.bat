@echo off
cd /d "%~dp0"

echo ============================================
echo   Auto-start setup
echo ============================================
echo.
echo This makes the bot start automatically every time
echo you log in to Windows.
echo.

set "STARTUP=%APPDATA%\Microsoft\Windows\Start Menu\Programs\Startup"
set "SHORTCUT=%STARTUP%\TelegramTranslatorBot.lnk"

powershell -NoProfile -Command ^
  "$s = (New-Object -ComObject WScript.Shell).CreateShortcut('%SHORTCUT%');" ^
  "$s.TargetPath = '%~dp0run.bat';" ^
  "$s.WorkingDirectory = '%~dp0';" ^
  "$s.WindowStyle = 7;" ^
  "$s.Description = 'Telegram Translator Bot';" ^
  "$s.Save()"

if errorlevel 1 (
    echo.
    echo ERROR: Could not create the startup shortcut.
    pause
    exit /b 1
)

echo Done. The bot will now start automatically when you log in.
echo.
echo To undo this later, delete this file:
echo %SHORTCUT%
echo.
pause
