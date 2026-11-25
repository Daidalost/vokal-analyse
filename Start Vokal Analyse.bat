@echo off
rem Vokal Analyse - double-click to start.
rem First start installs uv (Python manager), Python and the AI model libraries (a few GB, takes a while).
cd /d "%~dp0"
where uv >nul 2>nul
if errorlevel 1 (
  echo Installing uv, one time only...
  powershell -NoProfile -ExecutionPolicy Bypass -Command "irm https://astral.sh/uv/install.ps1 | iex"
  set "PATH=%USERPROFILE%\.local\bin;%PATH%"
)
echo Starting Vokal Analyse (the first start downloads everything it needs)...
uv run --python 3.12 python -m vokal_analyse
if errorlevel 1 (
  echo.
  echo Something went wrong. Copy the text above and send it to Claude.
  pause
)
