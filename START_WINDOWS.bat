@echo off
chcp 65001 >nul
cd /d "%~dp0"
where py >nul 2>nul
if %errorlevel% equ 0 (
    set PY=py -3
) else (
    set PY=python
)
%PY% -c "import numpy, PIL" >nul 2>nul
if errorlevel 1 (
    echo Installing Python dependencies...
    %PY% -m pip install -r requirements.txt
    if errorlevel 1 (
        echo Could not install dependencies. Check your internet connection and Python installation.
        pause
        exit /b 1
    )
)
%PY% main.py
if errorlevel 1 pause
