@echo off
setlocal enabledelayedexpansion

title FacePulse AI - Local Attendance Server

echo ==============================================================================
echo                      FacePulse AI - Local Launcher
echo ==============================================================================
echo.

:: Check if Python is installed and available in PATH
where python >nul 2>nul
if %ERRORLEVEL% neq 0 (
    echo [ERROR] Python was not found in your system PATH.
    echo Please install Python 3.9+ from https://www.python.org/downloads/
    echo and ensure "Add Python to PATH" is checked during installation.
    echo.
    pause
    exit /b 1
)

:: Run the Python launcher with all passed arguments
python run.py %*

if %ERRORLEVEL% neq 0 (
    echo.
    echo [WARNING] Server stopped with error code %ERRORLEVEL%.
    pause
)
