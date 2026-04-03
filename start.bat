@echo off
REM =============================================================
REM  Cyber Security Attacks Classifier — Launcher (Windows)
REM =============================================================

cd /d "%~dp0"

echo ============================================================
echo   Cyber Security Attacks Classifier
echo   Random Forest — ML Pipeline + Streamlit Dashboard
echo ============================================================
echo.

REM --- Find Python ---
echo Step 1/4: Checking Python installation...

set PYTHON=
where python3 >nul 2>&1 && set PYTHON=python3
if "%PYTHON%"=="" (
    where python >nul 2>&1 && set PYTHON=python
)
if "%PYTHON%"=="" (
    echo ERROR: Python is not installed or not in PATH.
    echo Please install Python 3.10+: https://www.python.org/downloads/
    pause
    exit /b 1
)

%PYTHON% --version
echo.

REM --- Create / activate virtual environment ---
echo Step 2/4: Setting up virtual environment...

if not exist "venv" (
    echo    Creating virtual environment...
    %PYTHON% -m venv venv
)

call venv\Scripts\activate.bat

echo    Installing dependencies...
pip install --upgrade pip 2>nul
pip install -r requirements.txt
echo    Dependencies installed
echo.

REM --- Run ML Pipeline ---
echo Step 3/4: Running ML Pipeline...
echo.
python pipeline.py

if %ERRORLEVEL% neq 0 (
    echo ERROR: Pipeline failed. Check the output above.
    pause
    exit /b 1
)

REM --- Launch Streamlit ---
echo.
echo Step 4/4: Launching Streamlit Dashboard...
echo.
echo    Dashboard will open in your browser at: http://localhost:8501
echo    Press Ctrl+C to stop the server.
echo.
streamlit run app.py --server.headless true

pause
