@echo off
setlocal

cd /d "%~dp0"

set "PYTHON_EXE=D:\vng\python.exe"
if exist "%PYTHON_EXE%" goto init_env
set "PYTHON_EXE=python"

:init_env
echo Starting Legal RAG Assistant...
echo Working directory: %cd%
echo Python: %PYTHON_EXE%
echo.

set "USE_TORCH=1"
set "USE_TF=0"

if "%~1"=="" goto run_desktop
if /I "%~1"=="desktop" goto run_desktop
if /I "%~1"=="app" goto run_desktop
if /I "%~1"=="eval-web" goto run_eval_web
if /I "%~1"=="streamlit" goto run_eval_web
if /I "%~1"=="eval-gui" goto run_harness_gui
if /I "%~1"=="eval-cli" goto run_harness_cli
if /I "%~1"=="menu" goto show_menu

echo Unknown mode: %~1
echo Supported modes: desktop, eval-web, eval-gui, eval-cli, menu
set "EXIT_CODE=1"
goto finish

:show_menu
echo 1. Start PySide6 desktop chat
echo 2. Start Streamlit evaluation dashboard
echo 3. Start Harness GUI
echo 4. Run Harness CLI
choice /c 1234 /n /m "Select mode [1/2/3/4]: "
if errorlevel 4 goto run_harness_cli
if errorlevel 3 goto run_harness_gui
if errorlevel 2 goto run_eval_web
goto run_desktop

:run_desktop
"%PYTHON_EXE%" legal_rag_desktop.py
set "EXIT_CODE=%ERRORLEVEL%"
goto finish

:run_eval_web
"%PYTHON_EXE%" -m streamlit run app.py
set "EXIT_CODE=%ERRORLEVEL%"
goto finish

:run_harness_gui
"%PYTHON_EXE%" legal_rag_harness_gui.py
set "EXIT_CODE=%ERRORLEVEL%"
goto finish

:run_harness_cli
"%PYTHON_EXE%" run_legal_rag_harness.py --modes hybrid llm_retrieval --output "%cd%\eval\reports\legal_rag_harness_latest.json"
set "EXIT_CODE=%ERRORLEVEL%"

:finish
if not "%EXIT_CODE%"=="0" (
    echo.
    echo Failed to run. Exit code: %EXIT_CODE%
    echo Check whether project dependencies and LLM configuration are available.
    pause
)

endlocal
