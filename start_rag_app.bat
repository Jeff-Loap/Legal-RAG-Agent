@echo off
setlocal EnableExtensions

cd /d "%~dp0"

set "PYTHON_EXE=D:\vng\python.exe"
if exist "%PYTHON_EXE%" goto init_env
set "PYTHON_EXE=python"

:init_env
set "STREAMLIT_PORT=8503"

echo Starting Legal RAG Assistant...
echo Working directory: %cd%
echo Python: %PYTHON_EXE%
echo Streamlit port: %STREAMLIT_PORT%
echo.

set "USE_TORCH=1"
set "USE_TF=0"

if "%~1"=="" goto run_all
if /I "%~1"=="all" goto run_all
if /I "%~1"=="desktop" goto run_desktop
if /I "%~1"=="app" goto run_desktop
if /I "%~1"=="eval-web" goto run_eval_web
if /I "%~1"=="streamlit" goto run_eval_web
if /I "%~1"=="eval-gui" goto run_harness_gui
if /I "%~1"=="eval-cli" goto run_harness_cli
if /I "%~1"=="menu" goto show_menu

echo Unknown mode: %~1
echo Supported modes: all, desktop, eval-web, eval-gui, eval-cli, menu
set "EXIT_CODE=1"
goto finish

:show_menu
echo 1. Start PySide6 desktop chat only
echo 2. Start Streamlit evaluation dashboard only
echo 3. Start both desktop chat and evaluation dashboard
echo 4. Start Harness GUI
echo 5. Run Harness CLI
choice /c 12345 /n /m "Select mode [1/2/3/4/5]: "
if errorlevel 5 goto run_harness_cli
if errorlevel 4 goto run_harness_gui
if errorlevel 3 goto run_all
if errorlevel 2 goto run_eval_web
goto run_desktop

:run_all
start "Legal RAG Desktop" "%PYTHON_EXE%" legal_rag_desktop.py
start "Legal RAG Eval" "%PYTHON_EXE%" -m streamlit run app.py --server.address 127.0.0.1 --server.port %STREAMLIT_PORT% --server.headless true --browser.gatherUsageStats false
call :wait_and_open_browser %STREAMLIT_PORT%
set "EXIT_CODE=0"
goto finish

:run_desktop
start "Legal RAG Desktop" "%PYTHON_EXE%" legal_rag_desktop.py
set "EXIT_CODE=0"
goto finish

:run_eval_web
start "Legal RAG Eval" "%PYTHON_EXE%" -m streamlit run app.py --server.address 127.0.0.1 --server.port %STREAMLIT_PORT% --server.headless true --browser.gatherUsageStats false
call :wait_and_open_browser %STREAMLIT_PORT%
set "EXIT_CODE=0"
goto finish

:run_harness_gui
start "Legal RAG Harness GUI" "%PYTHON_EXE%" legal_rag_harness_gui.py
set "EXIT_CODE=0"
goto finish

:run_harness_cli
"%PYTHON_EXE%" run_legal_rag_harness.py --modes hybrid llm_retrieval --output "%cd%\eval\reports\legal_rag_harness_latest.json"
set "EXIT_CODE=%ERRORLEVEL%"
goto finish

:wait_and_open_browser
set "PORT=%~1"
powershell -NoProfile -Command ^
  "$port = %PORT%; " ^
  "$ready = $false; " ^
  "for ($i = 0; $i -lt 60; $i++) { " ^
  "  try { " ^
  "    $client = New-Object Net.Sockets.TcpClient; " ^
  "    $client.Connect('127.0.0.1', $port); " ^
  "    $client.Close(); " ^
  "    $ready = $true; " ^
  "    break; " ^
  "  } catch { " ^
  "    Start-Sleep -Milliseconds 500; " ^
  "  } finally { " ^
  "    if ($client) { $client.Close() | Out-Null } " ^
  "  } " ^
  "} " ^
  "if ($ready) { Start-Process ('http://127.0.0.1:' + $port); exit 0 } else { exit 1 }"
if errorlevel 1 (
    echo.
    echo Streamlit dashboard did not become ready on port %PORT%.
)
goto :eof

:finish
if not "%EXIT_CODE%"=="0" (
    echo.
    echo Failed to run. Exit code: %EXIT_CODE%
    echo Check whether project dependencies and LLM configuration are available.
    pause
)

endlocal
