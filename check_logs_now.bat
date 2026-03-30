@echo off
setlocal EnableExtensions EnableDelayedExpansion

REM ============================================================================
REM check_logs_now.bat
REM
REM Zweck:
REM   Schneller Gesundheitscheck fuer lokale Laptop-Logs + Sync-Task.
REM
REM Verwendung:
REM   check_logs_now.bat
REM   check_logs_now.bat --run-sync
REM   check_logs_now.bat --no-pause
REM   check_logs_now.bat --run-sync --no-pause
REM   check_logs_now.bat "C:\Users\sebas\mt5_trading\logs"
REM ============================================================================

set "BASE_DIR=%~dp0"
if "%BASE_DIR:~-1%"=="\" set "BASE_DIR=%BASE_DIR:~0,-1%"

set "RUN_SYNC=0"
set "NO_PAUSE=0"
set "HAS_ERROR=0"
set "HAS_WARN=0"
set "ERROR_COUNT=0"
set "WARN_COUNT=0"
set "LOCAL_LOG_DIR="
set "RUNTIME_STALE=0"
set "DETAIL_FILE=%TEMP%\mt5_check_details_%RANDOM%_%RANDOM%.txt"
type nul > "%DETAIL_FILE%"

:parse_args
if "%~1"=="" goto :args_done

if /I "%~1"=="--run-sync" (
    set "RUN_SYNC=1"
    shift
    goto :parse_args
)

if /I "%~1"=="--no-pause" (
    set "NO_PAUSE=1"
    shift
    goto :parse_args
)

if "%LOCAL_LOG_DIR%"=="" (
    set "LOCAL_LOG_DIR=%~1"
) else (
    echo [WARNUNG] Unbekanntes Argument: %~1
    call :add_warn "Unbekanntes Argument ignoriert: %~1"
)
shift
goto :parse_args

:args_done

if "%LOCAL_LOG_DIR%"=="" (
    set "LOCAL_LOG_DIR=%USERPROFILE%\mt5_trading\logs"
)

if not exist "%LOCAL_LOG_DIR%\live_trader.log" (
    if exist "%BASE_DIR%\logs\live_trader.log" (
        set "LOCAL_LOG_DIR=%BASE_DIR%\logs"
    )
)

echo ========================================================================
echo MT5 LOG CHECK (Laptop)
echo ========================================================================
echo Zeit: %DATE% %TIME%
echo Log-Ordner: %LOCAL_LOG_DIR%
echo.

if not exist "%LOCAL_LOG_DIR%" (
    echo [FEHLER] Log-Ordner nicht gefunden.
    echo Erwartet z. B.: C:\Users\sebas\mt5_trading\logs
    call :add_error "Log-Ordner fehlt: %LOCAL_LOG_DIR%"
    goto :task_check
)

call :check_file "%LOCAL_LOG_DIR%\live_trader.log" "Runtime-Log"
call :check_file "%LOCAL_LOG_DIR%\USDCAD_signals.csv" "USDCAD Signals"
call :check_file "%LOCAL_LOG_DIR%\USDJPY_signals.csv" "USDJPY Signals"
call :check_file "%LOCAL_LOG_DIR%\live_log_watchdog_latest.json" "Watchdog JSON"
call :check_file "%LOCAL_LOG_DIR%\live_log_watchdog_latest.csv" "Watchdog CSV"

echo.
:task_check
set "TASK_NAME=\MT5_Sync_Live_Logs_To_Linux"
echo ========================================================================
echo TASK-SCHEDULER CHECK
echo ========================================================================

echo [INFO] Pruefe Task: %TASK_NAME%
schtasks /Query /TN "%TASK_NAME%" /FO LIST /V > "%TEMP%\mt5_task_status.txt" 2>nul
if errorlevel 1 (
    echo [WARNUNG] Task nicht gefunden: %TASK_NAME%
    echo          Bitte auf dem Laptop registrieren:
    echo          powershell -ExecutionPolicy Bypass -File scripts\windows_register_live_log_sync_task.ps1 -RunNow
    call :add_error "Task fehlt: %TASK_NAME%"
) else (
    findstr /I /C:"Status:" /C:"Last Run Time:" /C:"Last Result:" /C:"Task To Run:" "%TEMP%\mt5_task_status.txt"
)

del "%TEMP%\mt5_task_status.txt" >nul 2>&1

if "%RUN_SYNC%"=="1" (
    echo.
    echo [INFO] Starte Sync-Task sofort...
    schtasks /Run /TN "%TASK_NAME%"
    if errorlevel 1 (
        echo [WARNUNG] Task konnte nicht gestartet werden.
        call :add_warn "Task-Start fehlgeschlagen: %TASK_NAME%"
    ) else (
        echo [OK] Sync-Task wurde gestartet.
        call :add_info "Task manuell gestartet: %TASK_NAME%"
    )
)

call :runtime_diagnose

echo.
echo ========================================================================
echo HINWEIS
echo ========================================================================
echo Wenn die Dateien hier frisch sind, aber auf dem Server alt erscheinen,
echo ist meist der Sync-Task das Problem (nicht der Trader).

echo.
echo ========================================================================
echo DEBUG-DETAILS
echo ========================================================================
set "HAS_DEBUG_LINES=0"
for /f "usebackq tokens=1* delims=|" %%A in ("%DETAIL_FILE%") do (
    set "HAS_DEBUG_LINES=1"
    if /I "%%A"=="E" echo [E] %%B
    if /I "%%A"=="W" echo [W] %%B
    if /I "%%A"=="I" echo [I] %%B
)
if "%HAS_DEBUG_LINES%"=="0" (
    echo [OK] Keine Debug-Details vorhanden.
)

echo.
echo ========================================================================
echo ERGEBNIS
echo ========================================================================
echo Fehler: %ERROR_COUNT% ^| Warnungen: %WARN_COUNT%
if "%HAS_ERROR%"=="1" (
    echo [FEHLER] Mindestens ein kritischer Check ist fehlgeschlagen.
) else if "%HAS_WARN%"=="1" (
    echo [WARNUNG] Keine kritischen Fehler, aber es gibt Warnungen.
) else (
    echo [OK] Kritische Checks ohne Fehler abgeschlossen.
)

if "%NO_PAUSE%"=="0" (
    echo.
    echo [INFO] Fenster bleibt offen. Mit einer Taste beenden.
    pause >nul
)

if "%HAS_ERROR%"=="1" (
    if exist "%DETAIL_FILE%" del "%DETAIL_FILE%" >nul 2>&1
    exit /b 1
)

if "%HAS_WARN%"=="1" (
    if exist "%DETAIL_FILE%" del "%DETAIL_FILE%" >nul 2>&1
    exit /b 2
)

if exist "%DETAIL_FILE%" del "%DETAIL_FILE%" >nul 2>&1
exit /b 0

goto :eof

:check_file
set "FILE_PATH=%~1"
set "LABEL=%~2"

if not exist "%FILE_PATH%" (
    echo [FEHLER] %LABEL% fehlt: %FILE_PATH%
    call :add_error "%LABEL% fehlt: %FILE_PATH%"
    goto :eof
)

for %%I in ("%FILE_PATH%") do (
    set "FILE_SIZE=%%~zI"
    set "FILE_TIME=%%~tI"
)

for /f %%M in ('powershell -NoProfile -Command "$p='%FILE_PATH%'; if(Test-Path $p){[math]::Round(((Get-Date).ToUniversalTime() - (Get-Item $p).LastWriteTimeUtc).TotalMinutes,1)} else {-1}"') do set "AGE_MIN=%%M"

echo [OK] %LABEL%
echo      Datei: %FILE_PATH%
echo      Geaendert: !FILE_TIME! ^| Groesse: !FILE_SIZE! Bytes ^| Alter: !AGE_MIN! Min

for /f "tokens=1 delims=." %%A in ("!AGE_MIN!") do set "AGE_INT=%%A"
if not "!AGE_INT!"=="" (
    if !AGE_INT! GEQ 30 (
        echo      [WARNUNG] Datei ist aelter als 30 Minuten.
        call :add_warn "%LABEL% ist stale (!AGE_MIN! Min)"
        if /I "%LABEL%"=="Runtime-Log" set "RUNTIME_STALE=1"
    )
)

goto :eof

:runtime_diagnose
if "%RUNTIME_STALE%"=="1" (
    if exist "%LOCAL_LOG_DIR%\live_trader.log" (
        findstr /I /C:"Trader gestoppt" /C:"Ctrl+C" "%LOCAL_LOG_DIR%\live_trader.log" >nul 2>&1
        if not errorlevel 1 (
            call :add_info "Runtime-Log zeigt Stop-Muster (Trader gestoppt/Ctrl+C)."
        ) else (
            call :add_info "Runtime-Log stale ohne klares Stop-Muster; Prozessstatus pruefen."
        )
    )
)
goto :eof

:add_error
set "HAS_ERROR=1"
set /a ERROR_COUNT+=1
>>"%DETAIL_FILE%" echo E^|%~1
goto :eof

:add_warn
set "HAS_WARN=1"
set /a WARN_COUNT+=1
>>"%DETAIL_FILE%" echo W^|%~1
goto :eof

:add_info
>>"%DETAIL_FILE%" echo I^|%~1
goto :eof
