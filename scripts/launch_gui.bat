@echo off
setlocal
REM ============================================================
REM  Blenny GUI launcher
REM
REM  Double-click this file (or a desktop shortcut to it) to
REM  start the Blenny plate reader interface in your browser.
REM  Close this window to stop the server.
REM ============================================================

REM Run from the repository root so pipeline YAMLs and uploads
REM resolve correctly regardless of where the shortcut lives.
cd /d "%~dp0.."

set "LAUNCHER="

REM 1) Prefer an installed Python that can import blenny.
where python >nul 2>nul
if not errorlevel 1 (
    python -c "import blenny" >nul 2>nul
    if not errorlevel 1 set "LAUNCHER=python -m blenny"
)

REM 2) Fall back to a Poetry environment.
if not defined LAUNCHER (
    where poetry >nul 2>nul
    if not errorlevel 1 (
        poetry run python -c "import blenny" >nul 2>nul
        if not errorlevel 1 set "LAUNCHER=poetry run blenny"
    )
)

if not defined LAUNCHER (
    echo Blenny is not installed in any detected Python environment.
    echo Install it first, e.g.:  poetry install   or   pip install -e .
    echo.
    pause
    exit /b 1
)

echo Starting Blenny GUI... (close this window to stop the server)
%LAUNCHER% gui
