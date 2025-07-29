@echo off
setlocal

REM Get the directory of this script
set "SCRIPT_DIR=%~dp0"

REM Activate virtual environment if it exists
if exist "%SCRIPT_DIR%venv310\Scripts\activate.bat" (
    call "%SCRIPT_DIR%venv310\Scripts\activate.bat"
)

REM Run the Python script with all arguments
"%SCRIPT_DIR%venv310\Scripts\python.exe" "%SCRIPT_DIR%commands.py" %*

REM Deactivate virtual environment
if exist "%SCRIPT_DIR%venv310\Scripts\deactivate.bat" (
    call "%SCRIPT_DIR%venv310\Scripts\deactivate.bat"
)