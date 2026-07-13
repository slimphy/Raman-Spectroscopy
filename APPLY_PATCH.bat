@echo off
setlocal
set "REPO=%~1"
if "%REPO%"=="" (
  echo Usage: drag the Raman-Spectroscopy folder onto this BAT file
  echo        or run APPLY_PATCH.bat "C:\path\to\Raman-Spectroscopy"
  pause
  exit /b 2
)
python "%~dp0apply_stability_patch.py" "%REPO%" --dry-run
if errorlevel 1 (
  echo.
  echo Dry-run failed. No files were changed.
  pause
  exit /b 1
)
echo.
choice /M "Apply the patch now"
if errorlevel 2 exit /b 0
python "%~dp0apply_stability_patch.py" "%REPO%"
pause
