@echo off
rem Resolve the compose file relative to this script, so it works from any
rem directory and by double-click.
set "COMPOSE_FILE=%~dp0..\..\config\docker-compose.yml"

echo Starting the SPM bundle (builds the image only if it is missing)...
docker compose -f "%COMPOSE_FILE%" up -d
if errorlevel 1 (
  echo Failed to start the SPM bundle. Run logs.cmd for details.
  if "%~1"=="" pause
  exit /b 1
)
echo SPM is running.
echo   Admin dashboard: http://127.0.0.1:5050/admin
echo   SillyTavern URL: http://127.0.0.1:5050/v1
if "%~1"=="" pause
