@echo off
rem Resolve the compose file relative to this script, so it works from any
rem directory and by double-click.
set "COMPOSE_FILE=%~dp0..\..\config\docker-compose.yml"

echo Stopping SPM, then its database...
docker compose -f "%COMPOSE_FILE%" stop
if errorlevel 1 (
  echo Failed to stop the SPM bundle. Run status.cmd to see what is running.
  if "%~1"=="" pause
  exit /b 1
)
echo SPM and its database are stopped. Your data is kept.
if "%~1"=="" pause
