@echo off
rem Resolve the compose file relative to this script, so it works from any
rem directory and by double-click.
set "COMPOSE_FILE=%~dp0..\..\config\docker-compose.yml"

echo Following the last 100 lines of SPM logs (press Ctrl+C to stop watching)...
docker compose -f "%COMPOSE_FILE%" logs -f --tail 100 spm
if "%~1"=="" pause
