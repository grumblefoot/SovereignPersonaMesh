@echo off
rem Resolve the compose file relative to this script, so it works from any
rem directory and by double-click.
set "COMPOSE_FILE=%~dp0..\..\config\docker-compose.yml"

docker compose -f "%COMPOSE_FILE%" ps
if "%~1"=="" pause
