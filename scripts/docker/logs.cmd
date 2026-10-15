@echo off
rem Resolve the compose file relative to this script, so it works from any
rem directory and by double-click.
set "COMPOSE_FILE=%~dp0..\..\config\docker-compose.yml"
rem Brief FX-C: pass the repo's .env to compose when it exists, so ${POSTGRES_*}
rem interpolation and the env_file inside the compose file agree.
set "ENV_FILE_ARGS="
if exist "%~dp0..\..\.env" set ENV_FILE_ARGS=--env-file "%~dp0..\..\.env"

echo Following the last 100 lines of SPM logs (press Ctrl+C to stop watching)...
docker compose %ENV_FILE_ARGS% -f "%COMPOSE_FILE%" logs -f --tail 100 spm
if "%~1"=="" pause
