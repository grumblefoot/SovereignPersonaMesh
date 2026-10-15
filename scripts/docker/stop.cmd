@echo off
rem Resolve the compose file relative to this script, so it works from any
rem directory and by double-click.
set "COMPOSE_FILE=%~dp0..\..\config\docker-compose.yml"
rem Brief FX-C: pass the repo's .env to compose when it exists, so ${POSTGRES_*}
rem interpolation and the env_file inside the compose file agree.
set "ENV_FILE_ARGS="
if exist "%~dp0..\..\.env" set ENV_FILE_ARGS=--env-file "%~dp0..\..\.env"

echo Stopping SPM, then its database...
docker compose %ENV_FILE_ARGS% -f "%COMPOSE_FILE%" stop
if errorlevel 1 (
  echo Failed to stop the SPM bundle. Run status.cmd to see what is running.
  if "%~1"=="" pause
  exit /b 1
)
echo SPM and its database are stopped. Your data is kept.
if "%~1"=="" pause
