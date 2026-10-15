@echo off
rem Resolve the compose file relative to this script, so it works from any
rem directory and by double-click.
set "COMPOSE_FILE=%~dp0..\..\config\docker-compose.yml"
rem Brief FX-C: pass the repo's .env to compose when it exists, so ${POSTGRES_*}
rem interpolation and the env_file inside the compose file agree.
set "ENV_FILE_ARGS="
if exist "%~dp0..\..\.env" set ENV_FILE_ARGS=--env-file "%~dp0..\..\.env"

echo Starting the SPM bundle (builds the image only if it is missing)...
docker compose %ENV_FILE_ARGS% -f "%COMPOSE_FILE%" up -d
if errorlevel 1 (
  echo Failed to start the SPM bundle. Run logs.cmd for details.
  if "%~1"=="" pause
  exit /b 1
)
echo SPM is running.
echo   Admin dashboard: http://127.0.0.1:5050/admin
echo   SillyTavern URL: http://127.0.0.1:5050/v1
if "%~1"=="" pause
