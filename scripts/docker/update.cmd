@echo off
rem Resolve the compose file relative to this script, so it works from any
rem directory and by double-click.
set "COMPOSE_FILE=%~dp0..\..\config\docker-compose.yml"
rem Brief FX-C: pass the repo's .env to compose when it exists, so ${POSTGRES_*}
rem interpolation and the env_file inside the compose file agree.
set "ENV_FILE_ARGS="
if exist "%~dp0..\..\.env" set ENV_FILE_ARGS=--env-file "%~dp0..\..\.env"

set "REPO_DIR=%~dp0..\.."
where git >nul 2>nul
if errorlevel 1 (
  echo Git is not installed - skipping git pull.
) else (
  git -C "%REPO_DIR%" rev-parse --is-inside-work-tree >nul 2>nul
  if errorlevel 1 (
    echo Not a git checkout - skipping git pull.
  ) else (
    echo Updating the code (git pull)...
    git -C "%REPO_DIR%" pull || echo Warning: git pull failed; continuing with the code you have.
  )
)
echo Rebuilding and restarting the SPM bundle...
docker compose %ENV_FILE_ARGS% -f "%COMPOSE_FILE%" up -d --build
if errorlevel 1 (
  echo Failed to rebuild/restart the SPM bundle. Run logs.cmd for details.
  if "%~1"=="" pause
  exit /b 1
)
echo SPM updated and running: http://127.0.0.1:5050/admin
if "%~1"=="" pause
