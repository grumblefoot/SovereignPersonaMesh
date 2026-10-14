#!/usr/bin/env bash
# Resolve the compose file relative to this script, so it works from any directory.
COMPOSE_FILE="$(cd "$(dirname "$0")/../.." && pwd)/config/docker-compose.yml"

# Pick a working compose command: docker compose, else podman compose.
if command -v docker >/dev/null 2>&1 && docker compose version >/dev/null 2>&1; then
  COMPOSE="docker compose"
elif command -v podman >/dev/null 2>&1 && podman compose version >/dev/null 2>&1; then
  COMPOSE="podman compose"
else
  echo "Error: neither 'docker compose' nor 'podman compose' is available." >&2
  echo "Install Docker Desktop (https://www.docker.com/products/docker-desktop/) or Podman and try again." >&2
  exit 1
fi

echo "Starting the SPM bundle (builds the image only if it is missing)..."
$COMPOSE -f "$COMPOSE_FILE" up -d
if [ $? -eq 0 ]; then
  echo "SPM is running."
  echo "  Admin dashboard: http://127.0.0.1:5050/admin"
  echo "  SillyTavern URL: http://127.0.0.1:5050/v1"
else
  echo "Failed to start the SPM bundle. Run logs.sh for details." >&2
  exit 1
fi
