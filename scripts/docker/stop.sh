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

echo "Stopping SPM, then its database..."
$COMPOSE -f "$COMPOSE_FILE" stop
if [ $? -eq 0 ]; then
  echo "SPM and its database are stopped. Your data is kept."
else
  echo "Failed to stop the SPM bundle. Run status.sh to see what is running." >&2
  exit 1
fi
