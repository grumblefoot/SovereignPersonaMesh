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

REPO_DIR="$(cd "$(dirname "$0")/../.." && pwd)"
if command -v git >/dev/null 2>&1 && git -C "$REPO_DIR" rev-parse --is-inside-work-tree >/dev/null 2>&1; then
  echo "Updating the code (git pull)..."
  git -C "$REPO_DIR" pull || echo "Warning: git pull failed; continuing with the code you have." >&2
else
  echo "Not a git checkout (or git is not installed) - skipping git pull."
fi

echo "Rebuilding and restarting the SPM bundle..."
$COMPOSE -f "$COMPOSE_FILE" up -d --build
if [ $? -eq 0 ]; then
  echo "SPM updated and running: http://127.0.0.1:5050/admin"
else
  echo "Failed to rebuild/restart the SPM bundle. Run logs.sh for details." >&2
  exit 1
fi
