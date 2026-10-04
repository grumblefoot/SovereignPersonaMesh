#!/bin/bash
# Setup script for SPM 3:00 AM Sleep Cycle systemd user service and timer

SYSTEMD_DIR="$HOME/.config/systemd/user"
mkdir -p "$SYSTEMD_DIR"

SERVICE_FILE="$SYSTEMD_DIR/spm-sleep-cycle.service"
TIMER_FILE="$SYSTEMD_DIR/spm-sleep-cycle.timer"
# Run from the repo root. Uses the project's own venv (system python3 has no asyncpg) and runs the
# script as a module so the `core` and `scripts` packages import.
VENV_PYTHON="$(pwd)/.venv/bin/python"

cat << EOF > "$SERVICE_FILE"
[Unit]
Description=Sovereign Persona Mesh (SPM) Daily Sleep Cycle Memory Consolidation
After=network.target

[Service]
Type=oneshot
WorkingDirectory=$(pwd)
ExecStart=$VENV_PYTHON -m scripts.sleep_cycle
StandardOutput=journal
StandardError=journal

[Install]
WantedBy=default.target
EOF

cat << EOF > "$TIMER_FILE"
[Unit]
Description=Runs SPM Sleep Cycle daily at 3:00 AM

[Timer]
OnCalendar=*-*-* 03:00:00
Persistent=true

[Install]
WantedBy=timers.target
EOF

systemctl --user daemon-reload
systemctl --user enable --now spm-sleep-cycle.timer

echo "SPM Sleep Cycle systemd timer successfully installed and enabled for 3:00 AM daily."
