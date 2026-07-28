#!/bin/bash
# Setup script for SPM 3:00 AM Sleep Cycle systemd user service and timer

SYSTEMD_DIR="$HOME/.config/systemd/user"
mkdir -p "$SYSTEMD_DIR"

SERVICE_FILE="$SYSTEMD_DIR/spm-sleep-cycle.service"
TIMER_FILE="$SYSTEMD_DIR/spm-sleep-cycle.timer"
SCRIPT_PATH="$(pwd)/scripts/sleep_cycle.py"

cat << EOF > "$SERVICE_FILE"
[Unit]
Description=Sovereign Persona Mesh (SPM) Daily Sleep Cycle Memory Consolidation
After=network.target

[Service]
Type=oneshot
WorkingDirectory=$(pwd)
ExecStart=/usr/bin/env python3 $SCRIPT_PATH
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
