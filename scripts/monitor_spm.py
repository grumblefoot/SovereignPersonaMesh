#!/usr/bin/env python3
"""
CLI Terminal Live Stream Monitor for Sovereign Persona Mesh (SPM).

Polls /admin/api/v1/stats and displays real-time performance metrics,
active sessions, spatial gating breakdown, and request log stream.
"""

import sys
import time
import json
import urllib.request
import urllib.error

SPM_ADMIN_URL = "http://localhost:5050/admin/api/v1/stats"


def format_status():
    try:
        req = urllib.request.Request(SPM_ADMIN_URL, headers={"User-Agent": "SPM-Monitor/1.0"})
        with urllib.request.urlopen(req, timeout=2.0) as resp:
            data = json.loads(resp.read().decode("utf-8"))

        uptime = data.get("uptime_seconds", 0)
        requests = data.get("total_requests", 0)
        avg_lat = data.get("avg_latency_ms", 0.0)
        active_sess = data.get("active_sessions_count", 0)
        db_size = data.get("db_size_mb", 0.0)
        gating = data.get("spatial_gating_counts", {})

        print("\033[H\033[J", end="")  # Clear screen
        print("================================================================")
        print("     SOVEREIGN PERSONA MESH (SPM) — REAL-TIME MONITOR       ")
        print("================================================================")
        print(f" Proxy Status          : ONLINE (Port 5050)")
        print(f" Uptime               : {uptime} seconds")
        print(f" Total Requests       : {requests}")
        print(f" Avg Turn Latency     : {avg_lat} ms")
        print(f" Active Sessions      : {active_sess}")
        print(f" Database Size        : {db_size} MB")
        print("----------------------------------------------------------------")
        print(" Spatial Gating Breakdown:")
        print(f"   - Direct   (0 ft)   : {gating.get('direct', 0)}")
        print(f"   - Muffled  (1-30ft) : {gating.get('muffled', 0)}")
        print(f"   - Blackout (>30ft)  : {gating.get('blackout', 0)}")
        print("================================================================")
        print(" Press Ctrl+C to exit monitor.")
    except urllib.error.URLError:
        print("\033[H\033[J", end="")
        print("================================================================")
        print("     SOVEREIGN PERSONA MESH (SPM) — REAL-TIME MONITOR       ")
        print("================================================================")
        print(" Status: WAITING FOR PROXY (http://localhost:5050)...")
        print("================================================================")


def main():
    try:
        while True:
            format_status()
            time.sleep(1.5)
    except KeyboardInterrupt:
        print("\nMonitor stopped.")
        sys.exit(0)


if __name__ == "__main__":
    main()
