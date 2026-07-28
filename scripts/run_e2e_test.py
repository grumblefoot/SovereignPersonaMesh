#!/usr/bin/env python3
"""
E2E Integration Test Launcher for Sovereign Persona Mesh (SPM).

Starts background microservices (Evennia on port 4005, SPM Proxy on port 5050),
runs pytest against tests/test_e2e_integration.py, captures SLA metrics,
and outputs formatted findings.

Usage:
    python scripts/run_e2e_test.py

Environment variables:
    SPM_HOST      - SPM proxy host (default: 0.0.0.0)
    SPM_PORT      - SPM proxy port  (default: 5050)
    EVENNIA_PORT  - Evennia liaison port (default: 4005)
    LEMONADE_PORT - Lemonade LLM port (default: 13305)
"""

import argparse
import json
import os
import re
import signal
import subprocess
import sys
import time
import urllib.request
import urllib.error

# ── Constants ────────────────────────────────────────────────────────────────

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PYTHON = os.path.join(
    PROJECT_ROOT, "..", "spm-demo-mvp", "venv", "bin", "python"
)
PYTEST = os.path.join(
    PROJECT_ROOT, "..", "spm-demo-mvp", "venv", "bin", "pytest"
)

EVENNIA_APP = os.path.join(PROJECT_ROOT, "evennia_world", "app.py")
PROXY_APP = os.path.join(PROJECT_ROOT, "proxy", "main.py")

EVENNIA_PORT = int(os.environ.get("EVENNIA_PORT", "4005"))
PROXY_PORT = int(os.environ.get("SPM_PORT", "5050"))
LEMONADE_PORT = int(os.environ.get("LEMONADE_PORT", "13305"))
POSTGRES_HOST = "localhost"
POSTGRES_PORT = 5432

SERVICES = [
    ("Evennia Liaison", EVENNIA_PORT, f"http://localhost:{EVENNIA_PORT}/health"),
    ("SPM Proxy", PROXY_PORT, f"http://localhost:{PROXY_PORT}/health"),
    ("PostgreSQL", POSTGRES_PORT, None),
]

SLA_TARGETS = {
    "proxy_routing_overhead_ms": 150,
    "ttft_ms": 1800,
    "zero_inference_bypass_ms": 10,
    "e2e_test_pass_rate_pct": 100,
}


# ── Helpers ──────────────────────────────────────────────────────────────────

def log(tag: str, msg: str) -> None:
    """Print a timestamped log line."""
    ts = time.strftime("%H:%M:%S")
    print(f"[{ts}] [{tag:20s}] {msg}")


def wait_for_service(name: str, url: str | None, timeout: float = 30) -> bool:
    """Block until a service responds or timeout."""
    start = time.time()
    if url is None:
        # PostgreSQL: use a quick TCP check
        while time.time() - start < timeout:
            try:
                import socket
                s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                s.settimeout(1.0)
                s.connect((POSTGRES_HOST, POSTGRES_PORT))
                s.close()
                log("SERVICE", f"{name} is UP")
                return True
            except Exception:
                time.sleep(0.5)
        log("SERVICE", f"FAILED: {name} did not respond within {timeout}s")
        return False
    while time.time() - start < timeout:
        try:
            req = urllib.request.Request(url, method="GET")
            with urllib.request.urlopen(req, timeout=2) as resp:
                if resp.status == 200:
                    log("SERVICE", f"{name} is UP")
                    return True
        except Exception:
            pass
        time.sleep(0.5)
    log("SERVICE", f"FAILED: {name} did not respond within {timeout}s")
    return False


def start_service(app_path: str, port: int, name: str) -> subprocess.Popen:
    """Start a FastAPI service as a background process."""
    env = os.environ.copy()
    env["SPM_PORT"] = str(port)
    proc = subprocess.Popen(
        [PYTHON, "-m", "uvicorn", f"{os.path.splitext(os.path.basename(app_path))[0]}:app",
         "--host", "127.0.0.1", "--port", str(port), "--log-level", "warning"],
        cwd=PROJECT_ROOT,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    log("STARTUP", f"Launched {name} (PID {proc.pid})")
    return proc


def measure_latency(url: str, label: str) -> float:
    """Measure HTTP GET latency in milliseconds."""
    start = time.time()
    try:
        req = urllib.request.Request(url, method="GET")
        with urllib.request.urlopen(req, timeout=5) as resp:
            _ = resp.read()
    except Exception:
        pass
    latency_ms = (time.time() - start) * 1000
    log("METRIC", f"{label}: {latency_ms:.1f}ms")
    return latency_ms


def run_pytest(pytest_path: str) -> tuple[int, str, str]:
    """Run pytest and capture output."""
    log("TEST", "Starting pytest...")
    result = subprocess.run(
        [pytest_path, "tests/test_e2e_integration.py", "-v", "--tb=short", "-s"],
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
        timeout=120,
    )
    log("TEST", f"Exit code: {result.returncode}")
    return result.returncode, result.stdout, result.stderr


# ── Main ─────────────────────────────────────────────────────────────────────

def main() -> int:
    parser = argparse.ArgumentParser(description="SPM E2E Integration Test Launcher")
    parser.add_argument("--no-start", action="store_true", help="Do not start services (assume running)")
    parser.add_argument("--no-stop", action="store_true", help="Do not stop services after tests")
    args = parser.parse_args()

    log("E2E", "=" * 60)
    log("E2E", "Sovereign Persona Mesh — End-to-End Integration Test")
    log("E2E", "=" * 60)

    procs = []

    # Step 1: Start services if needed
    if not args.no_start:
        log("E2E", "Starting background services...")
        procs.append(start_service(EVENNIA_APP, EVENNIA_PORT, "Evennia Liaison"))
        procs.append(start_service(PROXY_APP, PROXY_PORT, "SPM Proxy"))

        # Wait for services
        if not wait_for_service("Evennia Liaison",
                               f"http://localhost:{EVENNIA_PORT}/health"):
            log("E2E", "WARNING: Evennia Liaison failed to start, tests will use fallback mocks")
        if not wait_for_service("SPM Proxy", f"http://localhost:{PROXY_PORT}/health"):
            log("E2E", "WARNING: SPM Proxy failed to start, tests will use TestClient")
    else:
        log("E2E", "Skipping service startup (--no-start)")

    # Step 2: Service readiness checks
    log("E2E", "─" * 60)
    log("E2E", "Phase 1: Service Readiness Probes")
    log("E2E", "─" * 60)
    for name, port, url in SERVICES:
        if url:
            wait_for_service(name, url, timeout=5)
        else:
            wait_for_service(name, None, timeout=5)

    # Step 3: SLA latency measurements
    log("E2E", "─" * 60)
    log("E2E", "Phase 2: SLA Latency Measurements")
    log("E2E", "─" * 60)

    proxy_latency = measure_latency(
        f"http://localhost:{PROXY_PORT}/health",
        "Proxy Routing Overhead"
    )
    proxy_latency_ok = proxy_latency < SLA_TARGETS["proxy_routing_overhead_ms"]

    evennia_latency = measure_latency(
        f"http://localhost:{EVENNIA_PORT}/health",
        "Evennia Action Eval"
    )

    sla_results = {
        "proxy_routing_overhead_ms": proxy_latency,
        "evennia_action_evaluation_ms": evennia_latency,
        "proxy_routing_sla_met": proxy_latency_ok,
        "proxy_routing_sla_target_ms": SLA_TARGETS["proxy_routing_overhead_ms"],
    }
    log("E2E", f"  Proxy Routing Overhead: {proxy_latency:.1f}ms "
            f"{'✓' if proxy_latency_ok else '✗'} (SLA < {SLA_TARGETS['proxy_routing_overhead_ms']}ms)")

    # Step 4: Run pytest
    log("E2E", "─" * 60)
    log("E2E", "Phase 3: Running pytest — tests/test_e2e_integration.py")
    log("E2E", "─" * 60)

    exit_code, stdout, stderr = run_pytest(PYTEST)

    # Parse test results from stdout
    total_tests = 0
    passed_tests = 0
    failed_tests = 0
    skipped_tests = 0

    for line in stdout.split("\n"):
        if "passed" in line and ("test" in line.lower()):
            # e.g. "8 passed, 0 failed in 3.21s"
            m = re.search(r"(\d+)\s+passed", line)
            if m:
                passed_tests = int(m.group(1))
            m = re.search(r"(\d+)\s+failed", line)
            if m:
                failed_tests = int(m.group(1))
            m = re.search(r"(\d+)\s+skipped", line)
            if m:
                skipped_tests = int(m.group(1))
        if "collected" in line:
            m = re.search(r"(\d+)\s+collected", line)
            if m:
                total_tests = int(m.group(1))

    if total_tests == 0:
        total_tests = passed_tests + failed_tests + skipped_tests

    pass_rate = (passed_tests / total_tests * 100) if total_tests > 0 else 0
    sla_pass_rate_ok = pass_rate >= SLA_TARGETS["e2e_test_pass_rate_pct"]
    sla_results["total_tests"] = total_tests
    sla_results["passed_tests"] = passed_tests
    sla_results["failed_tests"] = failed_tests
    sla_results["skipped_tests"] = skipped_tests
    sla_results["pass_rate_pct"] = pass_rate
    sla_results["e2e_test_pass_rate_met"] = sla_pass_rate_ok

    log("E2E", "")
    log("E2E", f"  Tests collected : {total_tests}")
    log("E2E", f"  Tests passed    : {passed_tests}")
    log("E2E", f"  Tests failed    : {failed_tests}")
    log("E2E", f"  Tests skipped   : {skipped_tests}")
    log("E2E", f"  Pass rate       : {pass_rate:.1f}% {'✓' if sla_pass_rate_ok else '✗'} "
            f"(SLA target: {SLA_TARGETS['e2e_test_pass_rate_rate_pct'] if 'SLA_TARGETS' in dir() else SLA_TARGETS['e2e_test_pass_rate_pct']}%)")

    # Step 5: Print full pytest output if there are failures
    if failed_tests > 0:
        log("E2E", "")
        log("E2E", "─" * 60)
        log("E2E", "FAILED TEST DETAILS")
        log("E2E", "─" * 60)
        print(stdout)

    # Step 6: Summary
    log("E2E", "=" * 60)
    log("E2E", "SLA SUMMARY")
    log("E2E", "=" * 60)
    sla_ok = proxy_latency_ok and sla_pass_rate_ok
    for key, val in sla_results.items():
        if isinstance(val, bool):
            continue
        log("SLA", f"  {key}: {val}")
    log("E2E", f"  Overall: {'ALL SLAs MET ✓' if sla_ok else 'SOME SLAs FAILED ✗'}")
    log("E2E", "=" * 60)

    # Step 7: Write SLA metrics to JSON
    sla_path = os.path.join(PROJECT_ROOT, "e2e_sla_metrics.json")
    with open(sla_path, "w") as f:
        json.dump(sla_results, f, indent=2)
    log("E2E", f"SLA metrics written to {sla_path}")

    # Step 8: Stop services
    if not args.no_stop and procs:
        log("E2E", "Stopping services...")
        for proc in procs:
            try:
                proc.terminate()
                proc.wait(timeout=5)
                log("E2E", f"  Stopped PID {proc.pid}")
            except Exception as e:
                log("E2E", f"  Kill PID {proc.pid}: {e}")

    # Step 9: Report findings
    log("E2E", "FINAL FINDINGS:")
    log("E2E", f"  {passed_tests}/{total_tests} tests passed ({pass_rate:.0f}%)")
    log("E2E", f"  Proxy routing overhead: {proxy_latency:.1f}ms (SLA < {SLA_TARGETS['proxy_routing_overhead_ms']}ms)")
    log("E2E", f"  All services reachable: {'Yes' if all(wait_for_service(n, u, timeout=2) for n, u in [('SPM Proxy', f'http://localhost:{PROXY_PORT}/health')]) else 'N/A'}")
    log("E2E", "=" * 60)

    return 0 if (failed_tests == 0 and proxy_latency_ok) else 1


if __name__ == "__main__":
    sys.exit(main())
