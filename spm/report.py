"""Bug reports (brief R2): the report file, the zip bundle, the GitHub issue URL.

Builds the Traditional report described in docs/plans/bug_report_v04.md: safe
diagnostic checks (spm/diagnostics.py) plus a redacted log excerpt, rendered as
Markdown and zipped under the git-ignored ``reports/`` directory, plus a
prefilled "New issue" URL. Nothing leaves the machine automatically: the user
sees the bundle path and the URL, and the private (``--full-logs``) report gets
no URL at all.

Safety: nothing here starts services or talks to the live database beyond what
the read-only checks do; Enhanced analysis (the R3 agent) is not built yet, so
``--enhanced`` only prints a notice.
"""

from __future__ import annotations

import asyncio
import json
import sys
import urllib.parse
import webbrowser
import zipfile
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Optional

from .diagnostics import Check, name_table, run_checks
from .redact import Redactor
from .run import child_env, default_repo_root

__all__ = ["ISSUE_URL", "MAX_ENCODED_URL", "Report", "build_report",
           "issue_url", "write_bundle", "cmd_report"]

ISSUE_URL = "https://github.com/grumblefoot/SovereignPersonaMesh/issues/new"
MAX_ENCODED_URL = 7500

# Log excerpt sizes (brief R2): the public report carries the filtered last 200
# lines; --full-logs keeps the last 2,000 lines with message content (private).
PUBLIC_LOG_LIMIT = 200
FULL_LOG_LIMIT = 2000

# Read at most this many trailing bytes per log file (a tail window, so a huge
# log never has to be read whole).
_MAX_LOG_WINDOW_BYTES = 1 << 20

PRIVATE_BANNER = ("PRIVATE: contains chat text. Send it only to the developers, "
                  "never post it publicly.")


@dataclass
class Report:
    markdown: str
    checks: List[Check]
    path: Optional[Path]
    issue_url: Optional[str]
    private: bool


# ── Markdown ──────────────────────────────────────────────────────────────────

def _utc_stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")


def _version(root: Path) -> str:
    try:
        text = (root / "VERSION").read_text(encoding="utf-8", errors="replace")
        return text.strip() or "unknown"
    except OSError:
        return "unknown"


def _status_table(checks: List[Check]) -> str:
    rows = ["| check | status | summary |", "|---|---|---|"]
    for c in checks:
        summary = " ".join(c.summary.split()).replace("|", "\\|")
        rows.append(f"| {c.name} | {c.status} | {summary} |")
    return "\n".join(rows)


def _report_markdown(description: str, checks: List[Check], *, version: str,
                     private: bool, enhanced=None) -> str:
    lines = ["# SPM bug report", ""]
    lines.append(f"- Time (UTC): {_utc_stamp()}")
    lines.append(f"- SPM version: {version}")
    if private:
        lines.append("- Private: yes (full logs included; send only to the developers)")
    lines += ["", "## Description", "", description.strip() or "(none given)"]
    lines += ["", "## Checks", "", _status_table(checks)]
    for c in checks:
        lines += ["", f"## Check: {c.name}", "", "```json",
                  json.dumps(c.details, indent=2, ensure_ascii=False, default=str),
                  "```"]
    if enhanced:  # placeholder until R3 builds the Enhanced agent
        text = enhanced if isinstance(enhanced, str) else json.dumps(
            enhanced, indent=2, ensure_ascii=False, default=str)
        lines += ["", "## Enhanced analysis", "", text]
    lines.append("")
    return "\n".join(lines)


# ── log excerpt ───────────────────────────────────────────────────────────────

def _tail_file(path: Path, limit: int) -> List[str]:
    try:
        size = path.stat().st_size
        with path.open("r", encoding="utf-8", errors="replace") as fh:
            if size > _MAX_LOG_WINDOW_BYTES:
                fh.seek(size - _MAX_LOG_WINDOW_BYTES)
                fh.readline()  # drop the partial first line
            return fh.read().splitlines()[-limit:]
    except OSError:
        return []


def _read_log_lines(root: Path, limit: int) -> List[str]:
    """The last `limit` lines across logs/*.out and logs/*.log (raw)."""
    files = sorted(list(root.glob("logs/*.out")) + list(root.glob("logs/*.log")))
    lines: List[str] = []
    for path in files:
        lines.extend(_tail_file(path, limit))
    return lines[-limit:]


# ── bundle ────────────────────────────────────────────────────────────────────

def write_bundle(report_md: str, checks: List[Check], log_lines: List[str], *,
                 private: bool, out_dir: Path) -> Path:
    """Zip report.md + checks.json + the log excerpt into reports/ and return
    the bundle path. Private bundles are named spm-report-PRIVATE-... and their
    report.md starts with the PRIVATE banner."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    name = f"spm-report-{'PRIVATE-' if private else ''}{stamp}.zip"
    path = out_dir / name

    markdown = f"{PRIVATE_BANNER}\n\n{report_md}" if private else report_md
    payload = [{"name": c.name, "status": c.status, "summary": c.summary,
                "details": c.details} for c in checks]
    log_name = "logs-redacted.txt" if not private else "logs-full-PRIVATE.txt"
    with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("report.md", markdown)
        zf.writestr("checks.json", json.dumps(payload, indent=2,
                                              ensure_ascii=False, default=str))
        zf.writestr(log_name, "\n".join(log_lines) + ("\n" if log_lines else ""))
    return path


# ── GitHub issue URL ──────────────────────────────────────────────────────────

def _encoded(title: str, body: str) -> str:
    query = urllib.parse.urlencode({"title": title, "body": body})
    return f"{ISSUE_URL}?{query}"


def issue_url(title: str, body: str) -> str:
    """Prefilled 'New issue' URL, with the ENCODED length capped at
    MAX_ENCODED_URL. The body is truncated on a code-point boundary (Python str
    slices never split a code point) until the whole URL fits: a binary search
    keeps the longest prefix that fits, even when characters encode to
    multi-byte escapes (e.g. Chinese)."""
    text = body.rstrip()
    if len(_encoded(title, text)) <= MAX_ENCODED_URL:
        return _encoded(title, text)
    lo, hi = 0, len(text)
    while lo < hi:
        mid = (lo + hi + 1) // 2
        if len(_encoded(title, text[:mid].rstrip())) <= MAX_ENCODED_URL:
            lo = mid
        else:
            hi = mid - 1
    return _encoded(title, text[:lo].rstrip())


# ── build ─────────────────────────────────────────────────────────────────────

async def build_report(env: dict, *, description: str = "", system_info: bool = False,
                       full_logs: bool = False, enhanced=None,
                       repo_root=None) -> Report:
    """Run the checks, redact, render the Markdown, write the zip bundle and
    (public reports only) build the issue URL. Never raises for failed checks:
    a report about a broken SPM is the point."""
    root = Path(repo_root) if repo_root is not None else default_repo_root()
    private = bool(full_logs)

    names = await name_table(env)
    redactor = Redactor(names=names)
    checks = await run_checks(env, system_info=system_info, repo_root=root,
                              redactor=redactor)

    if private:
        # Private keeps prose; secrets, paths, sessions and IPs are still removed.
        desc_shown = Redactor(names=names, keep_content=True).text(description)
    else:
        desc_shown = redactor.text(description)
    markdown = _report_markdown(desc_shown, checks, version=_version(root),
                                private=private, enhanced=enhanced)

    if private:
        raw = _read_log_lines(root, FULL_LOG_LIMIT)
        log_lines = Redactor(names=names, keep_content=True).lines(raw)
    else:
        raw = _read_log_lines(root, max(FULL_LOG_LIMIT, 4 * PUBLIC_LOG_LIMIT))
        log_lines = redactor.lines(raw)[-PUBLIC_LOG_LIMIT:]

    path = write_bundle(markdown, checks, log_lines, private=private,
                        out_dir=root / "reports")

    url = None
    if not private:
        first = (desc_shown.strip().splitlines() or [""])[0]
        title = "[bug] " + (first[:80] if first else "(no description)")
        body = ((desc_shown.strip() + "\n\n") if desc_shown.strip() else "") \
            + _status_table(checks) \
            + f"\n\nPlease attach {path.name} to this issue."
        url = issue_url(title, body)

    return Report(markdown=markdown, checks=checks, path=path, issue_url=url,
                  private=private)


# ── command line ──────────────────────────────────────────────────────────────

def _stdin_tty() -> bool:
    """Is stdin an interactive terminal? A seam so tests never block on input."""
    try:
        return sys.stdin.isatty()
    except (AttributeError, ValueError):
        return False


def cmd_report(repo_root: Path, *, description: str = "", system_info: bool = False,
               full_logs: bool = False, yes: bool = False, enhanced: bool = False,
               input_fn=None) -> int:
    """`python -m spm report`: print the check table, the bundle path and the
    issue URL. Always exits 0 (even when checks fail)."""
    if enhanced:
        print("Enhanced analysis arrives in the next build")

    if not description and (input_fn is not None or _stdin_tty()):
        prompt = "What went wrong? (one line, Enter to skip): "
        try:
            description = (input_fn(prompt) if input_fn else input(prompt)).strip()
        except (EOFError, KeyboardInterrupt):
            print()
            description = ""

    env = child_env(repo_root)
    report = asyncio.run(build_report(env, description=description,
                                      system_info=system_info,
                                      full_logs=full_logs, repo_root=repo_root))

    print(_status_table(report.checks))
    print()
    print(f"Report bundle: {report.path}")
    if report.issue_url:
        print(f"File an issue: {report.issue_url}")
        if not yes:
            try:
                webbrowser.open(report.issue_url)
            except Exception:
                pass  # headless: the printed URL is enough
    else:
        print("This report is private: no issue link. Send the file only to the "
              "developers, never post it publicly.")
    return 0
