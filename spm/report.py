"""Bug reports (brief R2): the report file, the zip bundle, the GitHub issue URL.

Builds the Traditional report (bug-report design): safe
diagnostic checks (spm/diagnostics.py) plus a redacted log excerpt, rendered as
Markdown and zipped under the git-ignored ``reports/`` directory, plus a
prefilled "New issue" URL. Nothing leaves the machine automatically: the user
sees the bundle path and the URL, and the private (``--full-logs``) report gets
no URL at all.

Safety: nothing here starts services or talks to the live database beyond what
the read-only checks do. Enhanced analysis (brief R3) sends the *redacted*
description and check results to the configured model server; with a remote
server the CLI prints a notice and requires consent first (plan: same
``_is_local`` rule as reasoning). Any failure falls back to the Traditional
report plus an "Enhanced analysis unavailable: <reason>" line.
"""

from __future__ import annotations

import asyncio
import json
import re
import sys
import urllib.parse
import webbrowser
import zipfile
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Optional

from .diagnostics import EXTRA_CHECKS, Check, name_table, run_checks
from .redact import Redactor
from .run import child_env, default_repo_root

__all__ = ["ISSUE_URL", "MAX_ENCODED_URL", "Report", "build_report",
           "enhanced_analysis", "issue_url", "write_bundle", "cmd_report"]

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


# ── enhanced analysis (brief R3) ─────────────────────────────────────────────
#
# The model-powered analysis: system prompt = docs/troubleshooting.md
# (truncated by section to fit the budget), user prompt = the redacted
# description + the check results as compact JSON. The model answers with one
# JSON object; it may ask once for extra checks from EXTRA_CHECKS. Any failure
# degrades to {"error": ...} — the Traditional report is still written.

_MODEL_TIMEOUT_S = 90.0        # per-request cap for the analysis call
_MODEL_MAX_TOKENS = 800        # utility-sized output
_MODEL_TEMPERATURE = 0.2
_FALLBACK_WINDOW_TOKENS = 6000
_PROMPT_BUDGET_FRACTION = 0.55
_MAX_NEED_CHECKS = 3

_SEVERITIES = ("low", "medium", "high")

_JSON_INSTRUCTION = (
    "You are the SPM troubleshooting analyst: read the playbook in the system "
    "prompt, the user description and the diagnostic checks, and reply with a "
    "single JSON object and nothing else, with exactly these fields: "
    '"summary" (string), "likely_causes" (list of objects with string fields '
    '"cause", "evidence" and "fix"), "what_to_try" (list of strings), '
    '"severity" (one of "low", "medium", "high"). If running one more check '
    'would genuinely help, also include "need_checks": a list of at most 3 '
    'names from: ' + ", ".join(EXTRA_CHECKS) + "."
)

_RETRY_HINT = "Reply with the JSON object only."

_FENCE_RE = re.compile(r"```(?:json)?\s*(.*?)```", re.DOTALL | re.IGNORECASE)


def _playbook_text() -> str:
    """The troubleshooting playbook (seam: tests replace this)."""
    try:
        return (default_repo_root() / "docs"
                / "troubleshooting.md").read_text(encoding="utf-8",
                                                  errors="replace")
    except OSError:
        return ""


async def _window_tokens(base_url: str, model: str, client) -> Optional[int]:
    """Detected context window for the prompt budget (seam: tests replace
    this). detect_window never raises; unknown returns None."""
    from proxy.backend_client.backend_profile import detect_window

    http = getattr(client, "_window_http", None) or getattr(client, "client", None)
    if http is None:
        import httpx
        http = httpx.AsyncClient(timeout=1.5)
        try:
            info = await detect_window(base_url, model, http=http)
        finally:
            await http.aclose()
        return info.tokens
    info = await detect_window(base_url, model, http=http)
    return info.tokens


def _checks_payload(checks: List[Check], *, drop_ok_details: bool) -> str:
    """Compact JSON of the check results; with drop_ok_details the (verbose,
    uninteresting) details of passing checks are left out first."""
    out = []
    for c in checks:
        row = {"name": c.name, "status": c.status, "summary": c.summary}
        if not (drop_ok_details and c.status == "ok"):
            row["details"] = c.details
        out.append(row)
    return json.dumps(out, separators=(",", ":"), ensure_ascii=False,
                      default=str)


def _truncate_playbook(playbook: str, user_prompt: str, budget: int,
                       model: str) -> str:
    """Cut '## ' sections from the END of the playbook until system + user
    prompt fit the budget (plan review item 5)."""
    from proxy.rag.budget import estimate_tokens

    parts = re.split(r"(?m)^(?=## )", playbook)  # [intro, sec1, sec2, ...]
    text = playbook
    while len(parts) > 1 and estimate_tokens(text + user_prompt, model) > budget:
        parts.pop()
        text = "".join(parts)
    return text


def _parse_json_response(text: str) -> Optional[dict]:
    """Strip code fences, take the outermost {...}, parse. None on failure."""
    m = _FENCE_RE.search(text or "")
    body = m.group(1) if m else (text or "")
    start, end = body.find("{"), body.rfind("}")
    if start == -1 or end <= start:
        return None
    try:
        data = json.loads(body[start:end + 1])
    except json.JSONDecodeError:
        return None
    return data if isinstance(data, dict) else None


def _valid_analysis(data: dict) -> bool:
    """Type validation of the breakdown schema (brief R3 item 4)."""
    if not isinstance(data.get("summary"), str):
        return False
    if data.get("severity") not in _SEVERITIES:
        return False
    causes = data.get("likely_causes")
    if not isinstance(causes, list):
        return False
    for c in causes:
        if not isinstance(c, dict):
            return False
        if not all(isinstance(c.get(k), str)
                   for k in ("cause", "evidence", "fix")):
            return False
    what = data.get("what_to_try")
    if not isinstance(what, list) or not all(isinstance(s, str) for s in what):
        return False
    need = data.get("need_checks")
    if need is not None and (not isinstance(need, list)
                             or not all(isinstance(s, str) for s in need)):
        return False
    return True


def _first_line(err: Exception) -> str:
    msg = str(err)
    return " ".join(msg.split())[:200] if msg else ""


async def enhanced_analysis(env: dict, checks: List[Check], description: str, *,
                            client=None, model: Optional[str] = None,
                            run_extra=run_checks) -> dict:
    """Ask SPM's configured model for a structured breakdown of the checks.

    Returns the breakdown dict (summary / likely_causes / what_to_try /
    severity, plus 'need_checks' listing the extra checks it asked for and we
    ran), or {"error": reason} on any failure — connection, timeout, or twice
    invalid JSON — so the caller can still deliver the Traditional report.
    Never raises.
    """
    from proxy.core import reasoning as reasoning_mod
    from proxy.rag.budget import estimate_tokens

    try:
        if client is None:
            from proxy.backend_client.lemonade_client import LemonadeLLMClient
            base = _backend_url(env)
            client = LemonadeLLMClient(base_url=base or None)
        base_url = str(getattr(client, "base_url", "") or "").strip().rstrip("/")

        if model is None:
            from proxy.backend_client.lemonade_client import DEFAULT_CHAT_MODEL
            model = str(env.get("SPM_DEFAULT_MODEL") or "").strip() \
                or DEFAULT_CHAT_MODEL

        # Thinking off (utility call, cloud backends must not be billed for it).
        extra_body = reasoning_mod.extra_body("none", {}, base_url)

        window = None
        try:
            window = await _window_tokens(base_url, model, client)
        except Exception:
            window = None
        budget = int((window or _FALLBACK_WINDOW_TOKENS)
                     * _PROMPT_BUDGET_FRACTION)

        safe_desc = Redactor().user_text(description or "")
        checks_json = _checks_payload(checks, drop_ok_details=False)
        user_prompt = ("User description:\n" + (safe_desc.strip() or "(none)")
                       + "\n\nDiagnostic checks (JSON):\n" + checks_json
                       + "\n\n" + _JSON_INSTRUCTION)
        if estimate_tokens(_playbook_text() + user_prompt, model) > budget:
            # Verbose passing-check details go first, then playbook sections.
            checks_json = _checks_payload(checks, drop_ok_details=True)
            user_prompt = ("User description:\n"
                           + (safe_desc.strip() or "(none)")
                           + "\n\nDiagnostic checks (JSON):\n" + checks_json
                           + "\n\n" + _JSON_INSTRUCTION)
        system_prompt = _truncate_playbook(_playbook_text(), user_prompt,
                                           budget, model)
        messages = [{"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}]

        async def ask(msgs: List[dict]) -> str:
            async def collect() -> str:
                chunks: List[str] = []
                gen = client.generate_stream(
                    messages=msgs, model=model, temperature=_MODEL_TEMPERATURE,
                    max_tokens=_MODEL_MAX_TOKENS, extra_body=extra_body)
                async for chunk in gen:
                    chunks.append(chunk)
                return "".join(chunks)
            return await asyncio.wait_for(collect(), timeout=_MODEL_TIMEOUT_S)

        async def ask_valid(msgs: List[dict]):
            """(data, None) on success; (None, error_reason) on failure."""
            for attempt in (1, 2):
                try:
                    text = await ask(msgs)
                except asyncio.TimeoutError:
                    return None, (f"the model request timed out after "
                                  f"{_MODEL_TIMEOUT_S:g} s")
                except Exception as err:
                    return None, (f"the model server call failed: "
                                  f"{type(err).__name__}: {_first_line(err)}")
                data = _parse_json_response(text)
                if data is not None and _valid_analysis(data):
                    return data, None
                msgs = msgs + [{"role": "user", "content": _RETRY_HINT}]
            return None, "the model did not return valid JSON (after one retry)"

        data, err = await ask_valid(messages)
        if err:
            return {"error": err}

        requested = [n for n in (data.get("need_checks") or [])
                     if n in EXTRA_CHECKS][:_MAX_NEED_CHECKS]
        if requested:
            # One extra round with the requested checks appended (brief R3
            # item 5); any further need_checks in the answer are ignored.
            try:
                extra = await run_extra(env, names=requested)
            except Exception as err:
                return {"error": f"the requested extra checks failed: "
                                 f"{type(err).__name__}: {_first_line(err)}"}
            round2 = messages + [
                {"role": "assistant", "content": json.dumps(data)},
                {"role": "user",
                 "content": "Requested extra check results (JSON):\n"
                            + _checks_payload(extra, drop_ok_details=True)
                            + "\n\n" + _JSON_INSTRUCTION},
            ]
            data2, err2 = await ask_valid(round2)
            if err2:
                return {"error": err2}
            data = data2

        result = {
            "summary": data["summary"],
            "likely_causes": data["likely_causes"],
            "what_to_try": data["what_to_try"],
            "severity": data["severity"],
        }
        if requested:
            result["need_checks"] = requested
        return result
    except Exception as err:  # belt and braces: the report must still build
        return {"error": f"{type(err).__name__}: {_first_line(err)}"}


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


def _enhanced_markdown(enhanced) -> List[str]:
    """The '## Enhanced analysis' section (brief R3 item 7). Accepts the
    breakdown dict from enhanced_analysis, {'error': reason}, or None (no
    section)."""
    if not enhanced:
        return []
    if not isinstance(enhanced, dict):
        return ["", "## Enhanced analysis", "", str(enhanced)]
    if enhanced.get("error"):
        return ["", "## Enhanced analysis", "",
                f"Enhanced analysis unavailable: {enhanced['error']}"]
    lines = ["", "## Enhanced analysis", "", "Summary: " + str(enhanced.get("summary", ""))]
    causes = enhanced.get("likely_causes") or []
    lines += ["", "Likely causes:"]
    if not causes:
        lines.append("- (none identified)")
    for c in causes:
        if isinstance(c, dict):
            lines.append(f"- {c.get('cause', '')} — evidence: {c.get('evidence', '')}"
                         f" — fix: {c.get('fix', '')}")
        else:
            lines.append(f"- {c}")
    lines += ["", "What to try:"]
    for step in (enhanced.get("what_to_try") or []):
        lines.append(f"- {step}")
    lines += ["", f"Severity: {enhanced.get('severity', 'unknown')}"]
    if enhanced.get("need_checks"):
        lines += ["", "Checks requested by the analysis: "
                  + ", ".join(str(n) for n in enhanced["need_checks"])]
    return lines


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
    lines += _enhanced_markdown(enhanced)
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
                       repo_root=None, enhanced_client=None) -> Report:
    """Run the checks, redact, render the Markdown, write the zip bundle and
    (public reports only) build the issue URL. Never raises for failed checks:
    a report about a broken SPM is the point.

    ``enhanced=True`` runs the R3 model-powered analysis over the redacted
    description and checks (``enhanced_client`` overrides the model client —
    tests pass a fake); a breakdown dict is rendered directly; failures degrade
    to an "Enhanced analysis unavailable" line."""
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
        desc_shown = redactor.user_text(description)

    if enhanced is True:
        enhanced = await enhanced_analysis(env, checks, desc_shown,
                                           client=enhanced_client)
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


def _backend_url(env: dict, repo_root=None) -> str:
    """The model server SPM really uses: the settings file wins over the environment
    (same rule as the checks). Reading only the environment pointed the Enhanced analysis
    at Lemonade on a machine where SPM used another server: Lemonade would have loaded a
    chat model next to a running gufo (Claude's review, 2026-10-14)."""
    from types import SimpleNamespace

    from .diagnostics import _effective_settings
    root = Path(repo_root) if repo_root is not None else default_repo_root()
    try:
        settings = _effective_settings(env, SimpleNamespace(root=root))[0]
        return str(settings.get("BACKEND_LLM_URL") or "").strip().rstrip("/")
    except Exception:
        return str(env.get("BACKEND_LLM_URL") or "").strip().rstrip("/")


def _enhanced_consent(env: dict, *, yes: bool, input_fn=None) -> bool:
    """Cloud confirmation for --enhanced (brief R3): when the configured model
    server is not local, the redacted diagnostics leave this machine, so ask
    first. True = go ahead; False = fall back to the Traditional report."""
    from proxy.core.reasoning import _is_local

    base = _backend_url(env)
    if not base or _is_local(base):
        return True
    host = urllib.parse.urlparse(base).hostname or base
    print(f"Enhanced analysis sends the redacted diagnostics to {host}, "
          "a remote service (this may cost tokens).")
    if yes:
        return True
    prompt = "Continue? [y/N]: "
    try:
        answer = (input_fn(prompt) if input_fn else input(prompt)).strip().lower()
    except (EOFError, KeyboardInterrupt):
        print()
        return False
    return answer in ("y", "yes")


def cmd_report(repo_root: Path, *, description: str = "", system_info: bool = False,
               full_logs: bool = False, yes: bool = False, enhanced: bool = False,
               input_fn=None, enhanced_client=None) -> int:
    """`python -m spm report`: print the check table, the bundle path and the
    issue URL. Always exits 0 (even when checks fail)."""
    if not description and (input_fn is not None or _stdin_tty()):
        prompt = "What went wrong? (one line, Enter to skip): "
        try:
            description = (input_fn(prompt) if input_fn else input(prompt)).strip()
        except (EOFError, KeyboardInterrupt):
            print()
            description = ""

    env = child_env(repo_root)
    if enhanced and not _enhanced_consent(env, yes=yes, input_fn=input_fn):
        print("Falling back to the Traditional report.")
        enhanced = False
    report = asyncio.run(build_report(env, description=description,
                                      system_info=system_info,
                                      full_logs=full_logs, enhanced=enhanced,
                                      repo_root=repo_root,
                                      enhanced_client=enhanced_client))

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
