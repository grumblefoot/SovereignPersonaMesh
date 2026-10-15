"""Redaction for bug reports (privacy core).

Pure, standard-library-only helpers that turn SPM logs and strings into
something safe to attach to a public GitHub issue. See
docs/plans/bug_report_v04.md ("Privacy rules", "Developer use", "Changes after
Hermes plan review") for the policy this encodes:

- message-content log lines are dropped entirely (public mode);
- secrets (api keys, passwords, secrets, bearer tokens, sk-... keys) -> ***;
- URL credentials (scheme://user:pass@host) -> scheme://***@host;
- home paths -> ~, other users' home roots -> ~user, the OS user name -> user;
- session ids -> stable short hashes (chat#a1b2);
- character names -> tokens (also plural/possessive forms: "zeldas",
  "Zelda's"), long quoted strings -> "<redacted N chars>" (public mode only);
- brackets containing 4+ quoted items (keyword bags) ->
  "[<redacted list: N items>]" (public mode only);
- runs of 8+ consecutive words in the message part of a log line ->
  "<redacted N words>", keeping the "<timestamp> [LEVEL] <logger>: " prefix;
  URLs, file paths, key=value pairs, numbers and quoted spans are not counted
  (public mode only);
- public IPv4 -> x.x.x.x, LAN IPs keep the first two octets, loopback stays.
"""

from __future__ import annotations

import getpass
import hashlib
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, List, Mapping, Optional

__all__ = ["LOG_LINE", "CONTENT_MARKERS", "Redactor"]

# A line produced by the logger (see proxy/core/logger.py LOG_FORMAT) or by
# uvicorn's console output. Anything else is a continuation of a multi-line
# message and carries prose, so the public report drops it.
LOG_LINE = re.compile(r"^\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}|^INFO:")

# Substrings that mark a line as carrying message content (public mode drops
# these lines entirely).
CONTENT_MARKERS = (
    "[SillyIntoSPMLog]",
    "[SPMIntoBackendLog]",
    "[BackendReturnSPMLog]",
    "[SPMReturnSillyLog]",
    "Request payload",
    "Sending to Backend",
)

# Lines whose JSON-ish body embeds message content.
_CONTENT_MARKS = ("'content':", '"content":', "'messages':", '"messages":')

# --- patterns (compiled once at import; all linear-time) -----------------------

# A secret key is a bare word (optionally quoted, optionally prefixed/suffixed,
# e.g. POSTGRES_PASSWORD, X_API_KEY, "api_key") containing one of the secret
# words, followed by = or : and a quoted or bare value. Bearer is masked first
# so "Authorization: Bearer <tok>" keeps its shape.
_SECRET_WORDS = r"(?:api[_-]?key|password|passwd|secret|token|authorization)"
_SECRET_KV_RE = re.compile(
    r"""(?xi)
    (?<![A-Za-z0-9_-])
    (?P<key>["']?[A-Za-z0-9_.-]*""" + _SECRET_WORDS + r"""[A-Za-z0-9_.-]*["']?)
    (?P<sep>[ \t]*[=:][ \t]*)
    (?P<val>"[^"\\]*(?:\\.[^"\\]*)*"            # "double-quoted value"
          |'[^'\\]*(?:\\.[^'\\]*)*'             # 'single-quoted value'
          |(?!\*\*\*)[^\s,;}\]"']+)             # or a bare value
    """
)
_BEARER_RE = re.compile(r"(?i)\bBearer[ \t]+[A-Za-z0-9._~+/=-]{2,}")
_SK_RE = re.compile(r"\bsk-[A-Za-z0-9_-]{16,}")

# scheme://user:pass@host -> scheme://***@host
_URL_CRED_RE = re.compile(r"([A-Za-z][A-Za-z0-9+.-]*://)[^\s/@:]+:[^\s/@]+@")

# Session ids: the SillyTavern-style id, and session= / session_id= values
# (already-tokenised "chat#xxxx" values are left alone).
_ST_CHAT_RE = re.compile(r"st_chat_[0-9A-Za-z_]+")
_SESSION_KV_RE = re.compile(
    r"(?i)\b(session_id|session)[ \t]*=[ \t]*(?!chat#)([A-Za-z0-9][^\s,;}\]]*)"
)

# Quoted spans with escape handling; kept linear-time for long lines.
_QUOTE_RE = re.compile(r'"[^"\\]*(?:\\.[^"\\]*)*"|\'[^\'\\]*(?:\\.[^\'\\]*)*\'')
_QUOTE_MIN = 40  # redact a quoted span when its inner text exceeds this

# Lists of short quoted words (keyword bags built from the chat, brief
# R1a-2 leak 1): any [...] or (...) holding 4 or more quoted items.
_LIST_BRACKET_RE = re.compile(r"\[[^[\]\n]*\]|\([^()\n]*\)")
_LIST_MIN_ITEMS = 4

# The message part of a log line is everything after the logger prefix
# "<timestamp> [LEVEL] <logger>: " (or uvicorn's "INFO:     "); the prefix is
# kept readable (brief R1a-2 leak 3).
_MESSAGE_PREFIX_RE = re.compile(
    r"^(?:\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}(?:[.,]\d+)?"
    r" \[[A-Za-z]+\] [^:]+: |INFO:[ \t]+)"
)
_WORD_RUN_MIN = 8
# Anything carrying one of these characters (URLs, file paths, key=value,
# numbers, IPs, session tokens, placeholders) is never a word and never part
# of a word run.
_TECH_CHARS = frozenset("=/\\@#%$&*<>_+")
# A prose word is letters with optional internal apostrophes, hyphens or
# commas; trailing punctuation (including a comma) is stripped before the
# check (a leading apostrophe is not, so quoted spans never count as words).
_WORD_EDGE_STRIP_RE = re.compile(r"[^\w'’\-]+$")
_WORD_CORE_RE = re.compile(r"[^\W\d_]+(?:['’‘,\-][^\W\d_]+)*")

# Other users' home roots: /home/<x>, /Users/<x>, C:\Users\<x> -> ~user
# (drive-letter paths are unreliable in case, so matching is case-insensitive).
_OTHER_HOME_RE = re.compile(
    r"(?:[A-Za-z]:)?[\\/](?:home|Users)[\\/][^\\/\s\"':,;}\]]+",
    re.IGNORECASE,
)

# IPv4 with plausible octets; the lookarounds keep versions and ports attached.
_IP_RE = re.compile(
    r"(?<![0-9A-Za-z.])(\d{1,3})\.(\d{1,3})\.(\d{1,3})\.(\d{1,3})(?![0-9.])"
)


def _mask_ip(m: re.Match) -> str:
    a, b, c, d = (int(g) for g in m.groups())
    if not all(0 <= v <= 255 for v in (a, b, c, d)):
        return m.group(0)
    if a == 127:  # loopback stays
        return m.group(0)
    if a == 10 or a == 192 or (a == 172 and 16 <= b <= 31):
        return f"{a}.{b}.x.x"
    return "x.x.x.x"


def _chat_token(value: str) -> str:
    """Stable short token for a session id (stable within a report)."""
    return "chat#" + hashlib.sha1(value.encode("utf-8")).hexdigest()[:4]


def _sub_secret(m: re.Match) -> str:
    val = m.group("val")
    quote = val[0] if val[:1] in ("'", '"') else ""
    if quote:
        return f"{m.group('key')}{m.group('sep')}{quote}***{quote}"
    return f"{m.group('key')}{m.group('sep')}***"


def _redact_quote(m: re.Match) -> str:
    span = m.group(0)
    inner = span[1:-1]
    if len(inner) > _QUOTE_MIN:
        return f'"<redacted {len(inner)} chars>"'
    return span


def _redact_quoted_list(m: re.Match) -> str:
    """[...]/(...) holding >= _LIST_MIN_ITEMS quoted items -> redacted list."""
    span = m.group(0)
    n_items = len(_QUOTE_RE.findall(span))
    if n_items >= _LIST_MIN_ITEMS:
        return f"[<redacted list: {n_items} items>]"
    return span


def _is_prose_word(tok: str) -> bool:
    """A bare letters word, optionally with internal apostrophes/commas.

    Tokens carrying digits or technical characters (URL, path, key=value,
    IP, session token, placeholder) are never words, so they break runs.
    """
    core = _WORD_EDGE_STRIP_RE.sub("", tok)
    if not core or _TECH_CHARS & set(core):
        return False
    return _WORD_CORE_RE.fullmatch(core) is not None and core[0].isalpha()


def _redact_word_run_seq(seq: str) -> str:
    """Replace runs of >= _WORD_RUN_MIN consecutive prose words in `seq`.

    Greedy maximal runs; the replacement is "<redacted N words>". Tokens
    that are not prose words (numbers, key=value, URLs, paths, quoted
    spans and their placeholders) break a run and pass through untouched.
    """
    parts = seq.split(" ")
    out: List[str] = []
    run: List[str] = []

    def flush() -> None:
        if len(run) >= _WORD_RUN_MIN:
            out.append(f"<redacted {len(run)} words>")
        else:
            out.extend(run)
        run.clear()

    for tok in parts:
        if _is_prose_word(tok):
            run.append(tok)
        else:
            flush()
            out.append(tok)
    flush()
    return " ".join(out)


_RUN_PLACEHOLDER_RE = re.compile(r"<redacted \d+ words>")
# Identifiers made of 7+ snake_case segments are text turned into ids (an old character
# id held its card description: zed_is_a_tall_figure_in_a_long_grey...).
_LONG_ID_RE = re.compile(r"(?<![A-Za-z0-9])[A-Za-z0-9]+(?:_[A-Za-z0-9]+){6,}(?![A-Za-z0-9])")


def _redact_message_prose(line: str) -> str:
    """Apply the word-run rule to the message part of a log line only.

    The "<timestamp> [LEVEL] <logger>: " (or uvicorn "INFO: ") prefix is
    kept; lines that do not look like logger output get the rule applied to
    the whole line.
    """
    m = _MESSAGE_PREFIX_RE.match(line)
    prefix, msg = (line[: m.end()], line[m.end():]) if m else ("", line)
    msg = _redact_word_run_seq(msg)
    # Whatever follows prose in the same message is usually more of it (a short quoted
    # word survived after a redacted lore rule in a real-log check): drop it.
    hit = _RUN_PLACEHOLDER_RE.search(msg)
    if hit and msg[hit.end():].strip():
        msg = msg[: hit.end()] + " <redacted rest>"
    return prefix + msg


@dataclass
class Redactor:
    """Redactor for bug-report material.

    names         exact strings -> tokens, e.g. {"zelda": "char#1"}; each key
                  is replaced as a whole word, case-insensitively, longest key
                  first, and also in plural/possessive form (trailing "s",
                  "'s" or "’s": "zeldas", "Zelda's", "zeds").
    home          the user's home directory; defaults to Path.home().
    user          the OS user name; defaults to getpass.getuser() (guarded);
                  ignored when shorter than 3 characters.
    keep_content  True only for the private full-logs mode: lines() skips the
                  content-line filter, and the quote and name rules of text()
                  are skipped; the secret, URL-credential, home-path,
                  user-name, session-id and IP rules still apply.
    """

    names: Mapping[str, str] = field(default_factory=dict)
    home: Optional[str] = None
    user: Optional[str] = None
    keep_content: bool = False

    # derived caches (not part of equality/repr)
    _cache_key: Optional[tuple] = field(default=None, repr=False, compare=False)
    _name_re: Optional[re.Pattern] = field(default=None, repr=False, compare=False)
    _name_lookup: Mapping[str, str] = field(default_factory=dict, repr=False, compare=False)
    _home_res: tuple = field(default=(), repr=False, compare=False)
    _user_re: Optional[re.Pattern] = field(default=None, repr=False, compare=False)

    # --- derived settings ---------------------------------------------------------

    def _derive(self) -> None:
        key = (tuple(sorted(self.names.items())), self.home, self.user)
        if self._cache_key == key:
            return
        self._cache_key = key

        # Names: longest key first (alternation order), whole word, ci lookup.
        # Also matches plural/possessive forms: a trailing "s", "'s" or "’s"
        # ("zeldas", "Zelda's", "zeds"); the token maps to the base
        # name's value.
        names = sorted((k for k in self.names if k), key=len, reverse=True)
        if names:
            joined = "|".join(re.escape(n) for n in names)
            self._name_re = re.compile(
                # Underscores count as a boundary, so names inside identifiers such as
                # csa_memory_zelda are replaced too (real-log check 2026-10-14).
                r"(?<![A-Za-z0-9])(?P<name>" + joined + r")(?:['’‘]?s)?(?![A-Za-z0-9])", re.IGNORECASE
            )
            self._name_lookup = {k.lower(): v for k, v in self.names.items()}
        else:
            self._name_re = None
            self._name_lookup = {}

        # Home directory -> "~": POSIX and Windows forms, forward and back
        # slashes, case-insensitive for drive-letter paths.
        home = self.home if self.home is not None else str(Path.home())
        variants = sorted(
            {home, home.replace("\\", "/"), home.replace("/", "\\")},
            key=len,
            reverse=True,
        )
        res = []
        for variant in variants:
            if not variant:
                continue
            flags = re.IGNORECASE if re.match(r"^[A-Za-z]:[\\/]", variant) else 0
            res.append(re.compile(re.escape(variant) + r"(?![\w.-])", flags))
        self._home_res = tuple(res)

        # OS user name as a whole word -> "user" (skip if shorter than 3 chars).
        user = self.user
        if user is None:
            try:
                user = getpass.getuser()
            except Exception:
                user = None
        if user and len(user) >= 3:
            self._user_re = re.compile(
                r"(?<![A-Za-z0-9_])" + re.escape(user) + r"(?![A-Za-z0-9_])"
            )
        else:
            self._user_re = None

    # --- public API -------------------------------------------------------------------

    def text(self, s: str) -> str:
        """Redact one string with the full public-mode rule set."""
        return self._apply(s, keep_content=False)

    def user_text(self, s: str) -> str:
        """Redact text the USER wrote for the report (the problem description). It is
        meant to be read: keep its prose and quotes, remove secrets, URL credentials,
        paths, the user name, session ids, IPs and character/persona names. The public
        rule set wiped a whole description, so the Enhanced analysis never saw the
        symptom (Claude's real run, 2026-10-14)."""
        s = self._apply(s, keep_content=True)
        if self._name_re is not None:
            lookup = self._name_lookup
            s = self._name_re.sub(lambda m: lookup[m.group("name").lower()], s)
        return s

    def lines(self, lines: Iterable[str]) -> List[str]:
        """Filter and redact log lines.

        Public mode (keep_content=False): keep only lines matching LOG_LINE,
        drop lines carrying message content (CONTENT_MARKERS and content/
        messages keys), then apply text() to what remains (which also redacts
        quoted-item lists, names and prose word runs of 8+ in the message
        part of the line).
        Private developer mode (keep_content=True): keep every line, applying
        only the secret, URL-credential, home-path, user-name, session-id and
        IP rules (not the quote and name rules).
        """
        out: List[str] = []
        for raw in lines:
            line = raw.rstrip("\r\n")
            if not self.keep_content:
                if not LOG_LINE.match(line):
                    continue
                if any(marker in line for marker in CONTENT_MARKERS):
                    continue
                if any(mark in line for mark in _CONTENT_MARKS):
                    continue
                out.append(self._apply(line, keep_content=False))
            else:
                out.append(self._apply(line, keep_content=True))
        return out

    # --- internals ----------------------------------------------------------------------

    def _apply(self, s: str, keep_content: bool) -> str:
        self._derive()

        # Secrets (Bearer first so the header keeps its shape).
        s = _BEARER_RE.sub("Bearer ***", s)
        s = _SECRET_KV_RE.sub(_sub_secret, s)
        s = _SK_RE.sub("***", s)

        # URL credentials.
        s = _URL_CRED_RE.sub(r"\1***@", s)

        # Home paths, other users' homes, the OS user name.
        for pat in self._home_res:
            s = pat.sub("~", s)
        s = _OTHER_HOME_RE.sub("~user", s)
        if self._user_re is not None:
            s = self._user_re.sub("user", s)

        # Session ids -> stable chat#xxxx tokens.
        s = _ST_CHAT_RE.sub(lambda m: _chat_token(m.group(0)), s)
        s = _SESSION_KV_RE.sub(lambda m: f"{m.group(1)}={_chat_token(m.group(2))}", s)

        if not keep_content:
            # Keyword-bag lists: [...] or (...) with 4+ quoted items first,
            # before the quote rule can touch their contents.
            s = _LIST_BRACKET_RE.sub(_redact_quoted_list, s)
            # Character / world / persona names (incl. plural/possessive).
            if self._name_re is not None:
                lookup = self._name_lookup
                s = self._name_re.sub(
                    lambda m: lookup[m.group("name").lower()], s
                )
            # Text turned into long snake_case identifiers.
            s = _LONG_ID_RE.sub("<id>", s)
            # Long quoted strings.
            s = _QUOTE_RE.sub(_redact_quote, s)
            # Unquoted prose runs in the message part of the line (uses the
            # redacted form so quoted spans, tokens and technical values no
            # longer count as words).
            s = _redact_message_prose(s)

        # IP addresses.
        s = _IP_RE.sub(_mask_ip, s)
        return s
