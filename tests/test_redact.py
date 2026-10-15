"""Tests for spm/redact.py (bug-report redaction).

All log lines are synthetic, in the shapes of the real logger output
(proxy/core/logger.py: "YYYY-MM-DD HH:MM:SS,mmm [LEVEL] name: message" plus
uvicorn's "INFO:     ..." console lines). No real log content is used.
"""

import time

import pytest

from spm.redact import CONTENT_MARKERS, LOG_LINE, Redactor

HOME = "/home/kestrel"
USER = "kestrel"
NAMES = {"zelda": "char#1", "Lord Vane": "char#2", "zed": "char#3"}

# Planted values that must never survive redaction.
SECRETS = [
    "hunter2hunter2",
    "abcdef0123456789abcdef01",          # sk- key body
    "secret-token-xyz789",
    "S3cr3tPassphrase!",
    "hunter2admin",
]
PRIVATE_SENTENCES = [
    "the dragon woke and the entire village fled into the dark mountains overnight",
    "A very private apology that nobody should ever read in a public bug report",
    "I love you more than all the stars in the winter sky above the old keep",
]
PATHS = ["/home/kestrel/Desktop", "C:\\Users\\Brenda", "Brenda", "kestrel"]
SESSION_IDS = ["st_chat_Qh8sKd2mZx47pLw9", "deadbeefcafe1234"]

# A quoted error sentence of exactly 60 characters.
SENTENCE_60 = "Backend connection refused during chat request; retry limit."
assert len(SENTENCE_60) == 60


def _synthetic_log():
    """30+ synthetic lines covering every shape the brief lists."""
    return [
        # 1-2: plain startup lines
        "2026-10-14 06:57:25,127 [INFO] SPMProxyMain: Starting SPM Proxy Server on http://127.0.0.1:5053 (reload=False)",
        "INFO:     Started server process [539404]",
        # 3-6: content-bearing lines (markers and quoted content keys)
        "2026-10-14 06:57:30,001 [INFO] proxy.routes: [SillyIntoSPMLog] incoming chat request from client",
        "2026-10-14 06:57:30,002 [INFO] proxy.routes: Request payload: {'messages': [{'role': 'user', 'content': 'the dragon woke and the entire village fled into the dark mountains overnight'}]}",
        "2026-10-14 06:57:30,010 [INFO] proxy.backend_client: Sending to Backend model=google/gemma-4-26b",
        '2026-10-14 06:57:31,000 [INFO] proxy.routes: [BackendReturnSPMLog] reply ready {"content": "A very private apology that nobody should ever read in a public bug report"}',
        # 7-8: a multi-line reply and its continuation (no timestamp — dropped
        #      in public mode, kept in developer mode)
        "2026-10-14 06:57:31,001 [INFO] proxy.routes: [SPMReturnSillyLog] streaming reply:",
        "I love you more than all the stars in the winter sky above the old keep",
        # 9: INFO access line with a loopback client
        'INFO:     127.0.0.1:56232 - "GET /health HTTP/1.1" 200 OK',
        # 10: error with a quoted 60-character sentence
        f"2026-10-14 06:58:00,500 [ERROR] proxy.engine: '{SENTENCE_60}'",
        # 11-14: secrets in several shapes
        "2026-10-14 06:58:01,000 [WARNING] spm.setup: POSTGRES_PASSWORD=hunter2hunter2 found in environment",
        '2026-10-14 06:58:01,100 [INFO] spm.setup: config loaded {"api_key": "sk-abcdef0123456789abcdef01", "model": "gemma"}',
        "2026-10-14 06:58:01,200 [INFO] httpx: HTTP Request: GET http://192.168.1.20:1234/v1/models Authorization: Bearer secret-token-xyz789",
        "2026-10-14 06:58:01,300 [INFO] spm.setup: db url postgresql://spm_user:***@localhost/spm_db",
        # 15-17: paths and user name
        "2026-10-14 06:58:02,000 [INFO] spm.setup: log rotation checked C:\\Users\\Brenda\\SPM\\logs\\x.log",
        "2026-10-14 06:58:02,100 [INFO] spm.setup: venv at /home/kestrel/Desktop/Experiments/SillyTavern/SPM/.venv",
        "2026-10-14 06:58:02,200 [INFO] spm.setup: service runs as user kestrel (home " + HOME + ")",
        # 18-19: session id and a character name
        "2026-10-14 06:58:03,000 [INFO] proxy.routes: resolving session=st_chat_Qh8sKd2mZx47pLw9 for chat window",
        "2026-10-14 06:58:03,100 [INFO] proxy.core.perception: memory write skipped for Zelda (cooldown)",
        # 20: a public and a private IP
        "2026-10-14 06:58:04,000 [WARNING] proxy.middleware: rate-limited client 203.0.113.9; peer lan 10.42.7.19 and 172.20.5.6",
        # 21-24: more ordinary lines to stay above 30 overall
        "2026-10-14 06:58:05,000 [INFO] proxy.core.telemetry: [Telemetry] Hydrated telemetry buffers from database.",
        "2026-10-14 06:58:05,100 [INFO] proxy.core.sleep_runner: [SleepCycle] next consolidation pass in 21.0 h (at 04:00)",
        "INFO:     Application startup complete.",
        "INFO:     Uvicorn running on http://127.0.0.1:5053 (Press CTRL+C to quit)",
        # 25-26: another payload variant (double-quoted messages key)
        '2026-10-14 06:58:06,000 [INFO] proxy.routes: Request payload: {"messages": [{"role": "user", "content": "more private roleplay prose that must not leak"}]}',
        "  ...and its continuation line, also prose",
        # 27-28: sk- key bare, and password in key: value form
        "2026-10-14 06:58:07,000 [ERROR] proxy.backend_client: backend rejected key sk-abcdef0123456789abcdef01 (401)",
        "2026-10-14 06:58:07,100 [WARNING] spm.setup: settings audit: password: S3cr3tPassphrase! admin password: 'hunter2admin'",
        # 29-32: ordinary tail
        "2026-10-14 06:58:08,000 [INFO] proxy.backend_client.backend_profile: [BackendProfile] window 32768 (lmstudio-loaded)",
        'INFO:     127.0.0.1:56246 - "GET /admin/api/v1/backend/window HTTP/1.1" 200 OK',
        "2026-10-14 06:58:09,000 [INFO] SPMProxyMain: [SPMProxyMain] Database pool closed.",
        "INFO:     Finished server process [539404]",
    ]


@pytest.fixture()
def log():
    lines = _synthetic_log()
    assert len(lines) >= 30
    return lines


@pytest.fixture()
def public():
    return Redactor(names=NAMES, home=HOME, user=USER)


@pytest.fixture()
def private():
    return Redactor(names=NAMES, home=HOME, user=USER, keep_content=True)


# --- module surface -------------------------------------------------------------


def test_log_line_pattern():
    assert LOG_LINE.match("2026-10-14 06:57:25,127 [INFO] x: hi")
    assert LOG_LINE.match("INFO:     Started server process [1]")
    assert not LOG_LINE.match("  continuation of a message")
    assert not LOG_LINE.match("2026-13-99 x")


def test_content_markers_exact():
    assert CONTENT_MARKERS == (
        "[SillyIntoSPMLog]",
        "[SPMIntoBackendLog]",
        "[BackendReturnSPMLog]",
        "[SPMReturnSillyLog]",
        "Request payload",
        "Sending to Backend",
    )


# --- public mode: nothing private survives ----------------------------------------


class TestPublicOutput:
    def test_no_secrets(self, public, log):
        out = "\n".join(public.lines(log))
        for s in SECRETS:
            assert s not in out

    def test_no_roleplay_prose(self, public, log):
        out = "\n".join(public.lines(log))
        for s in PRIVATE_SENTENCES:
            assert s not in out
        assert "more private roleplay prose" not in out

    def test_no_user_names_or_home_paths(self, public, log):
        out = "\n".join(public.lines(log))
        assert "kestrel" not in out
        assert "Brenda" not in out
        assert "/home/kestrel" not in out
        # the raw C:\Users\Brenda path is gone (the ~user form is acceptable)
        assert "C:\\Users\\Brenda\\SPM" not in out
        assert "\\Users\\Brenda" not in out

    def test_no_raw_session_ids(self, public, log):
        out = "\n".join(public.lines(log))
        for sid in SESSION_IDS:
            assert sid not in out
        assert "session=chat#" in out  # replaced by a stable token

    def test_no_character_names(self, public, log):
        out = "\n".join(public.lines(log))
        assert "Zelda" not in out
        assert "char#1" in out

    def test_error_line_timestamps_levels_survive(self, public, log):
        out = "\n".join(public.lines(log))
        assert "[ERROR]" in out
        # the quoted 60-char sentence is masked, the rest of the line readable
        assert "proxy.engine" in out
        assert "<redacted 60 chars>" in out
        assert "2026-10-14 06:57:25" in out
        assert "[INFO]" in out
        assert "[WARNING]" in out
        assert "GET /health" in out

    def test_content_lines_dropped(self, public, log):
        out = public.lines(log)
        joined = "\n".join(out)
        for marker in CONTENT_MARKERS:
            assert marker not in joined
        assert "'content':" not in joined
        assert '"content":' not in joined
        assert "'messages':" not in joined
        assert '"messages":' not in joined

    def test_continuation_lines_dropped(self, public, log):
        out = public.lines(log)
        assert all(LOG_LINE.match(line) for line in out)

    def test_ips_masked(self, public, log):
        out = "\n".join(public.lines(log))
        assert "203.0.113.9" not in out
        assert "x.x.x.x" in out           # public IP masked
        assert "10.42.x.x" in out         # private keeps two octets
        assert "172.20.x.x" in out
        assert "192.168.x.x" in out
        assert "127.0.0.1" in out         # loopback stays

    def test_quoted_over_40_redacted(self, public):
        sentence = "a quoted sentence that is comfortably longer than forty characters!!"
        out = public.text(f"failed on token '{sentence}' here")
        assert f"<redacted {len(sentence)} chars>" in out
        assert sentence not in out
        assert "failed on token" in out and "here" in out

    def test_short_quotes_kept(self, public):
        assert public.text('level "info" ok') == 'level "info" ok'

    # --- brief R1a-2: three leaks found on real logs (synthetic shapes) ------

    def test_quoted_word_list_redacted(self, public):
        # Leak 1: keyword bags built from the chat.
        line = (
            "2026-10-14 06:59:00,000 [INFO] proxy.worldbuilder: "
            "[HybridWorldBuilder] Best keyword candidate "
            "template=hidden_grove score=7.00 for keywords="
            "['write', 'seraphinas', 'next', 'reply', 'glade', 'lantern', 'dream']"
        )
        out = public.lines([line])[0]
        assert "seraphinas" not in out
        assert "[<redacted list: 7 items>]" in out
        # the technical part of the line stays readable
        assert "template=hidden_grove" in out
        assert "score=7.00" in out

    def test_parenthesized_quoted_list_redacted(self, public):
        out = public.text("picked items ('one', 'two', 'three', 'four') from bag")
        assert "[<redacted list: 4 items>]" in out
        assert "'two'" not in out
        # 3 quoted items: below the threshold, untouched
        assert public.text("picked ('one', 'two', 'three')") == "picked ('one', 'two', 'three')"

    def test_name_possessive_and_plural_forms(self, public):
        # Leak 2: trailing "s", "'s" and "’s" on names in `names`.
        assert public.text("the zeldas of the old keep") == "the char#1 of the old keep"
        assert public.text(" zeds marched north") == " char#3 marched north"
        assert "char#1" in public.text("Zelda's glade was quiet")
        assert "Zelda" not in public.text("Zelda\u2019s glade was quiet")
        # whole word only
        assert public.text("Zeldaspen") == "Zeldaspen"

    def test_unquoted_prose_run_redacted(self, public):
        # Leak 3: lore-rule prose after the logger prefix.
        line = (
            "2026-10-14 06:59:01,000 [INFO] "
            "proxy.rag.lore_extractor.LoreExtractionWorker: Skipping "
            "near-duplicate lore rule: If a character attempts to access "
            "the wine cellar, the keeper must be notified at once"
        )
        out = public.lines([line])[0]
        assert "attempts to access" not in out
        assert "wine cellar" not in out
        # the logger prefix survives
        assert out.startswith(
            "2026-10-14 06:59:01,000 [INFO] "
            "proxy.rag.lore_extractor.LoreExtractionWorker: "
        )
        assert "<redacted 20 words>" in out

    def test_short_technical_messages_survive(self, public):
        # Brief R1a-2: these must stay readable.
        turn = (
            "2026-10-14 06:59:02,000 [INFO] proxy.core.engine: "
            "Turn finished for char#1. Monologue chars: 0, Public chars: 2194"
        )
        assert public.lines([turn])[0] == turn
        req = (
            "2026-10-14 06:59:03,000 [INFO] httpx: HTTP Request: POST "
            "http://192.168.1.20:1234/v1/chat/completions HTTP/1.1"
        )
        out = public.lines([req])[0]
        assert "HTTP Request" in out
        assert "http://192.168.x.x:1234/v1/chat/completions" in out
        assert "HTTP/1.1" in out
        acc = 'INFO:     127.0.0.1:56232 - "GET /health HTTP/1.1" 200 OK'
        assert public.lines([acc])[0] == acc
        # key=value pairs, paths, URLs and numbers never join a word run
        # (the lead-in here is deliberately only 7 words)
        r = public.text(
            "synced settings from the fallback file at "
            "server_host=localhost workers=4 mode=hybrid "
            "/home/kestrel/etc/spm/config.yaml http://example.invalid/v1 42"
        )
        assert "<redacted" not in r


# --- developer mode: prose stays, secrets still go -----------------------------------


class TestKeepContentMode:
    def test_prose_and_payload_kept(self, private, log):
        out = "\n".join(private.lines(log))
        for s in PRIVATE_SENTENCES:
            assert s in out
        assert "Request payload" in out
        assert "[SillyIntoSPMLog]" in out
        # continuation lines survive too
        assert len(private.lines(log)) == len(log)

    def test_secrets_still_removed(self, private, log):
        out = "\n".join(private.lines(log))
        for s in SECRETS:
            assert s not in out

    def test_paths_and_user_still_removed(self, private, log):
        out = "\n".join(private.lines(log))
        assert "kestrel" not in out
        assert "Brenda" not in out
        assert "/home/kestrel" not in out
        assert "\\Users\\Brenda" not in out

    def test_names_and_quotes_not_touched(self, private):
        longish = "a quoted developer message that runs longer than forty characters!"
        line = f"2026-10-14 06:57:25,000 [INFO] x: Zelda said '{longish}'"
        out = private.lines([line])[0]
        assert "Zelda" in out           # name rule skipped in this mode
        assert longish in out             # quote rule skipped in this mode
        # text() itself always applies the full public rule set
        assert private.text(f"Zelda said '{longish}'").startswith("char#1")


# --- text() rule details ---------------------------------------------------------------


class TestTextRules:
    def test_secret_forms(self):
        r = Redactor(home=HOME, user=USER)
        assert "hunter2" not in r.text("POSTGRES_PASSWORD=hunter2hunter2")
        assert "abc" not in r.text('{"api_key": "abcdef0123456789abcdef01"}')
        assert "top" not in r.text("{'api_key': 'topsecretvalue'}")
        assert "***" in r.text("password: ***")
        assert r.text("Bearer abcdef.ghij-123") == "Bearer ***"
        assert "abcdefghij1234567890" not in r.text("key sk-abcdefghij1234567890 end")
        assert r.text("***") == "***"

    def test_url_credentials(self):
        r = Redactor(home=HOME, user=USER)
        out = r.text("postgresql://spm_user:***@localhost/db")
        assert out == "postgresql://***@localhost/db"

    def test_home_path_forms(self):
        r = Redactor(home="C:\\Users\\Brenda", user="brenda")
        assert r.text(r"C:\Users\Brenda\SPM\logs\x.log") == "~\\SPM\\logs\\x.log"
        assert r.text("C:/Users/brenda/spm/x.log") == "~/spm/x.log"

    def test_posix_home_and_other_users(self):
        r = Redactor(home="/home/kestrel", user="kestrel")
        assert r.text("/home/kestrel/Desktop/x.txt") == "~/Desktop/x.txt"
        assert r.text("/home/other/x.txt") == "~user/x.txt"

    def test_user_name_whole_word(self):
        r = Redactor(home=HOME, user="kestrel")
        assert r.text("runs as kestrel here") == "runs as user here"
        assert r.text("kestrelson") == "kestrelson"  # not a whole word
        short = Redactor(home=HOME, user="ab")
        assert short.text("ab road") == "ab road"  # < 3 chars: skipped

    def test_names_longest_first_case_insensitive(self):
        r = Redactor(names={"vane": "char#9", "Lord Vane": "char#2"}, home=HOME, user=USER)
        assert r.text("LORD VANE entered") == "char#2 entered"
        assert r.text("whispered by Vane") == "whispered by char#9"
        assert r.text("Vanechild") == "Vanechild"  # whole word only

    def test_session_tokens_stable(self):
        r = Redactor(home=HOME, user=USER)
        a1 = r.text("session=st_chat_Qh8sKd2mZx47pLw9")
        a2 = r.text("session=st_chat_Qh8sKd2mZx47pLw9")
        assert a1 == a2
        assert a1.startswith("session=chat#") and len(a1) == len("session=chat#xxxx")
        b = r.text("session_id=deadbeefcafe1234;")
        assert b.startswith("session_id=chat#")
        assert a1 != b

    def test_ip_rules(self):
        r = Redactor(home=HOME, user=USER)
        assert r.text("203.0.113.9") == "x.x.x.x"
        assert r.text("10.42.7.19") == "10.42.x.x"
        assert r.text("192.168.1.20") == "192.168.x.x"
        assert r.text("172.20.5.6") == "172.20.x.x"
        assert r.text("172.32.5.6") == "x.x.x.x"
        assert r.text("127.0.0.1:5053") == "127.0.0.1:5053"
        assert r.text("http://192.168.1.20:1234/v1") == "http://192.168.x.x:1234/v1"


# --- edge cases ---------------------------------------------------------------------------


class TestEdgeCases:
    def test_empty_input(self):
        r = Redactor(names=NAMES, home=HOME, user=USER)
        assert r.lines([]) == []
        assert r.text("") == ""
        pk = Redactor(names=NAMES, home=HOME, user=USER, keep_content=True)
        assert pk.lines([]) == []

    def test_defaults_do_not_crash(self):
        # No explicit home/user: Path.home()/getpass.getuser() defaults.
        r = Redactor()
        assert r.text("hello world") == "hello world"
        assert r.lines(["2026-10-14 06:57:25 [INFO] x: ok"]) == ["2026-10-14 06:57:25 [INFO] x: ok"]

    def test_unicode_quoted_prose(self):
        r = Redactor(home=HOME, user=USER)
        chinese = "他在冬夜的山顶上低声说，再也不会有人记得这座城市的名字了，" \
                  "只剩下风穿过废墟的声音一直持续到第二天清晨"
        assert len(chinese) > 40
        out = r.text(f'message: "{chinese}" done')
        assert chinese not in out
        assert "<redacted" in out
        # short unicode quotes survive
        assert r.text('他说"晚安"') == '他说"晚安"'

    def test_long_line_performance(self):
        r = Redactor(names=NAMES, home=HOME, user=USER)
        line = (
            "2026-10-14 06:57:25,000 [ERROR] proxy.engine: "
            + ('unquoted filler token and "a quoted run of text that is long enough to be checked" ' * 120)
        )
        assert len(line) > 10_000
        start = time.perf_counter()
        out = r.lines([line])
        elapsed = time.perf_counter() - start
        assert len(out) == 1
        assert "long enough to be checked" not in out[0]
        assert elapsed < 1.0, f"10k-char line took {elapsed:.2f}s"

    def test_whole_test_module_fast(self):
        # The performance brief: everything here completes quickly; this test
        # only asserts a wall-clock budget across a big redaction pass.
        r = Redactor(names=NAMES, home=HOME, user=USER)
        big = ["2026-10-14 06:57:25,000 [INFO] x: " + "token " * 2000 for _ in range(50)]
        start = time.perf_counter()
        assert len(r.lines(big)) == 50
        assert time.perf_counter() - start < 1.0


def test_real_log_gaps_identifiers_long_ids_and_prose_tails():
    """Real-log check 2026-10-14 (synthetic lines in the same shapes)."""
    from spm.redact import Redactor
    r = Redactor(names={"zelda": "char#1"}, home="/home/tester", user="tester")
    out = "\n".join(r.lines([
        "2026-10-14 04:00:37,378 [INFO] scripts.sleep_cycle: [Sleep Cycle] Starting memory "
        "consolidation for character: char#1 (csa_memory_zelda)",
        "2026-10-14 04:00:11,093 [INFO] scripts.sleep_cycle: [Sleep Cycle] Starting memory "
        "consolidation for character: zed_is_a_tall_and_fit_human_with_a_cloak (csa_memory_x)",
        "2026-10-13 15:29:55,390 [INFO] proxy.rag.lore_extractor.LoreExtractionWorker: Skipping "
        "near-duplicate lore rule: If a traveller tries to open the vault, the word 'bluebell' must be said",
    ]))
    assert "zelda" not in out.lower() and "csa_memory_char#1" in out
    assert "tall_and_fit" not in out and "<id>" in out
    assert "bluebell" not in out and "<redacted rest>" in out


def test_user_text_keeps_the_description_but_not_secrets_or_names():
    from spm.redact import Redactor
    r = Redactor(names={"zelda": "char#1"}, home="/home/tester", user="tester")
    out = r.user_text("Sometimes a turn is cut off after two seconds when Zelda replies; "
                      "my key is sk-ABCDEFGHIJKLMNOPQRST and logs are in /home/tester/spm")
    assert "Sometimes a turn is cut off after two seconds" in out
    assert "Zelda" not in out and "char#1" in out
    assert "sk-ABCDEF" not in out and "/home/tester" not in out
