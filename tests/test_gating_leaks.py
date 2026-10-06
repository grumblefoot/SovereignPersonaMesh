"""Sprint 2 exit gate: the gating leak suite (docs/plans/SPRINT_PLAN.md §2).

Planted high-entropy phrases must never appear in the prompt of a character who could
not perceive them. Runs the REAL proxy against the REAL world engine (ASGI-mounted,
no sockets, no live services); only the LLM is faked, and it captures every prompt.

Scenario map (gating.md):
  1 same-room speech reaches the target        8 shout vs closed door: silence
  2 closed door: silence                        9 user 'thought' private everywhere
  3 open door: muffled, never verbatim         10 reply overheard by same-room char
  4 metal partition: silence                   11 reply NOT heard behind closed door
  5 distant room: silence                      12 after a move, old room hears nothing
  6 whisper: bystander never gets the words    13 regenerate: no tick/row duplication
  7 shout at 20 ft: perceived but not verbatim 14 Vardus fixture + planted thought
"""
import itertools
import json
import time
from pathlib import Path

import httpx
import pytest
from starlette.testclient import TestClient

import evennia_world.app as world_app
from evennia_world.hybrid_builder import HybridWorldBuilder
from proxy.main import app as proxy_app

_counter = itertools.count(1)
# Unique per test RUN: session ids used to restart at leak1 every run, so rows from
# earlier runs persisted under the same ids in the test DB. Since seeding is decided
# by "does this session have perception rows yet?", that stale state could silently
# change behaviour between runs (QA 2026-10-05).
import uuid as _uuid
_RUN = _uuid.uuid4().hex[:8]
FIXTURE = Path(__file__).parent / "fixtures" / "test_chat_payload.json"
PH = lambda n: f"zq_leak_{n}_xylophone"


class Rig:
    def __init__(self, client, engine, prompts):
        self.client = client            # proxy TestClient
        self.engine = engine            # ENGINE TestClient (same module state)
        self.prompts = prompts          # captured LLM prompts, one list per call
        self.chat_id = f"leak{_RUN}{next(_counter)}"
        self.history = []               # accumulated user/assistant log (ST resends it)
        self._last_payload = None

    @property
    def session_id(self):
        return f"st_chat_{self.chat_id}"

    def _rows_for_turn(self, turn_id):
        import asyncio
        import asyncpg
        from tests._testdb import TEST_DB_CONFIG

        async def count():
            conn = await asyncpg.connect(**TEST_DB_CONFIG)
            try:
                return await conn.fetchval(
                    "SELECT count(*) FROM spm_perception WHERE session_id=$1 AND turn_id=$2",
                    self.session_id, turn_id)
            finally:
                await conn.close()
        return asyncio.run(count())

    def turn(self, messages, regenerate=False):
        # SillyTavern resends the FULL chat history every request; the proxy derives the
        # turn_id from the user-message count. The rig mirrors that: it accumulates the
        # user/assistant log across turns, so turn 2 is really turn 2 and is not treated
        # as a regenerate of turn 1. ``regenerate=True`` resends the previous payload.
        if regenerate:
            payload = self._last_payload
        else:
            sys_part = [m for m in messages if m.get("role") == "system"]
            new_part = [m for m in messages if m.get("role") != "system"]
            payload = sys_part + self.history + new_part
            self.history = self.history + new_part + [
                {"role": "assistant", "content": "A measured reply."}]
            self._last_payload = payload
        n_users = sum(1 for m in payload if m.get("role") == "user")
        r = self.client.post("/v1/chat/completions",
                             headers={"X-SPM-Chat-ID": self.chat_id},
                             json={"model": "spm-sovereign-mesh", "stream": True,
                                   "messages": payload})
        assert r.status_code == 200, r.text
        # The reply action is recorded by a background task on the proxy's loop. Each
        # extra request lets that loop run; poll until the reply turn's rows exist.
        reply_turn = f"{self.session_id}:{n_users}#reply"
        deadline = time.time() + 3.0
        while time.time() < deadline:
            self.client.get("/health")
            if self._rows_for_turn(reply_turn) >= 1:
                break
            time.sleep(0.05)
        return r

    def prompt_text(self, call_index=-1):
        return "\n".join(m.get("content", "") for m in self.prompts[call_index])

    # ── engine-side world setup (sync, same app_state) ──────────────
    def seed(self, template="dungeon_cellar", placements=()):
        r = self.engine.post("/api/v1/world/configure", json={
            "template_key": template, "session_id": self.session_id,
            "placements": [{"character_id": c, "room_id": room} for c, room in placements]})
        assert r.status_code == 200, r.text

    def barrier(self, a, b, barrier="closed_door", state="closed", distance_ft=15.0,
                template="dungeon_cellar"):
        r = self.engine.post("/api/v1/world/barrier", json={
            "session_id": self.session_id, "template_key": template, "a": a, "b": b,
            "barrier": barrier, "state": state, "distance_ft": distance_ft})
        assert r.status_code == 200, r.text

    def make_room(self, room_id, template="dungeon_cellar"):
        r = self.engine.post("/api/v1/world/rooms", json={
            "room_id": room_id, "room_name": room_id, "description": "test room",
            "template_key": template, "session_id": self.session_id})
        assert r.status_code == 200, r.text


@pytest.fixture
def rig(monkeypatch):
    # The engine app is driven from two loops here (its TestClient + the proxy's
    # ASGITransport); an asyncpg pool can't span loops, so the engine runs memory-only.
    monkeypatch.setenv("SPM_WORLD_DB", "0")
    st = world_app.app_state
    st._db_pool = None
    st.current_world = {}
    st.session_worlds = {}
    st.session_ticks = {}
    st.session_turn_ticks = {}
    st.idempotency_seen = {}
    st.session_edges = {}
    st.mutation_log = {}
    world_app.world_builder = HybridWorldBuilder()
    if not hasattr(world_app.app.state, "start_time"):
        world_app.app.state.start_time = 0.0

    import proxy.api.routes as routes
    real_lore_dispatch = routes._dispatch_lore_extraction
    prompts = []

    async def fake_stream(*args, **kwargs):
        prompts.append([dict(m) for m in kwargs.get("messages", [])])
        yield "<think>plan</think>"
        yield "A measured reply."

    monkeypatch.setattr(routes.lemonade_client, "generate_stream", fake_stream)
    monkeypatch.setattr(routes, "_dispatch_lore_extraction", lambda *a, **k: None)
    routes.evennia_client.base_url = "http://world.test/api/v1"
    routes.evennia_client._client = httpx.AsyncClient(
        transport=httpx.ASGITransport(app=world_app.app), base_url="http://world.test")

    with TestClient(proxy_app) as client, TestClient(world_app.app) as engine:
        r = Rig(client, engine, prompts)
        r.real_lore_dispatch = real_lore_dispatch   # for the side-channel scenarios
        yield r


def sysmsgs(char="Mira", scene="dungeon_cellar"):
    return [
        {"role": "system",
         "content": f"Write {char}'s next reply in a fictional chat between {char} and Tom. [scene:{scene}]"},
        {"role": "system", "content": f"[Character: {char}] A test character."},
    ]


def two_char_setup(rig, a_room="cellar", b_room="tavern_upstairs"):
    """mira with the user in a_room; watcher alone in b_room (default edge: closed door)."""
    rig.seed(placements=[("user", a_room), ("mira", a_room), ("watcher", b_room)])


def watcher_prompt(rig, n_before, room=None):
    """Probe the watcher: walk the user into their room (if needed), ask a neutral
    question, and return the captured prompt. If the planted phrase reached the watcher
    it will surface here, in their gated history. Without the walk, a probe through a
    closed door is (correctly) answered by the proxy's no-perception bypass and no LLM
    call ever happens."""
    if room is not None:
        r = rig.engine.post("/api/v1/world/move", json={
            "character_id": "user", "room_id": room,
            "template_key": "dungeon_cellar", "session_id": rig.session_id})
        assert r.status_code == 200, r.text
    rig.turn(sysmsgs("Watcher") + [{"role": "user", "content": '"Anything new?"'}])
    assert len(rig.prompts) > n_before, "no LLM call captured for the watcher turn"
    return rig.prompt_text(-1)


# ── direct reach and simple occlusion ───────────────────────────────────────

def test_s1_same_room_direct_speech_reaches_target(rig):
    rig.turn(sysmsgs() + [{"role": "user", "content": f'"{PH(1)}"'}])
    assert PH(1) in rig.prompt_text(0)


def test_s2_closed_door_blocks_the_phrase(rig):
    two_char_setup(rig)
    rig.turn(sysmsgs() + [{"role": "user", "content": f'"{PH(2)}"'}])
    text = watcher_prompt(rig, 1, room="tavern_upstairs")
    assert PH(2) not in text


def test_s3_open_door_gives_muffle_never_verbatim(rig):
    two_char_setup(rig)
    rig.barrier("cellar", "tavern_upstairs", barrier="open_door", state="open", distance_ft=10.0)
    rig.turn(sysmsgs() + [{"role": "user", "content": f'"{PH(3)}"'}])
    text = watcher_prompt(rig, 1, room="tavern_upstairs")
    assert PH(3) not in text
    assert "muffled" in text.lower() or "indistinct" in text.lower()


def test_s4_metal_partition_blocks_the_phrase(rig):
    two_char_setup(rig)
    rig.barrier("cellar", "tavern_upstairs", barrier="metal_partition", state="closed",
                distance_ft=10.0)
    rig.turn(sysmsgs() + [{"role": "user", "content": f'"{PH(4)}"'}])
    assert PH(4) not in watcher_prompt(rig, 1, room="tavern_upstairs")


def test_s5_distant_room_blocks_the_phrase(rig):
    rig.seed(placements=[("user", "cellar"), ("mira", "cellar")])
    rig.make_room("far_study")
    rig.barrier("cellar", "far_study", barrier="closed_door", state="closed", distance_ft=45.0)
    rig.engine.post("/api/v1/world/move", json={
        "character_id": "watcher", "room_id": "far_study",
        "template_key": "dungeon_cellar", "session_id": rig.session_id})
    rig.turn(sysmsgs() + [{"role": "user", "content": f'"{PH(5)}"'}])
    assert PH(5) not in watcher_prompt(rig, 1, room="far_study")


# ── whisper and shout ───────────────────────────────────────────────────────

def test_s6_whisper_bystander_never_gets_the_words(rig):
    rig.seed(placements=[("user", "cellar"), ("mira", "cellar"), ("watcher", "cellar")])
    rig.turn(sysmsgs() + [{"role": "user", "content": f'[whisper:Mira] "{PH(6)}"'}])
    assert PH(6) in rig.prompt_text(0)          # the addressed target hears it
    assert PH(6) not in watcher_prompt(rig, 1)  # the bystander never does


def test_s7_shout_at_20ft_is_perceived_but_not_verbatim(rig):
    two_char_setup(rig)
    rig.barrier("cellar", "tavern_upstairs", barrier="open_door", state="open", distance_ft=20.0)
    rig.turn(sysmsgs() + [{"role": "user", "content": f'[shout] "{PH(7)}"'}])
    text = watcher_prompt(rig, 1, room="tavern_upstairs")
    assert PH(7) not in text                    # degraded band: tone, never words
    assert "muffled" in text.lower() or "indistinct" in text.lower()


def test_s8_shout_cannot_pierce_a_closed_door(rig):
    two_char_setup(rig)
    rig.turn(sysmsgs() + [{"role": "user", "content": f'[shout] "{PH(8)}"'}])
    assert PH(8) not in watcher_prompt(rig, 1, room="tavern_upstairs")


# ── privacy of thoughts ─────────────────────────────────────────────────────

def test_s9_thought_absent_from_every_prompt(rig):
    rig.turn(sysmsgs() + [{"role": "user", "content": f'"Hello." \'{PH(9)}\''}])
    assert PH(9) not in rig.prompt_text(0)
    assert "Hello." in rig.prompt_text(0)


# ── replies as world events ─────────────────────────────────────────────────

def test_s10_reply_is_overheard_in_the_same_room(rig):
    rig.seed(placements=[("user", "cellar"), ("mira", "cellar"), ("watcher", "cellar")])
    rig.turn(sysmsgs() + [{"role": "user", "content": '"Tell me everything."'}])
    text = watcher_prompt(rig, 1)
    assert "A measured reply." in text          # mira's reply, perceived by the watcher


def test_s11_reply_not_heard_behind_a_closed_door(rig):
    two_char_setup(rig)
    rig.turn(sysmsgs() + [{"role": "user", "content": '"Tell me everything."'}])
    text = watcher_prompt(rig, 1, room="tavern_upstairs")
    assert "A measured reply." not in text


def test_s12_after_moving_away_old_room_hears_nothing(rig):
    rig.seed(placements=[("user", "cellar"), ("mira", "cellar"), ("watcher", "cellar")])
    rig.make_room("garden")
    rig.barrier("cellar", "garden", barrier="closed_door", state="closed", distance_ft=30.0)
    # user and mira leave; watcher stays in the cellar
    for char in ("user", "mira"):
        rig.engine.post("/api/v1/world/move", json={
            "character_id": char, "room_id": "garden",
            "template_key": "dungeon_cellar", "session_id": rig.session_id})
    rig.turn(sysmsgs() + [{"role": "user", "content": f'"{PH(12)}"'}])
    assert PH(12) not in watcher_prompt(rig, 1, room="cellar")


# ── regeneration semantics ──────────────────────────────────────────────────

def test_s13_regenerate_no_tick_advance_no_duplicate_rows(rig):
    msgs = sysmsgs() + [{"role": "user", "content": '"Once more."'}]
    rig.turn(msgs)
    tick1 = dict(world_app.app_state.session_ticks).get(rig.session_id, 0)
    rig.turn(msgs, regenerate=True)              # resend of the same payload
    tick2 = dict(world_app.app_state.session_ticks).get(rig.session_id, 0)
    assert tick2 == tick1
    n = rig._rows_for_turn(f"{rig.session_id}:1")
    assert n <= 2, f"regenerate duplicated perception rows: {n}"  # mira row (+1 if a second recipient exists)


# ── the real-world fixture ──────────────────────────────────────────────────

def test_s14_vardus_fixture_planted_thought_stays_private(rig):
    payload = json.loads(FIXTURE.read_text())
    msgs = payload["messages"]
    for m in reversed(msgs):
        if m["role"] == "user":
            m["content"] = m["content"] + f" '{PH(14)}'"
            break
    rig.turn(msgs)
    assert PH(14) not in rig.prompt_text(0)


# ── side channels (gating phase 5): background jobs never see private thoughts ──

def _drain_until(rig, pred, what):
    deadline = time.time() + 3.0
    while time.time() < deadline:
        rig.client.get("/health")                # lets the proxy loop run its tasks
        if pred():
            return
        time.sleep(0.05)
    raise AssertionError(f"timed out waiting for {what}")


def test_s15_lore_extractor_never_sees_private_thoughts(rig, monkeypatch):
    import proxy.api.routes as routes
    from proxy.rag.lore_extractor import LoreExtractionWorker
    captured = {}

    async def fake_extract(self, session_id, character_id, context_text, model=""):
        captured["text"] = context_text

    monkeypatch.setattr(LoreExtractionWorker, "extract_initial_rules", fake_extract)
    monkeypatch.setattr(routes, "_dispatch_lore_extraction", rig.real_lore_dispatch)
    rig.turn(sysmsgs() + [{"role": "user", "content": f'"Hello there." \'{PH(15)}\''}])
    _drain_until(rig, lambda: "text" in captured, "lore extraction dispatch")
    assert PH(15) not in captured["text"]        # the private thought never reaches lore
    assert "Hello there." in captured["text"]    # the spoken words still do


def test_s16_bulk_import_rows_never_carry_private_thoughts(rig, monkeypatch):
    from proxy.rag.import_worker import BulkImportWorker
    captured = {}

    async def fake_process(self, session_id, character_id, messages, skip_registration=False):
        captured["messages"] = messages

    async def fake_register(self, session_id, character_id, n):
        return "job-1"

    async def fake_status(self, session_id):
        return None

    monkeypatch.setattr(BulkImportWorker, "process_bulk_import_background", fake_process)
    monkeypatch.setattr(BulkImportWorker, "register_import_job", fake_register)
    monkeypatch.setattr(BulkImportWorker, "check_import_status", fake_status)
    msgs = sysmsgs()
    for i in range(6):                           # >10 messages triggers the import
        msgs += [{"role": "user", "content": f'"line {i}" \'{PH(16)}\''},
                 {"role": "assistant", "content": f"reply {i}"}]
    rig.turn(msgs)
    _drain_until(rig, lambda: "messages" in captured, "bulk import dispatch")
    joined = json.dumps(captured["messages"])
    assert PH(16) not in joined                  # imported memories carry no thoughts
    assert "line 3" in joined                    # the spoken words still import


# ── A2 acceptance: the GM directive is assembled by mode ───────────────────

def _set_gm_mode(mode):
    # Saved settings (config.json) outrank env by design; write the temp config the
    # way the admin UI would. conftest's isolated_settings points this at tmp_path.
    from config.manager import get_settings_manager
    mgr = get_settings_manager()
    mgr.write_settings({**mgr.get_settings(), "gm_actions_mode": mode})


def test_gm_directive_absent_when_mode_off(rig):
    _set_gm_mode("off")
    rig.turn(sysmsgs() + [{"role": "user", "content": '"Hello."'}])
    assert "GM_ACTION" not in rig.prompt_text(0)   # ~90 prompt tokens saved per turn


def test_gm_directive_move_only_hides_create_room(rig):
    _set_gm_mode("move_only")
    rig.turn(sysmsgs() + [{"role": "user", "content": '"Hello."'}])
    text = rig.prompt_text(0)
    assert "CREATE_ROOM" not in text
    assert '"type": "MOVE"' in text


# ── Token budget P0: no request ever exceeds the window ───────────────────

def test_backend_max_tokens_is_clamped_to_window(rig, monkeypatch):
    import proxy.api.routes as routes
    captured = {}
    orig = None

    async def capturing_stream(*args, **kwargs):
        captured["max_tokens"] = kwargs.get("max_tokens")
        yield "<think>plan</think>"
        yield "A measured reply."

    monkeypatch.setattr(routes.lemonade_client, "generate_stream", capturing_stream)
    rig.turn(sysmsgs() + [{"role": "user", "content": '"Hello."'}])
    window = routes.prompt_builder.config.max_context_tokens
    assert captured["max_tokens"] is not None
    assert captured["max_tokens"] <= window          # old code sent a flat 128000
    assert captured["max_tokens"] >= 256


# ── B5: quiet/impersonate generations write nothing ────────────────────────

def test_quiet_generation_persists_nothing(rig):
    msgs = sysmsgs() + [{"role": "user", "content": '"Summarize so far."'}]
    sys_part = [m for m in msgs if m["role"] == "system"]
    r = rig.client.post("/v1/chat/completions",
                        headers={"X-SPM-Chat-ID": rig.chat_id,
                                 "X-SPM-Gen-Type": "quiet"},
                        json={"model": "spm-sovereign-mesh", "stream": True,
                              "messages": msgs})
    assert r.status_code == 200
    assert len(rig.prompts) == 1                        # the LLM still answered
    tick = dict(world_app.app_state.session_ticks).get(rig.session_id, 0)
    assert tick == 0                                    # no world tick
    assert rig._rows_for_turn(f"{rig.session_id}:1") == 0          # no perception rows
    assert rig._rows_for_turn(f"{rig.session_id}:1#reply") == 0    # no reply action


# ── QA A2: the tick counts USER turns; replies don't advance it (decision 10) ──

def test_tick_counts_user_turns_not_replies(rig):
    rig.turn(sysmsgs() + [{"role": "user", "content": '"First."'}])
    rig.turn(sysmsgs() + [{"role": "user", "content": '"Second."'}])
    tick = dict(world_app.app_state.session_ticks).get(rig.session_id, 0)
    assert tick == 2          # was 3: each reply advanced the clock too
    # replies still land, at their own turn's tick
    assert rig._rows_for_turn(f"{rig.session_id}:2#reply") >= 1


# ── QA A4: a branch arrives with history; it must still be seeded ──────────

def test_branch_first_request_with_history_is_seeded_and_placed(rig):
    """The first request SPM sees for a session may carry many user messages
    (a branch, or a chat older than SPM). It must still seed the world and place
    the player and character; it used to skip seeding (user_msg_count > 1)."""
    history = [{"role": "user", "content": '"Earlier line one."'},
               {"role": "assistant", "content": "An earlier reply."},
               {"role": "user", "content": '"Earlier line two."'},
               {"role": "assistant", "content": "Another earlier reply."},
               {"role": "user", "content": '"Now, on the branch."'}]
    r = rig.client.post("/v1/chat/completions",
                        headers={"X-SPM-Chat-ID": rig.chat_id},
                        json={"model": "spm-sovereign-mesh", "stream": True,
                              "messages": sysmsgs() + history})
    assert r.status_code == 200
    snap = rig.engine.get("/api/v1/world/snapshot",
                          params={"session_id": rig.session_id, "template_key": ""}).json()
    where = {o["entity_id"]: o["room_id"] for o in snap["occupants"]}
    assert "user" in where and "mira" in where              # both placed
    assert where["user"] == where["mira"]                    # together, at the seed


def test_quiet_first_turn_does_not_seed_a_world(rig):
    r = rig.client.post("/v1/chat/completions",
                        headers={"X-SPM-Chat-ID": rig.chat_id, "X-SPM-Gen-Type": "quiet"},
                        json={"model": "spm-sovereign-mesh", "stream": True,
                              "messages": sysmsgs() + [{"role": "user", "content": '"Hi."'}]})
    assert r.status_code == 200
    assert rig.session_id not in world_app.app_state.session_worlds or not any(
        room.present_characters
        for w in world_app.app_state.session_worlds[rig.session_id].values()
        for room in w.values())


# ── QA A5: impersonate/quiet get SillyTavern's prompt, not the character pipeline ──

def test_impersonate_is_passed_through_without_character_framing(rig):
    instr = ("[Write your next reply from the point of view of Tom. "
             "Don't write as Mira or system.]")
    msgs = sysmsgs() + [{"role": "user", "content": '"Hello."'},
                        {"role": "assistant", "content": "Hi there."},
                        {"role": "system", "content": instr}]
    r = rig.client.post("/v1/chat/completions",
                        headers={"X-SPM-Chat-ID": rig.chat_id, "X-SPM-Gen-Type": "impersonate"},
                        json={"model": "spm-sovereign-mesh", "stream": True, "messages": msgs})
    assert r.status_code == 200
    sent = rig.prompt_text(-1)
    assert instr in sent                          # SillyTavern's instruction reached the model
    assert "GAME MASTER" not in sent              # no SPM scratchpad directive
    assert "<think>" not in sent                  # no monologue prefill
    assert "A measured reply." in r.text          # the reply came back (reasoning stripped)
    assert "plan" not in r.text                   # the fake model's <think>plan</think> is gone


# ── QA group chat (Mei + her mother): late-joining members ─────────────────

def test_group_member_joining_later_is_placed_and_hears_the_player(rig):
    rig.turn(sysmsgs("Mei") + [{"role": "user", "content": '"Hello, Mei."'}])
    rig.turn(sysmsgs("Lian") + [{"role": "user", "content": '"You must be her mother. zq_join_word"'}])
    snap = rig.engine.get("/api/v1/world/snapshot",
                          params={"session_id": rig.session_id, "template_key": ""}).json()
    where = {o["entity_id"]: o["room_id"] for o in snap["occupants"]}
    assert where.get("lian") == where.get("user")          # placed with the player
    assert "zq_join_word" in rig.prompt_text(-1)           # and she heard the line
    assert "muffled sounds" not in rig.prompt_text(-1)


def test_target_never_inherits_another_characters_blackout(rig):
    """Mei out of earshot must not make a DIFFERENT, unrowed target blacked out."""
    rig.seed(placements=[("user", "cellar"), ("mei", "tavern_upstairs")])
    n = len(rig.prompts)
    r = rig.turn(sysmsgs("Lian") + [{"role": "user", "content": '"Lian, are you there?"'}])
    assert len(rig.prompts) == n + 1                        # Lian's turn reached the LLM
    assert "muffled sounds" not in r.text                   # not bypassed via Mei's row


def test_joining_does_not_teleport_a_character_already_placed(rig):
    rig.seed(placements=[("user", "cellar"), ("mei", "tavern_upstairs")])
    rig.turn(sysmsgs("Mei") + [{"role": "user", "content": '"Mei?"'}])
    snap = rig.engine.get("/api/v1/world/snapshot",
                          params={"session_id": rig.session_id, "template_key": ""}).json()
    assert {o["entity_id"]: o["room_id"] for o in snap["occupants"]}["mei"] == "tavern_upstairs"


# ── QA F11: bulk import must not hand a joining member the whole transcript ──

def _patch_import(monkeypatch):
    from proxy.rag.import_worker import BulkImportWorker
    captured = []

    async def fake_process(self, session_id, character_id, messages, skip_registration=False):
        captured.append((character_id, messages))

    async def fake_register(self, session_id, character_id, n):
        return "job"

    async def fake_status(self, session_id):
        return None
    monkeypatch.setattr(BulkImportWorker, "process_bulk_import_background", fake_process)
    monkeypatch.setattr(BulkImportWorker, "register_import_job", fake_register)
    monkeypatch.setattr(BulkImportWorker, "check_import_status", fake_status)
    return captured


def _long_history(n=6):
    msgs = []
    for i in range(n):
        msgs += [{"role": "user", "content": f'"line {i}"'},
                 {"role": "assistant", "content": f"reply {i}"}]
    return msgs


def test_group_member_joining_a_known_session_gets_no_bulk_import(rig, monkeypatch):
    rig.turn(sysmsgs("Mei") + [{"role": "user", "content": '"Hello."'}])   # session now known
    captured = _patch_import(monkeypatch)
    r = rig.client.post("/v1/chat/completions", headers={"X-SPM-Chat-ID": rig.chat_id},
                        json={"model": "spm-sovereign-mesh", "stream": True,
                              "messages": sysmsgs("Lian") + _long_history()})
    assert r.status_code == 200
    rig.client.get("/health")
    assert captured == []          # was: Lian imported every line, scenes she never saw


def test_bulk_import_never_contains_system_messages(rig, monkeypatch):
    captured = _patch_import(monkeypatch)
    rig.client.post("/v1/chat/completions", headers={"X-SPM-Chat-ID": rig.chat_id},
                    json={"model": "spm-sovereign-mesh", "stream": True,
                          "messages": sysmsgs("Mei") + _long_history()})
    deadline = time.time() + 3
    while not captured and time.time() < deadline:
        rig.client.get("/health"); time.sleep(0.05)
    assert captured, "a pre-SPM chat (new session) should still be imported"
    roles = {m["role"] for _, msgs in captured for m in msgs}
    assert roles <= {"user", "assistant"}    # no cards, persona or instructions


# ── QA F16/F17: group nudge last + bold narration (the live Lian turn) ─────

def test_group_nudge_is_not_heard_as_user_speech_and_bold_narration_is(rig):
    rig.turn(sysmsgs("Mei") + [{"role": "user", "content": '"Hello, Mei."'}])
    msgs = sysmsgs("Lian") + [
        {"role": "user", "content": '"Hello, Mei."'},
        {"role": "assistant", "content": "A measured reply."},
        {"role": "user", "content": "**Not long after Mei leaves, Lian enters the room, a fine tea set in hand.**"},
        {"role": "system", "content": "[Write the next reply only as Lian.]"},   # ST group nudge, LAST
    ]
    r = rig.client.post("/v1/chat/completions", headers={"X-SPM-Chat-ID": rig.chat_id},
                        json={"model": "spm-sovereign-mesh", "stream": True, "messages": msgs})
    assert r.status_code == 200
    prompt = rig.prompt_text(-1)
    # F16: SillyTavern's nudge is not the user's speech
    assert 'User: "[Write the next reply only as Lian.]"' not in prompt
    # F17: **bold** is an out-of-character DIRECTION (common RP convention): the model
    # receives it as an author's direction for this turn ...
    assert "AUTHOR'S DIRECTION" in prompt and "fine tea set in hand" in prompt
    # ... but no character perceives it: it is not in anyone's perception rows.
    import asyncio, asyncpg
    from tests._testdb import TEST_DB_CONFIG

    async def perceived():
        c = await asyncpg.connect(**TEST_DB_CONFIG)
        try:
            return [r["perceived_text"] for r in await c.fetch(
                "SELECT perceived_text FROM spm_perception WHERE session_id = $1", rig.session_id)]
        finally:
            await c.close()
    assert not any("tea set" in (t or "") for t in asyncio.run(perceived()))


def test_ooc_forms_carry_their_text_as_directions():
    from proxy.gating.action_parser import parse_message
    r = parse_message('"Hi." **Make her nervous** ((keep it short)) [OOC: no time skips]')
    assert [a.content for a in r.actions] == ["Hi."]                 # only the speech is in-world
    assert r.ooc_texts == ["Make her nervous", "keep it short", "no time skips"]
