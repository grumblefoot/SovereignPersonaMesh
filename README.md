# Sovereign Persona Mesh (SPM)

The **Sovereign Persona Mesh (SPM)** is an edge-computing, multi-agent roleplay orchestration system designed to eliminate "omniscient character" syndrome. By introducing strict spatial gating, private memory isolation, and deterministic routing, SPM ensures that characters can only perceive what their physical sensory organs can access.

> **Status (2026-10-03):** working end to end after a re-assessment; several design gaps remain open (no authentication, stub embedder, inert-by-default gating). Read [docs/STATE_ASSESSMENT_2026-10-03.md](docs/STATE_ASSESSMENT_2026-10-03.md) and [known_issues.md](known_issues.md) before relying on it.

## Architecture Overview

- **SPM FastAPI Router Proxy (Port 5050)**: OpenAI-compatible API emulator for SillyTavern. Intercepts chat completions, executes spatial routing, and strips inner monologues (`<think>`/`<thinking>`; originally `<ctrl94>`). Talks to **Lemonade Server** on port 13305; `spm-sovereign-mesh` maps to `Gemma-4-26B-A4B-it-GGUF` (`SPM_DEFAULT_MODEL`). A sequential GPU queue exists but is not yet wired in (OPEN-004).
- **World State Engine (Port 4005)**: Headless FastAPI REST service (an in-memory stand-in for Evennia; Evennia itself is not used) tracking objective physical ground truth, character coordinates, doors/walls, and acoustic decay.
- **PostgreSQL / pgvector Container (`spm-postgres`, database `litellm_postgres`)**: Private episodic memory per character (`csa_memory_{character_id}`) with Game AI decay scoring. Note: the embedder is currently a stub, so vector recall is not meaningful yet (OPEN-002).
- **Nightly Sleep Cycle (3:00 AM systemd timer)**: Consolidates volatile logs older than 24 h, per session and day, into single-sentence core memory nodes (`is_core_memory=TRUE`) using `Gemma-4-E4B-it-GGUF`. A failed summary writes and deletes nothing.

## Quick Start

All commands run from this directory with SPM's own virtualenv (`.venv`; Python 3.13+, tested on 3.13 and 3.14). Runtime: `pip install -r requirements.txt` (six packages). Tests: `pip install -r requirements-dev.txt`.

### 1. Database
```bash
podman start spm-postgres            # first time: podman-compose -f config/docker-compose.yml up -d
# new database: psql ... -f scripts/init_db.sql
```

### 2. LLM backend
Lemonade Server on port 13305 (`systemctl --user start lemond`; see `~/Desktop/lemonade_playbook.md`).

### 3. Run the world engine and proxy
```bash
~/.local/bin/start_spm.sh            # or, by hand:
.venv/bin/python -m evennia_world.app
.venv/bin/python -m proxy.main       # SPM_RELOAD=1 for auto-reload while developing
```
Point SillyTavern (Chat Completion → Custom) at `http://localhost:5050/v1`.

### 4. Tests
```bash
.venv/bin/python -m pytest tests/    # rebuilds a throwaway spm_test DB; never touches the live DB or services
```

### 5. Nightly sleep cycle
```bash
./scripts/setup_systemd_timer.sh     # installs spm-sleep-cycle.{service,timer} using .venv
```

## Hardware Fallback Tiers

- **Sovereign Tier (Default)**: AMD Strix Halo 128GB Unified (Balanced 96GB GTT Profile).
- **Performance Tier**: Discrete GPU VRAM >= 16GB / Host RAM >= 32GB.
- **Experimental Tier**: Low resource 16GB shared memory.

## License & Credits

This project is open-source software made freely available under the terms of the [MIT License](LICENSE). 

For details on our AI-assisted development workflow and acknowledgments, please see [CREDITS.md](CREDITS.md).

## Known Issues & Backlog

See [known_issues.md](known_issues.md) for the active defect log, session tracking issues, dashboard UI fixes, and upcoming feature roadmap.
