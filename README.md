# Sovereign Persona Mesh (SPM)

The **Sovereign Persona Mesh (SPM)** is an edge-computing, multi-agent roleplay orchestration system designed to eliminate "omniscient character" syndrome. By introducing strict spatial gating, private memory isolation, and deterministic routing, SPM ensures that characters can only perceive what their physical sensory organs can access.

## Architecture Overview

- **SPM FastAPI Router Proxy (Port 5050)**: OpenAI-compatible API emulator for SillyTavern. Intercepts chat completions, executes spatial routing, manages sequential GPU inference queues, and strips inner monologues (`<ctrl94>thoughts`).
- **Evennia World State Engine (Port 4005)**: Headless REST API tracking objective physical ground truth, character coordinates, doors/walls, and acoustic decay.
- **PostgreSQL / pgvector Container (`litellm_postgres`)**: Provides private episodic memory vector search (`csa_memory_{character_id}`) with Game AI decay scoring algorithms.
- **Nightly Sleep Cycle (3:00 AM systemd timer)**: Offline memory consolidation pipeline using Gemma 9B to summarize sensory logs into single-sentence core memory nodes (`is_core_memory=TRUE`).

## Quick Start

### 1. Database Setup
```bash
podman-compose -f config/docker-compose.yml up -d
```

### 2. Run Evennia World State Engine
```bash
python -m evennia_world.app
```

### 3. Run SPM FastAPI Proxy
```bash
python -m proxy.main
```

### 4. Running Unit Tests
```bash
pytest tests/
```

## Hardware Fallback Tiers

- **Sovereign Tier (Default)**: AMD Strix Halo 128GB Unified (Balanced 96GB GTT Profile).
- **Performance Tier**: Discrete GPU VRAM >= 16GB / Host RAM >= 32GB.
- **Experimental Tier**: Low resource 16GB shared memory.
