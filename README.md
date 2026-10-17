# Sovereign Persona Mesh (SPM)

SPM is a proxy that sits between [SillyTavern](https://github.com/SillyTavern/SillyTavern) and a
local LLM server and stops characters from being omniscient. Each character only knows what it
could actually perceive: a world engine tracks where everyone is, conversations are gated by
distance and walls, and every character keeps its own private memories.

## How it works

- **SPM proxy (port 5050):** an OpenAI-compatible endpoint for SillyTavern. For each turn it
  works out who perceived what, builds the character's prompt from *their* history and
  memories (not the raw transcript), strips the model's private reasoning from the reply, and
  applies world changes the model proposes (new rooms, characters moving), after validation.
- **World engine (port 4005):** a small FastAPI service holding rooms, positions, doors and
  walls for each chat.
- **PostgreSQL + pgvector (port 5432):** per-character memories, perception logs and lore rules.
- **Your LLM server:** any OpenAI-compatible server. Developed against
  [Lemonade Server](https://github.com/lemonade-sdk/lemonade) on port 13305.
- **Admin dashboard:** `http://localhost:5050/admin` (memories, world map, lore rules,
  settings).

## Requirements

- Python 3.13 or newer.
- Podman or Docker, for PostgreSQL with pgvector.
- An OpenAI-compatible LLM server with:
  - a chat model; the default is `Gemma-4-26B-A4B-it-GGUF` (set `SPM_DEFAULT_MODEL` to change it);
  - an embedding model; the default is `Qwen3-Embedding-0.6B-GGUF-Q8_0`.

  The server can be on this machine, on another machine in your network, or a hosted API (see
  [Using a hosted model server](#using-a-hosted-model-server)).
- SillyTavern 1.19 or newer.

### Using a hosted model server

No hardware for a local model? Point SPM at any hosted OpenAI-compatible API in `.env`:

```
BACKEND_LLM_URL=https://<provider>/v1
BACKEND_API_KEY=<your key>
SPM_DEFAULT_MODEL=<the provider's model id>
```

- The key is sent only to that server, and the dashboard never shows it.
- **Embeddings** are requested from the same server unless you set `EMBEDDING_URL`. If your
  provider has no embedding endpoint, run a small local one (the default embedding model runs on
  a CPU, e.g. with Lemonade or `llama-server --embedding`) and set `EMBEDDING_URL`, or set
  `EMBEDDING_PROVIDER=none` to recall memories by text search only.
- **Privacy and cost:** every turn sends the prompt (chat, character card, recalled memories) to
  the provider. Besides the reply, SPM makes a short GM call per turn and occasional background
  calls (rule extraction, memory consolidation, imported-chat backfill), all billed by the
  provider. Reasoning mode "auto" stays off for hosted servers (hidden thinking is billed too).
- SPM cannot read a hosted model's context window; it assumes 32,768 tokens. Set the context
  window override in the dashboard if your model's is smaller or larger.

## Setup

### 1. Python environment

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt        # add requirements-dev.txt to run the tests
cp .env.example .env                             # then set the database password (python -m spm init asks the rest)
```

### 2. Database

SPM needs PostgreSQL 16 with the pgvector extension. Either run it in a container (only the
database; SPM itself runs from step 3):

```bash
docker compose --env-file .env -f config/docker-compose.db.yml up -d   # or: podman compose ...
```

It listens on `127.0.0.1:5432` and uses the `POSTGRES_*` values from your `.env`. Add
`SPM_DB_CONTAINER=spm-postgres` to `.env` and `python -m spm start` / `stop` will start and
stop it with SPM. Then run `python -m spm init` (step 3) to create the tables.

(To run everything in Docker instead, skip steps 2 and 3 and see "Run with Docker" below.)

**Without Docker** (PostgreSQL already installed on your machine):

1. Install PostgreSQL 16 with the pgvector extension (e.g. `postgresql16-server` plus
   `pgvector_16` on Fedora/RHEL, `postgresql-16-pgvector` on Debian/Ubuntu).
2. Create a user that may create databases:
   ```sql
   CREATE ROLE spm_user LOGIN PASSWORD 'change-me' CREATEDB;
   ```
3. Make sure the user can run `CREATE EXTENSION vector` in the SPM database — either the
   user owns the database / has the needed rights, or a superuser runs
   `CREATE EXTENSION vector;` once inside it.
4. Set `POSTGRES_HOST`, `POSTGRES_PORT`, `POSTGRES_USER`, `POSTGRES_PASSWORD` and
   `POSTGRES_DB` in `.env`, then:
   ```bash
   python -m spm init          # creates the database and the schema
   python -m spm init --existing   # an existing SPM database: apply migrations only
   ```

After the database step, `python -m spm init` continues with first-run setup: it asks for
your model server URL, chat model, embedding server and embedding model, probes them
(lists the models, warns if your chat model is not there, checks `/embeddings`), writes the
answers to the git-ignored `config/config.local.json` (the tracked `config/config.json`
stays untouched) and pins the chat model in `.env` as `SPM_DEFAULT_MODEL`. Use
`--backend-url`, `--model`, `--embedding-url`, `--embedding-model` (or `none`) and `--yes`
to answer non-interactively, or `--no-setup` for the database only.

**Which settings win?** Settings saved from the admin dashboard win; otherwise SPM uses your
`.env` (environment); otherwise built-in defaults. `config/config.local.json` holds **only**
what the dashboard (or `spm init`) saved — delete a line there to fall back to `.env` for that
setting. The tracked `config/config.json` carries no env-backed keys, so editing `BACKEND_LLM_URL`
or `EMBEDDING_MODEL` in `.env` takes effect until you save that setting on the dashboard.

### 3. Start SPM

Start your LLM server first, then:

```bash
python -m spm start      # world engine (port 4005) + SPM proxy (port 5050)
python -m spm status     # PID + health of each service
python -m spm stop       # stop both
```

(`.venv/bin/python` on Linux/macOS, `.venv\Scripts\python` on Windows; run it from the repo
root.) `start` detaches both services, waits for their health endpoints, writes their PIDs to
`run/spm.pids` and appends their output to `logs/`. `start --foreground` stays attached and
stops both on Ctrl+C — that is what a container runs.

Or, by hand (note: the proxy reads `.env` itself, the world engine does not — export the
`POSTGRES_*` values first or use `spm start`, which passes `.env` to both):

```bash
.venv/bin/python -m evennia_world.app &   # world engine, port 4005
.venv/bin/python -m proxy.main            # SPM proxy, port 5050
```

Both services listen on this machine only (127.0.0.1) and refuse requests that other web pages
open in your browser try to send them. To reach SPM from another device on your
network (for example SillyTavern on a laptop), set `SPM_HOST=0.0.0.0` (and
`SPM_ENGINE_HOST=0.0.0.0` only if the engine must be reachable too) — but the admin dashboard has
no login, so do this only on a trusted network. Containers set both to `0.0.0.0` themselves but
publish port 5050 on 127.0.0.1 only; to use the Docker bundle from another device, publish
`0.0.0.0:5050:5050` in `config/docker-compose.yml` and set `SPM_LAN_ACCESS=1` in `.env`.

### 4. Connect SillyTavern

1. Copy `st_extension/SPM-Chat-ID/` into `<SillyTavern>/data/<user>/extensions/` and reload the
   SillyTavern tab.
2. In **API Connections → Chat Completion → Custom (OpenAI-compatible)**, set the URL to
   `http://localhost:5050/v1`.
3. In **Additional Headers** ("Include Headers"), paste:

   ```yaml
   X-SPM-Chat-ID: "{{spmChatId}}"
   X-SPM-Parent-Chat: "{{spmParentChat}}"
   X-SPM-Gen-Type: "{{spmGenType}}"
   X-SPM-User-Turns: "{{spmUserTurns}}"
   ```

4. Optional: turn off SillyTavern's **Summarize** extension. SPM keeps its own memories and
   already strips SillyTavern's Summary and Author's Note blocks from character prompts (they
   can't be filtered per character), so Summarize only costs extra model time.

Details and troubleshooting: [docs/st_extension_setup.md](docs/st_extension_setup.md).

## Run with Docker

The compose bundle runs the whole of SPM itself (database plus the proxy and world engine).
Ready-made scripts live in `scripts/docker/` — double-click the `.cmd` files on Windows, run the
`.sh` files on Linux/macOS. Each one finds `config/docker-compose.yml` on its own, so they work
from any directory:

| Do this            | Windows              | Linux/macOS              | Raw compose command                                    |
|--------------------|----------------------|--------------------------|--------------------------------------------------------|
| Start              | `start.cmd`          | `scripts/docker/start.sh` | `docker compose --env-file .env -f config/docker-compose.yml up -d`     |
| Stop               | `stop.cmd`           | `scripts/docker/stop.sh`  | `docker compose --env-file .env -f config/docker-compose.yml stop`      |
| See what is running| `status.cmd`         | `scripts/docker/status.sh`| `docker compose --env-file .env -f config/docker-compose.yml ps`        |
| Follow the logs    | `logs.cmd`           | `scripts/docker/logs.sh`  | `docker compose --env-file .env -f config/docker-compose.yml logs -f --tail 100 spm` |
| Update to latest   | `update.cmd`         | `scripts/docker/update.sh`| `git pull && docker compose --env-file .env -f config/docker-compose.yml up -d --build` |

Stopping keeps all your data (memories, chats, settings live on named volumes), and Docker
Desktop's Stop/Start buttons on the "spm" group do the same thing as the scripts. The admin
dashboard's **Quit** button also explains how to stop the bundle.

> **Warning:** `docker compose -f config/docker-compose.yml down -v` **deletes all memories,
> chats and settings** (the `-v` removes the data volumes). Use it only to start over from
> scratch.

- The model server must listen on the **host**, not inside the container: Lemonade, LM Studio
  with "Serve on Local Network", or `llama-server --host 0.0.0.0`.
- If it is not reachable at `host.docker.internal:13305`, set `BACKEND_LLM_URL` in `.env`
  (e.g. `BACKEND_LLM_URL=http://192.168.1.10:1234/v1`). A `localhost` or `127.0.0.1` URL in
  `.env` (for the model server or `EMBEDDING_URL`) is rewritten to `host.docker.internal`
  inside the container, so the same `.env` works with and without Docker.
- Podman works too (`podman compose ...`, and the `.sh` scripts pick it up automatically); the
  healthcheck is ignored there unless images are built in Docker format.
- On first boot the container runs `spm init` to create and prepare the database; settings
  saved from the admin dashboard persist on a named volume across rebuilds.
- Always pass `--env-file .env` to raw `docker compose` commands (the scripts do): Compose
  reads `config/.env`, not the repo's `.env`, for the `${POSTGRES_*}` values, so without it the
  database is created with the default password while SPM uses yours. Needs Compose 2.24+.
- The dashboard's Quit button and the "full logs" report option do not work inside Docker;
  use the stop scripts, and `docker compose ... exec spm python -m spm report --full-logs`.
- Admin dashboard: `http://127.0.0.1:5050/admin`
- SillyTavern URL: `http://127.0.0.1:5050/v1`

## Writing conventions

SPM reads your messages with these conventions (they are hints; anything else is treated as
speech):

| You write | SPM treats it as |
|---|---|
| `"Hello there"` | speech, heard by whoever is close enough |
| `*walks to the kitchen*` | an action; movement updates the world |
| `'I don't trust him'` | a private thought: never heard by anyone |
| `**Lian enters with tea**`, `((...))`, `[OOC: ...]` | a direction to the AI, not part of the scene |
| `[whisper:Mai] ...`, `[move:kitchen]`, `[shout] ...` | explicit tags, which win over everything else |
| a block of `Name: / Age: / ...` lines | your character sheet: kept as setup, never spoken |

Scenario cards (one card voicing a whole world, e.g. "... RPG World") are detected from the card
name and treated as a narrator that follows you. You can force this on or off per chat in the
admin dashboard.

## Tests

```bash
.venv/bin/python -m pytest tests/ -q
```

The suite builds a throwaway `spm_test` database in the same PostgreSQL server and never
touches your live data or running services.

## Hardware tiers

`SPM_HARDWARE_TIER` sets the context window and defaults:

- **SOVEREIGN** (default): large unified-memory machines (e.g. 128 GB).
- **PERFORMANCE:** a discrete GPU with at least 16 GB VRAM and 32 GB RAM.
- **EXPERIMENTAL:** about 16 GB of shared memory.

## Known issues

See [known_issues.md](known_issues.md).

## Reporting a problem

Common fixes are in [docs/troubleshooting.md](docs/troubleshooting.md). To file
a bug: `python -m spm report` runs safe read-only checks, writes a redacted
bundle to `reports/` and prints a prefilled GitHub issue link. Read the bundle
before attaching it — nothing is uploaded automatically. `--enhanced` adds an
analysis by your own model server; if that server is not local, the command asks
first (`--confirm-remote` allows it non-interactively; `--yes` alone does not).

## License and credits

MIT License: see [LICENSE](LICENSE). Credits: [CREDITS.md](CREDITS.md).
