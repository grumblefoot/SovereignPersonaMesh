# SPM troubleshooting

Short answers to the common problems. Run `python -m spm status` first: it shows
whether the two SPM services are up. When something is broken beyond these
notes, see "Filing a report" at the end.

## What SPM is

SPM (Sovereign Persona Mesh) sits between SillyTavern and your LLM server and
gives each character its own knowledge and memories. Four parts:

- **SPM proxy** — port 5050. The OpenAI-compatible endpoint SillyTavern talks
  to. Builds each character's prompt from what *that* character perceived.
- **World engine** — port 4005. Tracks rooms, positions and doors per chat.
- **PostgreSQL + pgvector** — port 5432. Stores memories, perception logs and
  lore rules.
- **Your model server** — any OpenAI-compatible server (Lemonade, LM Studio,
  llama-server). SPM sends chat requests to it and gets replies back.

Logs are in `logs/` (bare metal) or `docker compose logs` (Docker bundle). The
admin dashboard is at `http://localhost:5050/admin`.

## SPM will not start

Common causes:

- **Port in use.** If 5050 or 4005 is taken, start fails. Find the holder with
  `ss -ltnp | grep 5050` (Linux) or `netstat -ano | findstr 5050` (Windows),
  then close it or change `SPM_PORT` / `SPM_ENGINE_PORT` in `.env`.
- **Database password wrong.** `.env` must match the password your PostgreSQL
  container or server was created with. If you recreated the database
  container with `down -v`, the password is set from `.env` again.
- **Never initialised.** Run `python -m spm init` once: it creates the database,
  applies the schema and asks about your model server.
- **An old run is half-alive.** `python -m spm stop`, then `python -m spm start`.

The start command prints the last log lines when a service dies — read them.

## Which settings win?

Settings saved from the admin dashboard win; otherwise SPM uses `.env`
(the environment); otherwise built-in defaults. `config/config.local.json`
holds **only** what the dashboard saved — if you edited `.env` and nothing
changed, a saved setting is overriding it: delete that line from
`config/config.local.json` (in Docker: from the container's `spm_state`
volume) and restart to fall back to `.env`.

## The admin dashboard is unreachable

SPM listens on `127.0.0.1` only, by design: the dashboard has no login. From
the same machine, open `http://localhost:5050/admin`. To reach it from another
device, set `SPM_HOST=0.0.0.0` in `.env` and restart — only on a network you
trust. Inside Docker the container already binds `0.0.0.0` but publishes port 5050
on `127.0.0.1` only. For another device, change the port line in
`config/docker-compose.yml` to `0.0.0.0:5050:5050` and set `SPM_LAN_ACCESS=1` in
`.env`; without it the page loads but every dashboard action answers
`cross-site request refused`. Check the published port with `docker compose ps`.

## "Model server unreachable"

SPM cannot reach the chat model server. Check in order:

1. **Is the server running?** Open its URL in a browser:
   `http://127.0.0.1:13305/v1/models` (adjust the port) should return JSON.
2. **Is the URL right?** The value comes from the Configuration page or
   `BACKEND_LLM_URL` in `.env`. It must end in `/v1`.
3. **LM Studio:** enable "Serve on Local Network", and use the machine's real
   address, not `localhost`, when SPM runs elsewhere (e.g. in Docker).
4. **Docker:** inside the container `localhost` is the container itself. SPM
   rewrites a `localhost`/`127.0.0.1` URL to `host.docker.internal`. If your
   model server listens only on `127.0.0.1` of the host, it is still
   unreachable — make it listen on the network (`--host 0.0.0.0` or the LM
   Studio setting).
5. **Firewall** on the model server machine blocking the port.

## Replies are empty or cut off

Almost always the context window. SPM builds a large prompt (memories, lore,
world state); if the model loads with a small window, the server truncates or
refuses the request and the reply comes back empty or mid-sentence.

- Load the model with a **larger context** (LM Studio slider, llama-server
  `-c 16384`, Lemonade equivalent). The dashboard's Configuration page shows the
  window SPM detected.
- If detection is wrong, set the **context window override** in Configuration.
- SPM trims older memories to fit, but it cannot fit a 32k prompt into a 4k
  window.

## No memories are recalled

Characters answer as if they never met you. Causes:

- **Embeddings set to `none`.** Without embeddings, semantic recall is off.
  The Configuration page and `python -m spm report` show the embedding provider
  and whether the embedding model answers.
- **Embedding server unreachable** — same fixes as "Model server unreachable",
  for `EMBEDDING_URL`.
- **A brand-new chat or a rebuilt database** has no memories yet; they build up
  turn by turn.
- The chat id changed (new SillyTavern chat, missing `X-SPM-Chat-ID` header):
  SPM sees a different session. Re-check the SillyTavern extension setup in
  `docs/st_extension_setup.md`.

## Docker problems

- Start/stop/status/logs: the scripts in `scripts/docker/`
  (`start.cmd` / `start.sh`, `logs.cmd` / `logs.sh`, ...).
- A container stuck or misbehaving: `docker compose --env-file .env -f
  config/docker-compose.yml restart`, and read `... logs --tail 200 spm`. Always pass
  `--env-file .env` (or use the scripts): without it Compose does not read the repo's
  `.env`, and a first `up` creates the database with the default password while SPM
  uses yours.
- **`down -v` DELETES ALL DATA** (memories, chats, settings live on the named
  volumes). Never use it to "just restart" — use `stop`.
- The **Quit** button and the "include full logs" report option are not available in
  Docker: stop the bundle with the scripts, and for a private full-log report run
  `docker compose --env-file .env -f config/docker-compose.yml exec spm python -m spm report
  --full-logs`, then copy the file out with `docker cp`.
- The compose file needs Docker Compose 2.24 or newer (Docker Desktop 4.27+).
- After an update: `git pull` then `update.cmd` / `update.sh` (rebuilds the
  image and runs pending migrations automatically).

## Database errors

- **"vector extension missing".** The PostgreSQL image/server lacks pgvector.
  Use the compose bundle (its image has it), install `pgvector` for your
  PostgreSQL, or have a superuser run `CREATE EXTENSION vector;` in the SPM
  database.
- **"pending migration(s)".** The database is older than the code. Run
  `python -m spm init --existing` (safe: it applies only pending files and
  records them), or the Docker bundle does it on its next start.
- **Cannot connect at all:** check `POSTGRES_HOST/PORT/USER/PASSWORD/DB` in
  `.env`, and that the database container or service is running.

## Filing a report

From the repo root:

```bash
python -m spm report
```

It runs safe, read-only checks, writes a redacted bundle to `reports/`, and
prints a prefilled GitHub issue link. Read the bundle before you attach it:
it is filtered, but you decide what you post. Options:

- `--description "one line about the problem"` (asked interactively otherwise);
- `--system-info` adds CPU/RAM/disk/GPU;
- `--full-logs` includes the raw recent logs including chat text. The bundle is
  then named `spm-report-PRIVATE-...` — send it only to the developers, never
  post it publicly;
- `--yes` prints the issue link without opening a browser and never prompts;
- `--enhanced` asks your model server to analyse the results. If that server is
  not on your machine or LAN, the redacted diagnostics would leave your machine:
  the command asks first, and `--yes` alone is **not** consent — add
  `--confirm-remote` to allow it, otherwise you get the Traditional report.

Reports work even when SPM is completely down — that is exactly when you need
one. Nothing is uploaded automatically.
