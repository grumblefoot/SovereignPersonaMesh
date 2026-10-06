# AGENTS.md — Sovereign Persona Mesh (SPM)

Standing briefing for coding agents (Hermes loads this automatically; Claude Code and others
should read it first).

- **Project guide:** [`playbook.md`](playbook.md) (status, stack, how to run).
  Open issues: [`known_issues.md`](known_issues.md). Plans: [`docs/plans/`](docs/plans/).
- **This machine's AI tools** (Lemonade, Hermes' LLM servers, NPU) live outside this repo:
  `~/Desktop/Playbooks/PLAYBOOK.md`. Don't copy those playbooks into the repo.

## Hard rules

1. **Never touch the live database** `litellm_postgres`. Tests use a throwaway database that is
   rebuilt each run: `SPM_TEST_DB=<unique name> .venv/bin/python -m pytest tests/ -q`.
2. **SPM live and Hermes on Gufo never run together** (memory). Tests are fine either way; a live
   SillyTavern session is not. Details: `playbook.md` (top) and `~/Desktop/Playbooks/hermes_llm_playbook.md`.
3. **Use SPM's own `.venv`** (`.venv/bin/python`), Python 3.13+.
4. Never request the `hermes-coder` model from SPM code; chat models come from Lemonade (:13305).
5. Read only the files your task names; no unrelated refactors. Leave committing to the
   orchestrating agent unless told otherwise.
