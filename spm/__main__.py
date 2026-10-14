"""argparse entry point: `python -m spm start|stop|status|init`."""

from __future__ import annotations

import argparse
import sys

from . import run
from . import db as db_init
from . import setup as setup_mod


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="spm",
        description="Start, stop and check the Sovereign Persona Mesh services.")
    sub = parser.add_subparsers(dest="command", required=True)

    p_start = sub.add_parser("start", help="start the world engine and the proxy")
    p_start.add_argument("--foreground", action="store_true",
                         help="stay attached and stop both on Ctrl+C/SIGTERM "
                              "(this is what a container runs)")

    sub.add_parser("stop", help="stop the services started by `spm start`")
    sub.add_parser("status", help="show PID and health of each service")
    p_init = sub.add_parser("init", help="create and prepare the database, then first-run setup")
    p_init.add_argument("--existing", action="store_true",
                        help="the database already holds SPM data: apply pending "
                             "migrations only, never re-run init_db.sql")
    p_init.add_argument("--no-setup", action="store_true",
                        help="database only: skip the first-run questions")
    p_init.add_argument("--backend-url", default=None,
                        help="model server URL (default http://localhost:13305/v1)")
    p_init.add_argument("--model", default=None, help="chat model id")
    p_init.add_argument("--embedding-url", default=None,
                        help="embedding server URL (default: the model server)")
    p_init.add_argument("--embedding-model", default=None,
                        help="embedding model id, or 'none' (default "
                             "Qwen3-Embedding-0.6B-GGUF-Q8_0)")
    p_init.add_argument("--yes", action="store_true",
                        help="non-interactive: accept the defaults for anything not given "
                             "as a flag (what Docker uses)")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    repo_root = run.default_repo_root()
    if args.command == "start":
        return run.cmd_start(repo_root, foreground=args.foreground)
    if args.command == "stop":
        return run.cmd_stop(repo_root)
    if args.command == "status":
        return run.cmd_status(repo_root)
    if args.command == "init":
        rc = db_init.cmd_init(repo_root, existing=args.existing)
        if rc != 0 or args.no_setup:
            return rc
        return setup_mod.cmd_setup(
            repo_root,
            backend_url=args.backend_url,
            model=args.model,
            embedding_url=args.embedding_url,
            embedding_model=args.embedding_model,
            assume_yes=args.yes)
    return 2  # unreachable: argparse enforces the subcommand


if __name__ == "__main__":
    sys.exit(main())
