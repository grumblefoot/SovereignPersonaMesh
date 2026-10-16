"""argparse entry point: `python -m spm start|stop|status|init|report`."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from . import run
from . import db as db_init
from . import report as report_mod
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
                        help="the database may already hold SPM data: an empty database "
                             "gets the full schema, an SPM database only its pending "
                             "migrations (init_db.sql is never re-run on one)")
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

    p_report = sub.add_parser("report", help="write a bug report bundle (reports/) and "
                                             "print a prefilled GitHub issue link")
    p_report.add_argument("--description", default="",
                          help="one line about what went wrong (asked interactively "
                               "when omitted and stdin is a terminal)")
    p_report.add_argument("--system-info", action="store_true",
                          help="include CPU / RAM / disk / GPU information")
    p_report.add_argument("--full-logs", action="store_true",
                          help="include the full recent logs (PRIVATE: chat text; "
                               "send only to the developers, never post publicly)")
    p_report.add_argument("--yes", action="store_true",
                          help="non-interactive: do not open the issue URL in a "
                               "browser and never prompt (this is NOT consent to a "
                               "remote model server; see --confirm-remote)")
    p_report.add_argument("--enhanced", action="store_true",
                          help="model-powered analysis of the checks (sends the "
                               "redacted diagnostics to the configured model "
                               "server; a remote server asks for confirmation)")
    p_report.add_argument("--confirm-remote", action="store_true",
                          help="with --enhanced: allow sending the redacted "
                               "diagnostics to a REMOTE model server (not loopback, "
                               "LAN or a .local/.lan host); without it, a remote "
                               "server gets an interactive [y/N] prompt, or with "
                               "--yes the Traditional report instead")
    return parser


def main(argv: list[str] | None = None, *, repo_root: Path | None = None,
         enhanced_client=None, input_fn=None) -> int:
    args = build_parser().parse_args(argv)
    if repo_root is None:
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
    if args.command == "report":
        return report_mod.cmd_report(
            repo_root,
            description=args.description,
            system_info=args.system_info,
            full_logs=args.full_logs,
            yes=args.yes,
            enhanced=args.enhanced,
            confirm_remote=args.confirm_remote,
            enhanced_client=enhanced_client,
            input_fn=input_fn)
    return 2  # unreachable: argparse enforces the subcommand


if __name__ == "__main__":
    sys.exit(main())
