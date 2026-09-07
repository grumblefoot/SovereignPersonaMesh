#!/usr/bin/env python3
"""Migrate <ctrl94>/</ctrl94> tags to <thinking>/</thinking> across the codebase.

Replacements performed (in order, per file):
  1.  Regex: r'<ctrl94\b'  -> r'<think\b'             (tag-start fragments in raw-strings)
  2.  Regex: r'<ctrl94|'   -> r'<think|'              (wrapped form like <|ctrl94|>)
  3.  Regex: r'</ctrl94>?' -> r'</think\b>?'           (optional closing tag in regex)
  4.  Literal: '</ctrl94>' -> '</thinking>'
  5.  Literal: '<ctrl94>'  -> '<thinking>'
  6.  Literal: 'ctrl94'    -> 'think'                  (bare occurrences, safe because steps 4-5 consume tagged forms first)

Files processed:
  proxy/core/stream_parser.py
  proxy/rag/prompt_builder.py
  proxy/api/routes.py
  proxy/ui/index.html
  tests/*.py  (all Python files in the tests/ directory)

The script skips itself (migrate_tags.py) to avoid mutating the migration tool.
"""

import pathlib
import re
import sys

ROOT = pathlib.Path(__file__).resolve().parent

# Explicit file list
EXPLICIT_FILES = [
    ROOT / "proxy" / "core" / "stream_parser.py",
    ROOT / "proxy" / "rag" / "prompt_builder.py",
    ROOT / "proxy" / "api" / "routes.py",
    ROOT / "proxy" / "ui" / "index.html",
]

# All .py files under tests/ (skip __pycache__, migrate_tags.py)
TESTS_DIR = ROOT / "tests"
TEST_FILES = sorted(
    p for p in TESTS_DIR.glob("*.py")
    if p.name not in ("__init__.py", "conftest.py")
)

TARGET_FILES = EXPLICIT_FILES + TEST_FILES

# Regex replacements (handle patterns inside raw strings first).
# These are applied before the literal replacements so that, e.g.,
# r'</ctrl94>?' becomes r'</think\b>?' rather than breaking apart.
REPLACE_REGEXES = [
    # </ctrl94>?   ->   </think\b>?
    (re.compile(r"</ctrl94>\?"), r"</think\\b>?"),
    # <ctrl94\b    ->   <think\b
    (re.compile(r"<ctrl94\b"), r"<think\\b"),
    # <ctrl94|     ->   <think|
    (re.compile(r"<ctrl94\|"), r"<think|"),
]

# Literal replacements (safe to apply everywhere, including HTML / string literals).
# Order matters: longer/more-specific patterns first.
REPLACE_LITERALS = [
    ("</ctrl94>", "</thinking>"),
    ("<ctrl94>", "<thinking>"),
    ("ctrl94", "think"),
]


def migrate_file(filepath: pathlib.Path) -> list[str]:
    """Apply all replacements to *filepath* in-place. Returns a list of
    description strings summarising what was changed."""
    original = filepath.read_text(encoding="utf-8")
    changed = original

    # Apply regex replacements first (these handle regex-string contexts).
    for pattern, repl in REPLACE_REGEXES:
        new_text = pattern.sub(repl, changed)
        if new_text != changed:
            changed = new_text

    # Apply literal replacements next.
    for old, new in REPLACE_LITERALS:
        new_text = changed.replace(old, new)
        if new_text != changed:
            changed = new_text

    messages: list[str] = []
    if changed != original:
        filepath.write_text(changed, encoding="utf-8")
        messages.append(f"  OK  {filepath.relative_to(ROOT)}")

    return messages


def main() -> int:
    print("Starting tag migration...")
    total = 0
    modified = 0

    for path in TARGET_FILES:
        if not path.exists():
            print(f"  SKIP (missing) {path.relative_to(ROOT)}")
            continue
        msgs = migrate_file(path)
        total += 1
        if msgs:
            modified += 1
            print(f"[{total}] {path.relative_to(ROOT)}")
            for msg in msgs:
                print(msg)

    print(f"\nDone. {modified}/{total} files modified.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
