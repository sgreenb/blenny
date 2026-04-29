"""Tiny CLI placeholder. Real subcommands (run, init, ...) arrive in Step 3."""

from __future__ import annotations

import sys

from blenny import __version__


def app(argv: list[str] | None = None) -> int:
    args = list(sys.argv[1:] if argv is None else argv)
    if args and args[0] in {"-V", "--version", "version"}:
        print(f"blenny {__version__}")
        return 0
    print(f"blenny {__version__} — pre-alpha. CLI coming in Step 3.")
    print("See README.md for the project roadmap.")
    return 0


if __name__ == "__main__":
    raise SystemExit(app())
