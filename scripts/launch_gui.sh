#!/usr/bin/env bash
# POSIX launcher for the Blenny GUI (macOS / Linux).
#
# Mirrors scripts/launch_gui.bat: run from anywhere, the script locates the
# repo root from its own path, then starts the Streamlit server with the best
# available Python environment. Close the window / Ctrl-C to stop the server.
#
#   ./scripts/launch_gui.sh
set -u

# Repo root = parent of this script's directory.
cd "$(dirname "$0")/.." || exit 1

LAUNCHER=""

# 1) Prefer an installed Python that can import blenny.
if command -v python3 >/dev/null 2>&1; then
    if python3 -c "import blenny" >/dev/null 2>&1; then
        LAUNCHER="python3 -m blenny"
    fi
elif command -v python >/dev/null 2>&1; then
    if python -c "import blenny" >/dev/null 2>&1; then
        LAUNCHER="python -m blenny"
    fi
fi

# 2) Fall back to a Poetry environment.
if [ -z "$LAUNCHER" ] && command -v poetry >/dev/null 2>&1; then
    if poetry run python -c "import blenny" >/dev/null 2>&1; then
        LAUNCHER="poetry run blenny"
    fi
fi

if [ -z "$LAUNCHER" ]; then
    echo "Blenny is not installed in any detected Python environment."
    echo "Install it first, e.g.:  poetry install   or   pip install -e ."
    echo
    read -rp "Press Enter to close..." _
    exit 1
fi

echo "Starting Blenny GUI... (close this window to stop the server)"
# shellcheck disable=SC2086  # intentional word-splitting of "poetry run blenny"
exec $LAUNCHER gui
