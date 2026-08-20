#!/usr/bin/env bash
# macOS only: copies the Blenny GUI app bundle onto the Desktop so you get a
# proper double-clickable icon (with the blenny artwork) that launches the GUI
# in a Terminal window. Double-click this file in Finder to run it.
set -e

REPO="$(cd "$(dirname "$0")/.." && pwd)"
DEST="$HOME/Desktop/Blenny GUI.app"

if [ ! -d "$REPO/Blenny GUI.app" ]; then
    echo "Could not find 'Blenny GUI.app' in the repository root: $REPO"
    exit 1
fi

rm -rf "$DEST"
cp -R "$REPO/Blenny GUI.app" "$DEST"

# If the repo came from a download/zip, drop the quarantine flag that macOS
# uses to block first launch of unsigned apps.
xattr -dr com.apple.quarantine "$DEST" 2>/dev/null || true

echo "Installed 'Blenny GUI' on the Desktop."
echo "Double-click it to launch the plate reader GUI (close the window to stop)."
