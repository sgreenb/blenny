#!/usr/bin/env python3
"""
Blenny CLI Shim.
This allows running Blenny directly from the source tree without installation.
Usage: python blenny.py [ARGS]
"""

import sys
from pathlib import Path

# Add the src directory to sys.path so we can import blenny
src_path = Path(__file__).parent / "src"
if src_path.exists():
    sys.path.insert(0, str(src_path))

try:
    from blenny.cli.main import main
except ImportError as e:
    print(f"Error: Could not find blenny source in {src_path}")
    print(f"Detailed error: {e}")
    sys.exit(1)

if __name__ == "__main__":
    main()
