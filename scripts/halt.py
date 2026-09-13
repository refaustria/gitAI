#!/usr/bin/env python3
"""Operator stop button.

    python scripts/halt.py stop  "why"   # loop stops at its next boundary
    python scripts/halt.py clear         # allow it to run again
    python scripts/halt.py status

The signal lives in control/HALT, deliberately outside every directory the loop
is permitted to write to. This is the only script in the repository that the
loop itself must never invoke.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from gitai.safety import HaltSwitch

SIGNAL = Path(__file__).resolve().parent.parent / "control" / "HALT"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    stop = sub.add_parser("stop", help="request a halt")
    stop.add_argument("reason", nargs="?", default="operator requested halt")
    sub.add_parser("clear", help="clear the halt signal")
    sub.add_parser("status", help="show whether a halt is pending")
    args = parser.parse_args()

    switch = HaltSwitch(SIGNAL)
    if args.command == "stop":
        switch.request(args.reason)
        print(f"halt requested: {args.reason}\nsignal: {SIGNAL}")
    elif args.command == "clear":
        switch.clear()
        print("halt cleared")
    else:
        print(f"HALT PENDING: {switch.reason()}" if switch.requested() else "running (no halt)")


if __name__ == "__main__":
    main()
