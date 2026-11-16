"""Check every tool's copy of dataforseo_queue.py matches the master in shared/.

Usage: python scripts/check_dataforseo_queue.py   (exit code 1 if a copy differs)

Author: Lee Foot
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
MASTER = ROOT / 'shared' / 'dataforseo_queue.py'

master = MASTER.read_bytes()
copies = [p for p in ROOT.rglob('dataforseo_queue.py') if p != MASTER and '.ruff_cache' not in p.parts]
stale = [p for p in copies if p.read_bytes() != master]
for p in copies:
    print(('DIFFERS ' if p in stale else 'ok      ') + str(p.relative_to(ROOT)))
if stale:
    print(f"{len(stale)} copy(ies) differ from shared/dataforseo_queue.py; copy the master over them.")
    sys.exit(1)
print(f"{len(copies)} copy(ies) match the master.")
