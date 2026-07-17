"""Put the proc_track parent on sys.path so `import proc_track` works when
pytest is run from the repo root (users/ dirs aren't a package chain)."""

import sys
from pathlib import Path

# .../users/amoghmpanhale -- parent of the proc_track package
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
