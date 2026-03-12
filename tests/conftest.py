"""Pytest-Konfiguration für Projekt-weite Test-Imports."""

from pathlib import Path
import sys

# Projektwurzel (eine Ebene über /tests) in sys.path einhängen,
# damit Imports wie `from features...` und `from reports...` funktionieren.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# live/ in sys.path einhängen, damit die Module untereinander mit
# `import config`, `import mt5_connector` usw. funktionieren.
LIVE_DIR = PROJECT_ROOT / "live"
if str(LIVE_DIR) not in sys.path:
    sys.path.insert(0, str(LIVE_DIR))
