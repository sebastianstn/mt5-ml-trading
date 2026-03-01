"""Pytest-Konfiguration für Projekt-weite Test-Imports."""

from pathlib import Path
import sys

# Projektwurzel (eine Ebene über /tests) in sys.path einhängen,
# damit Imports wie `from features...` und `from reports...` funktionieren.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
