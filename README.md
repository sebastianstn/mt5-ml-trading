# MT5 ML-Trading-System

Ein automatisches Handelssystem mit **XGBoost** und **LightGBM** + Regime-Detection, das in MetaTrader 5 live handelt.

---

## System-Architektur

| Gerät | Rolle | Was läuft hier? |
| --- | --- | --- |
| **Windows 11 Laptop** | MT5-Host & Live-Trading | MT5 Terminal, `data_loader.py`, `live_trader.py` |
| **Linux Server (1TB SSD)** | Datenspeicher & Training | Rohdaten, Modelle, Training, Backtesting |
| **VS Code Remote SSH** | Entwicklung | Code wird remote auf dem Linux-Server bearbeitet |

---

## Setup – Linux-Server

```bash
# 1. Repository klonen
git clone https://github.com/sebastianstn/mt5-ml-trading.git
cd mt5-ml-trading

# 2. Virtuelle Umgebung erstellen
python -m venv venv
source venv/bin/activate

# 3. Abhängigkeiten installieren
pip install -r requirements-server.txt

# 4. Umgebungsvariablen einrichten
cp .env.example .env
# .env mit echten API-Keys befüllen (niemals in Git einchecken!)
```

## Setup – Windows 11 Laptop

```bash
# 1. Virtuelle Umgebung erstellen
python -m venv venv
venv\Scripts\activate

# 2. Abhängigkeiten installieren (nur für Laptop!)
pip install -r requirements-laptop.txt

# 3. .env-Datei anlegen und befüllen
```

---

## Projektstruktur

```

mt5-ml-trading/
├── .github/
│   └── copilot-instructions.md   # Instruktionen für GitHub Copilot
├── CLAUDE.md                     # Instruktionen für Claude Code
├── ROADMAP.md                    # Projektplan mit allen Phasen
├── data/                         # Rohdaten CSV (nicht in Git)
├── features/                     # Feature-Engineering
│   └── feature_engineering.py
├── models/                       # Trainierte Modelle .pkl (nicht in Git)
├── backtest/                     # Backtesting
│   └── backtest.py
├── live/                         # Live-Trading (→ auf Windows Laptop)
│   └── live_trader.py
├── notebooks/                    # Jupyter Notebooks
├── tests/                        # Unit-Tests
├── requirements-server.txt       # Linux-Server Abhängigkeiten
├── requirements-laptop.txt       # Windows Laptop Abhängigkeiten
├── .env.example                  # API-Key Template
└── README.md                     # Diese Datei
```

---

## Phasenübersicht

| Phase | Beschreibung | Status |
| --- | --- | --- |
| 0 | Vorbereitung | 🔄 In Arbeit |
| 1 | Umgebung & Datenbeschaffung | ⬜ Offen |
| 2 | Feature Engineering | ⬜ Offen |
| 3 | Regime Detection | ⬜ Offen |
| 4 | Labeling & Modelltraining | ⬜ Offen |
| 5 | Backtesting | ⬜ Offen |
| 6 | Live-Integration (MT5) | ⬜ Offen |
| 7 | Überwachung & Wartung | ⬜ Offen |

Details → [ROADMAP.md](ROADMAP.md)

---

## Wichtige Regeln

- **Look-Ahead-Bias:** Features dürfen keine Zukunftsdaten enthalten (`.shift(1)` bei Rolling-Features)
- **Zeitliche Datentrennung:** Training → Validierung → Test (niemals zufällig!)
- **Paper-Trading zuerst:** Niemals Live-Trading ohne vorherigen Paper-Trading-Test
- **Stop-Loss Pflicht:** Niemals ohne Absicherung handeln

---

Das läuft auf dem Linux-Server aus.

Wie entsteht das in Zukunft? Wenn jemand direkt auf GitHub Commits macht (z.B. Dateien bearbeitet oder Workflows hinzufügt) während du lokal arbeitest, divergieren die Branches. Mit kannst du das immer sauber lösen.

```bash
git pull --no-rebase origin main
```

kannst du das immer sauber lösen.

---

Tipp für die Zukunft: Immer zuerst source

```bash
cd /mnt/1T-Data/XGBoost-LightGBM
source venv/bin/activate
```

ausführen, bevor du irgendein Skript in diesem Projekt startest.
