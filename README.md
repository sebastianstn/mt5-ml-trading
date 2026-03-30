# MT5 ML-Trading-System

Automatisches Trading-System mit LightGBM/XGBoost, Regime-Detection und MT5-Integration.

Aktueller Betriebsmodus: Paper-Trading aktiv für USDCAD und USDJPY mit Two-Stage v4 auf H1 + M15.

## Inhalt

- System-Architektur
- Schnellstart
- Projektstruktur
- Dateiordnung
- Phasenstatus
- Operative Leitplanken
- Nächster Fokus

## System-Architektur

| Gerät | Rolle | Was läuft hier? |
| --- | --- | --- |
| Linux-Server | Training und Auswertung | Feature-Engineering, Labeling, Training, Backtests, Reports, Tests |
| Linux Mint Laptop | Paper-Betrieb | MT5 via Wine, mt5linux/RPyC-Bridge, live_trader.py, Log-Sync |
| Windows 11 Laptop | Alternativer Host | BAT-Startskripte, MT5, Dashboard, Sync |
| VS Code Remote SSH | Entwicklung | Code und Dokumentation auf dem Linux-Server |

Wichtig:

- MetaTrader5 läuft nativ nur unter Windows.
- Auf Linux Mint wird MT5 über Wine betrieben und von Python über mt5linux plus RPyC angesprochen.
- Training, Backtests und Walk-Forward bleiben auf dem Linux-Server.

## Schnellstart

### Linux-Server

```bash
cd /mnt/1Tb-Data/XGBoost-LightGBM
python -m venv .venv
source .venv/bin/activate
pip install -r requirements-server.txt
pytest tests -q
```

### Linux-Mint-Laptop

```bash
cd ~/mt5_trading
python3 -m venv venv
source venv/bin/activate
pip install -r requirements-laptop.txt
bash scripts/start_mt5_rpyc_server.sh
bash start_paper_trading_linux.sh
```

### Windows-Laptop

```powershell
cd C:\Users\<USER>\mt5_trading
python -m venv .venv
.\.venv\Scripts\activate
pip install -r requirements-laptop.txt
start_testphase_topconfig_H1_M15.bat
```

Operativer Einstieg:

- Root-Befehle und Startpfade: BEFEHLE.md
- STALE- und Sync-Notfälle: BEFEHLE_STALE_NOTFALL.md
- Dashboard und MT5-Dateien: live/mt5/
- Server-Laptop-Deploy: reports/deploy_server_to_laptop.md

## Projektstruktur

```text
/mnt/1Tb-Data/XGBoost-LightGBM/
├── docs/                    # Analyse- und Testplan-Dokumente
├── data/                    # Rohdaten, Features, Labels
├── features/                # Feature-Engineering und Labeling
├── models/                  # Modellartefakte
├── backtest/                # Backtests und Ergebnisdateien
├── live/                    # Live-/Paper-Trading, MT5-Integration
├── logs/                    # Laufzeit- und Trainingslogs
├── reports/                 # Operative Reports und Runbooks
├── scripts/                 # Hilfs- und Betriebs-Skripte
├── tests/                   # Testsuite
├── BEFEHLE.md               # Operative Befehle
├── BEFEHLE_STALE_NOTFALL.md # Notfallhilfe
├── README.md                # Überblick
└── Roadmap.md               # Projektphase und nächste Schritte
```

## Dateiordnung

Die Projektdateien sind funktional getrennt:

- Root: Einstieg, Steuerung, Startskripte
- docs/analysis/: langfristige Analysen und Fragebögen
- docs/reference/: Nachschlagewerke und Begriffserklärungen
- docs/testplan/: Testplan-Zusammenfassungen und Präsentationen
- reports/: operative Ausgaben, Runbooks, KPI-Artefakte
- logs/: generierte Logs statt Root-Dateien

Schneller Einstieg für Menschen:

- README.md für Überblick
- Roadmap.md für Prioritäten
- BEFEHLE.md für operative Kommandos
- docs/README.md für eingeordnete Hintergrunddokumente
- docs/reference/NOMENKLATUR.md für Begriffe und Abkürzungen

## Phasenstatus

| Phase | Beschreibung | Status |
| --- | --- | --- |
| 0 | Vorbereitung | abgeschlossen |
| 1 | Umgebung und Datenbeschaffung | abgeschlossen |
| 2 | Feature Engineering | abgeschlossen |
| 3 | Regime Detection | abgeschlossen |
| 4 | Labeling und Modelltraining | abgeschlossen |
| 5 | Backtesting | abgeschlossen |
| 6 | Live-Integration | abgeschlossen |
| 7 | Überwachung und Wartung | aktiv |

Aktiver Fokus:

- Phase 7 Monitoring und Wartung
- Operative Symbole: USDCAD und USDJPY
- Setup: Two-Stage v4, H1-Bias plus M15-Entry, Paper-Modus

## Operative Leitplanken

- Keine Zukunftsdaten in Features
- Zeitliche Splits statt Shuffle
- Test-Set bleibt unangetastet bis zur finalen Bewertung
- Stop-Loss und Kill-Switch sind Pflicht
- Echtgeld erst nach stabilen KPI-Gates und Paper-Nachweis

## Nächster Fokus

1. Paper-Betrieb stabil halten
2. Weekly KPI Report und Daily Dashboard regelmäßig prüfen
3. Log-Sync zwischen Host und Server überwachen
4. Retraining nur datenbasiert und mit Walk-Forward vorbereiten
