# PythonSignalExecutor – EA Installationsanleitung

## Was macht dieser EA?

Der **PythonSignalExecutor** liest die Signal-CSV-Dateien, die dein Python-System
(`live_trader.py`) ins MT5-Common/Files-Verzeichnis schreibt, und führt die Trades
**tatsächlich im MT5-Terminal** aus.

**Ergebnis:** Alle Trades erscheinen vollständig im MT5-Terminal:

- **Toolbox → Handel:** Offene Positionen mit SL/TP
- **Toolbox → Historie:** Geschlossene Trades mit PnL
- **Toolbox → Orders:** Alle Order-Events
- **Magic Number:** `20260101` (identisch mit Python)
- **SL/TP:** Nativ von MT5 verwaltet (Broker-seitig garantiert!)

---

## Architektur-Überblick

```text
┌──────────────────┐     CSV-Datei        ┌─────────────────────────┐
│ Python            │  ──────────────►     │ MT5 Terminal             │
│ live_trader.py    │  USDJPY_signals.csv  │                         │
│                   │  (Common/Files)      │  PythonSignalExecutor   │
│ Signal-Erzeugung  │                      │  (liest CSV → OrderSend)│
│ Feature-Berechnung│                      │                         │
│ ML-Vorhersage     │                      │  LiveSignalDashboard    │
│ Paper-PnL-Tracking│                      │  (liest CSV → Anzeige)  │
└──────────────────┘                       └─────────────────────────┘
```

**Ablauf:**1. Python erzeugt ML-Signal und schreibt es in `{SYMBOL}_signals.csv`
2. CSV wird per `mirror_csv_to_mt5_common()` nach Common/Files kopiert
3. **PythonSignalExecutor** (EA) liest die CSV alle 5 Sek.
4. Bei neuem Long/Short-Signal → `OrderSend()` → Trade im MT5 sichtbar
5. **LiveSignalDashboard** (Indikator) zeigt weiterhin das Dashboard an

---

## Installation

### Schritt 1: EA-Datei kopieren

Kopiere `PythonSignalExecutor.mq5` in den MT5-Experts-Ordner:

```text
C:\Users\<BENUTZERNAME>\AppData\Roaming\MetaQuotes\Terminal\<TERMINAL-ID>\MQL5\Experts\
```

> **Tipp:** Im MT5: `Datei → Dateiordner öffnen` zeigt den richtigen Pfad.

### Schritt 2: Kompilieren

1. Öffne den **MetaEditor** (F4 in MT5)
2. Öffne `PythonSignalExecutor.mq5`
3. Drücke **F7** (Kompilieren)
4. Es sollte "0 errors, 0 warnings" erscheinen

### Schritt 3: EA auf Chart ziehen

1. Im **Navigator** (Strg+N) unter "Expert Advisors" den EA finden
2. Per Drag&Drop auf einen **USDCAD** oder **USDJPY** Chart ziehen
3. Im Einstellungsdialog:
   - **Symbol:** Leer lassen (nutzt Chart-Symbol) oder explizit setzen
   - **Lot:** `0.01` (Micro-Lot, Standard)
   - **DryRun:** `true` für ersten Test! (loggt nur, handelt nicht)
   - **Magic Number:** `20260101` (Standard, identisch mit Python)

### Schritt 4: Algo-Trading aktivieren

1. **Algo Trading** Button in der MT5-Toolbar anklicken (grünes Symbol)
2. Prüfe: Oben rechts im Chart muss ein "Smiley"-Symbol erscheinen

### Schritt 5: Ersten Test mit DryRun

1. Starte Python `live_trader.py` (Paper-Modus)
2. Warte auf das nächste Signal
3. Im MT5-Journal (Strg+T → Journal) prüfen:
   - `[USDCAD] *** NEUES SIGNAL: Long | Prob=67.0% ...`
   - `[USDCAD] [DRY-RUN] Würde ausführen: Long | Lot=0.01 ...`

### Schritt 6: Live schalten

Wenn DryRun-Test erfolgreich:

1. EA vom Chart entfernen
2. Neu laden mit **DryRun = false**
3. Trades erscheinen jetzt im Terminal!

---

## Einstellungen (Input-Parameter)

| Parameter | Standard | Beschreibung |
| --------- | -------- | ----------- |
| `InpSymbol` | *(leer)* | Symbol (leer = Chart-Symbol) |
| `InpFileSuffix` | `_signals.csv` | CSV-Dateiende |
| `InpUseCommonFiles` | `true` | Common/Files verwenden |
| `InpRefreshSeconds` | `5` | CSV-Abfrage alle N Sekunden |
| `InpLot` | `0.01` | Lot-Größe pro Trade |
| `InpMagicNumber` | `20260101` | Magic Number (= Python) |
| `InpSlippage` | `20` | Max. Slippage in Points |
| `InpDryRun` | `false` | DRY-RUN: nur loggen, nicht handeln |
| `InpCloseOpposite` | `true` | Gegenposition schließen |
| `InpUseCsvPrices` | `true` | SL/TP aus CSV verwenden |
| `InpFallbackSlPct` | `0.003` | Fallback-SL (0.3%) |
| `InpFallbackTpPct` | `0.006` | Fallback-TP (0.6%) |
| `InpMaxStaleMinutes` | `10` | Max. Signal-Alter in Minuten |
| `InpRequireSlTp` | `true` | Trade nur mit SL+TP |

---

## Zusammenspiel mit Python

### Python-Seite (keine Änderung nötig!)

Dein `live_trader.py` muss **NICHT** geändert werden. Es schreibt die CSV bereits
korrekt per `mirror_csv_to_mt5_common()`. Der EA liest genau diese Datei.

### Zwei Modi möglich

#### Modus A: Python Paper-Trading + EA führt aus (empfohlen)

```text
Python:  --paper_trading 1   (loggt und trackt PnL in CSV/SQLite)
EA:      InpDryRun = false   (führt Trade tatsächlich aus → sichtbar in MT5)
```

→ Python behält sein Paper-PnL-Tracking, Trades erscheinen ZUSÄTZLICH in MT5

#### Modus B: Python Live + EA deaktiviert

```text
Python:  --paper_trading 0   (sendet direkt über mt5.order_send())
EA:      Nicht nötig         (Python sendet selbst)
```

### Doppel-Trade-Schutz

Der EA erkennt doppelte Signale anhand des CSV-Zeitstempels.

- Gleicher Zeitstempel + gleiche Richtung → wird übersprungen
- Bereits offene Position in gleicher Richtung → wird übersprungen
- Gegenposition → erst schließen, dann neue öffnen

---

## Troubleshooting

| Problem | Lösung |
| ------- | ------ |
| EA zeigt kein Smiley | Algo-Trading Button aktivieren |
| "CSV nicht gefunden" | Prüfe ob Python läuft und CSV schreibt |
| "Signal veraltet" | `InpMaxStaleMinutes` erhöhen oder Python-Cycle prüfen |
| "Trade ABGEBROCHEN: SL/TP fehlt" | Python-Signal hat kein SL/TP → `InpRequireSlTp` auf `false` oder `InpUseCsvPrices` auf `false` |
| "OrderCheck FEHLGESCHLAGEN" | Broker-Restriktionen prüfen (Margin, Lot-Größe, Handelszeiten) |
| Doppelte Trades | Magic Number zwischen Python und EA muss identisch sein (20260101) |

---

## Sicherheitshinweise

1. **IMMER zuerst mit DryRun=true testen!**
2. **Demo-Konto** verwenden bis alles funktioniert
3. **Stop-Loss ist Pflicht** – der EA verweigert Trades ohne SL (`InpRequireSlTp`)
4. **Nur 0.01 Lot** – erst skalieren wenn System bewiesen
5. **Magic Number nicht ändern** – sonst erkennt Python die Positionen nicht mehr
