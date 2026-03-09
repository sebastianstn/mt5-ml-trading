<!-- markdownlint-disable MD036 MD040 MD060 -->

# Wie lese ich den MT5-Chart grundsätzlich?

Diese Anleitung hilft dir, den Dashboard-Chart fachlich korrekt zu lesen und zwischen normalem Verhalten und Problemfall zu unterscheiden.

---

## 1) Grundprinzip: Was zeigt der Chart wirklich?

Im Chart siehst du **Monitoring-Daten** aus dem Python-System, nicht die vollständige Handelslogik selbst.

- `start_shadow_compare.bat` / `live_trader.py` = erzeugt Signale + schreibt CSV/Logs
- `LiveSignalDashboard.mq5` = liest CSV/Logs und visualisiert Zustand + Entry/SL/TP

Merksatz:

- Engine arbeitet im Hintergrund.
- Dashboard zeigt den Zustand im Chart.

---

## 2) Die wichtigsten Felder im Dashboard-Text

### `STATUS=...` (Gesamtzustand)

- `CONNECTED`: beide Symbole liefern nutzbare Live-Daten
- `PARTIAL`: nur teilweise verfügbar
- `PARTIAL_STALE`: mindestens ein Symbol hat veraltete Daten
- `WAITING_FOR_CSV`: noch keine Datendateien gefunden

### `STALE>... (SignalTF/ChartTF)`

Das ist die Frischegrenze in Minuten.

- `SignalTF` (empfohlen): orientiert sich am Signal-Zeitrahmen (z. B. H1 = 60)
- `ChartTF`: orientiert sich am aktuell geöffneten Chart (z. B. M5 = 5)

Für H1-Signale auf M5-Chart sollte typischerweise stehen:

- `STALE>70min (SignalTF)`

### Pro Symbol (`USDCAD | ...`, `USDJPY | ...`)

- `State=LIVE_SIGNAL`: aktives Signal vorhanden
- `State=LIVE_NO_SIGNAL`: aktuell kein Trade-Signal (normal)
- `State=STALE`: Daten älter als Frischegrenze
- `State=MISSING`: Datei fehlt/ist nicht lesbar
- `Dir=Long/Short/Kein`: Richtung
- `Prob=...`: Modellwahrscheinlichkeit
- `Regime=...`: Marktphase laut Modell
- `Rows=...`: Anzahl gelesener Datenzeilen (soll über Zeit steigen)

### `History-Entries: ...`

Debug-Anzeige, wie viele historische Long/Short-Entries im Chart verwendet werden.

- Wert > 0: Historie vorhanden, Pfeile sollten sichtbar sein
- Wert = 0: keine passenden Entry-Zeilen gefunden (kann in ruhigen Phasen normal sein)

---

## 3) Was soll ich aktiv beobachten?

### [A] Stabilität

1. `STATUS` bleibt überwiegend `CONNECTED`
2. kein dauerhaftes `MISSING`/`STALE`
3. `Rows` wächst weiter

### [B] Plausibilität

1. `Dir` wechselt nicht chaotisch ohne Marktbewegung
2. `Prob` wirkt stabil und nachvollziehbar
3. `Regime` passt grob zum Preisbild (Trend/Seitwärts)

### [C] Visualisierung

1. EMA/RSI Overlay sichtbar (wenn aktiviert)
2. Entry/SL/TP am richtigen Preisniveau
3. History-Pfeile sichtbar, sobald History-Entries vorhanden sind

---

## 4) Wichtige Momente im Tagesverlauf

- direkt nach Start/Restart
- nach neuer H1-Kerze (Signal-Update)
- Sessionwechsel (London/NY)
- nach Deploy oder Versionwechsel

In diesen Fenstern treten Inkonsistenzen am häufigsten auf.

---

## 5) Wie erkenne ich, dass alles korrekt läuft?

Das System läuft erwartungsgemäß, wenn:

1. `CONNECTED` stabil ist
2. `STALE` nur kurzzeitig erscheint und wieder verschwindet
3. `Rows` ansteigt
4. keine dauerhaften Datei-/Sync-Fehler auftreten
5. Visualisierung und Textinformationen konsistent sind

---

## 6) Wann solltest du mich kontaktieren?

Bitte direkt melden (mit Screenshot + Uhrzeit + Symbol), wenn:

- `MISSING` länger als 5 Minuten bleibt
- `PARTIAL_STALE`/`STALE` über mehrere Prüfungen bleibt
- `Rows` nicht mehr steigt
- Dashboard etwas anderes zeigt als Logs/CSV
- nach Deploy die erwartete Version/Funktion nicht aktiv ist
- Entry/SL/TP sichtbar falsch gezeichnet wird

---

## 7) Empfohlene Zusammenarbeit bei Feintuning

Wenn das System stabil läuft, aber du optimieren willst, justieren wir gemeinsam:

- Schwellen (`--schwelle`) pro Symbol
- Regime-Filter
- Anzeige-Dichte (History-Länge, Farben, Marker)
- Alert-Verhalten (zu laut/zu leise)

Vorgehen: Kleine Änderung -> kurzer Lauf -> Wirkung prüfen -> nächste Änderung.

---

## 8) Betriebsregel (dauerhaft)

Single Source of Truth für den Indikator bleibt:

- `live/mt5/LiveSignalDashboard.mq5` im Linux-Repo

Workflow:

1. Änderung im Repo
2. Version erhöhen
3. Deploy auf Laptop
4. in MT5 kompilieren/testen

---

## TEIL 2: Technische Chartanalyse – Kerzen + Indikatoren lesen

Diese Sektion hilft dir, die **Preisaktion** und die **Overlays** (EMA20/EMA50/EMA200 + RSI14) zu verstehen und damit den Dashboard-Status zu verifizieren.

---

## 9) Die Kerze als Grundelement

Jede Kerze zeigt den Preiskampf in einem Zeitrahmen (z. B. H1 = 1 Stunde).

### Anatonie einer Kerze

```
        Oberer Docht (Upper Wick)
              |
        +-----+-----+ <- High (Höchstpreis)
        |           |
        |   Körper  | <- Open bis Close
        |           |
        +-----+-----+ <- Low (Tiefstpreis)
              |
        Unterer Docht (Lower Wick)
```

### Was bedeutet jeder Teil?

| Teil | Bedeutung |
|------|-----------|
| **Körper (grün/rot)** | Käufer vs. Verkäufer im Zeitfenster. Grün=Close>Open (Bullen gewinnen), Rot=Close<Open (Bären gewinnen). |
| **Oberer Docht** | Käufer versuchten, noch höher zu treiben, wurden aber zurückverkauft. = Verkaufsdruck von oben |
| **Unterer Docht** | Verkäufer versuchten, noch tiefer zu gehen, wurden aber hochgekauft. = Kaufdruck von unten |
| **Körpergröße** | Größer = stärkere Überzeugung. Winzig = Unentschlossenheit |

### Praktische Beispiele

**Beispiel 1: Starke Bullische Kerze**

```
Kerze: Open 1.2500, High 1.2520, Close 1.2515, Low 1.2508
→ Großer grüner Körper, kleiner Docht unten = Käufer hatten Kontrolle, kurzzeitig getestet, aber gehalten
→ Signal: BULLISCHE ÜBERZEUGUNG
```

**Beispiel 2: Schwache Kerze (Indecision)**

```
Kerze: Open 1.2510, High 1.2513, Close 1.2511, Low 1.2508
→ Winziger Körper, Dochte oben + unten = Unentschlossen, keine klare Richtung
→ Signal: MARKTUNENTSCHLOSSENHEIT (Seitwärts-Phase)
```

**Beispiel 3: Rejection Kerze (Verkäufer siegen)**

```
Kerze: Open 1.2520, High 1.2522, Close 1.2505, Low 1.2503
→ Großer roter Körper, langer oberer Docht = Käufer versuchten, wurden aber böse abverkauft
→ Signal: VERKAUFSDRUCK, Potenzielle Reversal-Zone
```

---

## 10) EMA (Exponential Moving Average) verstehen

Das System zeichnet **EMA20**, **EMA50** und **EMA200** im Chart – diese zeigen dir **Trend und Momentum**.

### Was ist ein EMA?

Ein EMA ist ein durchschnittlicher Preis der letzten N Kerzen, gewichtet (neueste Kerzen zählen stärker).

- **EMA20** (schnell): Reagiert schnell auf Preisänderungen
- **EMA50** (mittel): Mittleres Momentum
- **EMA200** (langsam): Langfristige Richtung ("Trend")

### EMA-Struktur und Trend-Signale

#### [A] BULLISCHER TREND

```
Visuell im Chart:

EMA20 (blaue Linie, oben)
EMA50 (rote Linie, Mitte)
EMA200 (grüne Linie, unten)
Kerzen über allen EMAs

→ STRUKTUR: EMA20 > EMA50 > EMA200
→ SIGNAL: STARKER AUFWÄRTSTREND
→ Preis lagert über allen EMAs = Käufer dominieren
```

**Praktisches Beispiel:**

```
Beim USDCAD H1:
- Preis bei 1.3650
- EMA20 bei 1.3640
- EMA50 bei 1.3620
- EMA200 bei 1.3580

→ Klare aufsteigende Struktur
→ Bullische Kerzen sollte landen auf/über EMA20
→ Dashboard sollte: Prob=0.65-0.80, Regime=Trend/Bullish
```

#### [B] BEARISCHER TREND

```
Visuell im Chart:

EMA200 (grüne Linie, oben)
EMA50 (rote Linie, Mitte)
EMA20 (blaue Linie, unten)
Kerzen unter allen EMAs

→ STRUKTUR: EMA20 < EMA50 < EMA200
→ SIGNAL: STARKER ABWÄRTSTREND
→ Preis lagert unter allen EMAs = Verkäufer dominieren
```

**Praktisches Beispiel:**

```
Beim USDJPY H1:
- Preis bei 145.20
- EMA20 bei 145.40
- EMA50 bei 145.70
- EMA200 bei 146.10

→ Klare absteigende Struktur (Short-Trend)
→ Bearische Kerzen sollten landen auf/unter EMA20
→ Dashboard sollte: Prob=0.65-0.80, Regime=Trend/Bearish
```

#### [C] SEITWÄRTS / CONFUSION (Problem!)

```
Visuell im Chart:

EMAs durcheinander / kreuzen sich wild
Kerzen schlagartiger Wechsel über/unter

→ STRUKTUR: EMA20 kreuzt EMA50 oder EMA200
→ SIGNAL: ÜBERGANGSZUSTAND / UNKLARHEIT
→ Modell kann hier hohe Fehler haben!
```

**Was zu tun ist:**

- Dashboard zeigt dann: `Regime=Sideways` oder `Prob=0.40-0.55` (niedrig!)
- **Nicht traden in diesem Fall** – System ist unsicher
- Warten auf klare neue Struktur

---

## 11) RSI14 – Momentum und Overbought/Oversold

RSI (Relative Strength Index) misst **Kaufdruck vs. Verkaufsdruck** in einem Zeitfenster (14 Kerzen = Standard).

### RSI-Werte und Interpretation

```
RSI > 70  = OVERBOUGHT (Käufer überanstrengt, Pullback wahrscheinlich)
RSI 50-70 = BULLISCH (stärker mittleres Momentum)
RSI 30-50 = NEUTRAL (Kräfte im Gleichgewicht)
RSI < 30  = OVERSOLD (Verkäufer überanstrengt, Bounce wahrscheinlich)
```

### Praktische Beispiele mit Charts

**Beispiel 1: Bullischer Trend mit hohem RSI**

```
Szenario: USDCAD H1 steigt
- EMA20 > EMA50 > EMA200
- Kerzen grün und über EMA20
- RSI = 72

Interpretation:
→ BULLISCH, aber OVERBOUGHT
→ Kurzfristigu: Pullback zur EMA20 wahrscheinlich
→ Langfristig: Trend weiter intakt
→ Dashboard Prob könnte sinken bei Pullback

Aktion: Beobachten, ob Preis zu EMA20 zurückfällt
```

**Beispiel 2: Bearischer Trend mit niedrigem RSI**

```
Szenario: USDJPY H1 fällt
- EMA20 < EMA50 < EMA200
- Kerzen rot und unter EMA20
- RSI = 28

Interpretation:
→ BEARISCH, aber OVERSOLD
→ Kurzfristig: Bounce/Reversal möglich
→ Langfristig: Bären haben Kontrolle
→ Dashboard könnte Signal drehen, wenn Bounce kommt

Aktion: Schauen, ob RSI > 30 → könnte neuer Short folgen
```

**Beispiel 3: Divergenz (Warnsignal!)**

```
Szenario: Preis erreicht neues High, aber RSI nicht
- Kerze macht neues Hoch (z. B. 1.3700)
- RSI aber sinkt (war 78, ist jetzt 65)

Interpretation:
→ BULLISCHE DIVERGENZ
→ Preis + Momentum divergieren = Kraftverlust
→ Trend könnte bald umschlag

Aktion: Vorsicht vor Short, oder warten auf Bestätigung
```

---

## 12) Integration: Dashboard + Kerzen + Indikatoren

### Workflow beim täglichen Monitoring

**1) Schaue zuerst auf den Dashboard-Text (oben links)**

```
Status=CONNECTED
Regime=Trend_Bullish
Prob=0.72
```

↓

**2) Verifiziere mit dem Chart (Kerzen + EMA)**

Checklist:

- ✅ Kerzen überwiegend grün? OR ❌ wechselhaft?
- ✅ EMA20 > EMA50 > EMA200? OR ❌ durcheinander?
- ✅ RSI > 50? OR ❌ < 50?

**3) Wenn alles passt → System läuft OK**

```
Dashboard sagt: Bullish + Prob 0.72
Chart zeigt: Grüne Kerzen, EMA bullisch, RSI 65
→ KONSISTENT – sehr gut!
```

**4) Wenn etwas nicht passt → Problem!**

```
Dashboard sagt: Bullish + Prob 0.75
Chart zeigt: Rote Kerzen, EMA chaotisch, RSI 22 (Oversold!)
→ INKONSISTENZ – etwas stimmt nicht
→ Mögliche Ursachen:
   - Signalverzögerung (Dashboard liest alte CSV)
   - Markt hat gerade Wendepunkt (Reversal live)
   - Konfiguration falsch (z. B. falscher Regime)
```

---

## 13) Praktische Tagesbeobachtung

### Morgens (Session-Start)

1. **Eröffnungs-Kerze anschauen**
   - Wie groß ist der erste Körper?
   - Größer = Sessionstart-Momentum
   - Winzig = Zögernd

   Beispiel:

   ```
   USDCAD H1 09:00 UTC:
   - High 1.3580, Low 1.3550, Close 1.3575
   → Großer grüner Körper = Bullischer Start
   ```

2. **EMA-Position prüfen**
   - Liegen Kerzen darüber/darunter?
   - Wenn darunter = potenzielle Bounce-Zone

3. **RSI-Startwert**
   - RSI > 60? = Bullisch in den Tag
   - RSI < 40? = Bearisch in den Tag

### Nach wichtigen News / Volatilität

```
Beispiel: Zinsankündigung um 14:30 UTC

VORHER (14:15):
- EMA bullisch, Kerzen stabil über EMA20
- RSI 60, Prob 0.70

SOFORT NACH (14:31):
- Große Candle-Körper, große Dochte
- RSI springt zu 85 (sofort Overbought)
- Prob kann abstürzen auf 0.45

HANDLING:
→ System kann hier verwirrt sein
→ Warten bis Volatilität sinkt (~5-10min)
→ Neue Struktur abwarten
```

### Vor Open einer neuen H1-Kerze

```
Kurz vor 13:00 UTC (H1-Wechsel auf USDCAD):

→ Dashboard aktualisiert sich (neue CSV-Zeile kommt)
→ Prob/Regime können sich ändern
→ Chart kann neue Linie zeichen

Was du siehst SOLLTE:
1. Kerze schließt grün/rot (Entscheidung)
2. EMA-Struktur bleibt oder ändert sich GRADUELL
3. RSI macht neuen Extremwert oder zieht zurück

Was NICHT passieren sollte:
- Chaotische Wechsel ohne Preisbewegung
- RSI springt wild hin-her
- Prob ändert sich um >0.20 ohne Kerzen-Bewegung
```

---

## 14) Häufige Fehler bei der Interpretation

| Fehler | Richtige Interpretation |
|--------|----------------------|
| "RSI 75 = Kauf-Signal" | Falsch! RSI 75 = Overbought. Bullische Divergenz oder Pullback wahrscheinlich. |
| "EMA-Kreuzung = Signalwechsel" | Nicht immer. Oft nur Übergang. Bestätigung durch RSI + Kerzen-Muster abwarten. |
| "Große Kerze = starkes Signal" | Nicht zwingend. Wenn Dochte groß = Indecision im Kampf. Körper ist wichtiger. |
| "Preis über EMA200 = bullisch" | Auch falsch allein. Struktur EMA20>50>200 ist entscheidend, nicht nur Position. |
| "Regime + Prob = allein heilsam" | Nope. Dashboard ist nur Modell. Immer mit Chart verifizieren! |

---

## 15) Troubleshooting: Wenn Chart und Dashboard nicht passen

### Szenario 1: Dashboard zeigt "SIGNAL | Bullish | 0.75", Chart zeigt rote Kerzen

**Ursachen:**

1. **Zeitverzögerung** → CSV ist älter als aktuelle Kerze
   - Fix: Warten 1-2 Kerzen, sollte aktualisieren
2. **Chart zoom falsch** → Schaust auf andere Periode (M5 statt H1)
   - Fix: Auf richtige TF wechseln
3. **Modell ist falsch** → Falsch trainiert auf Historien-Daten
   - Fix: Zu mir kontaktieren + Backtest prüfen

**Was tu tun:** Screenshot + Uhrzeit + beide Symbole → ich prüfe live

### Szenario 2: RSI macht Werte wie 15 → 88 innerhalb 2 Kerzen

**Ursachen:**

1. **News-Spike** → Große Marktbewegung
   - Normal, System ist OK, nur Volatilität hoch
2. **Kerzen-Lücke** (Gap)
   - Auch normal nach Sessionwechsel
3. **Fehlerhafte Daten** → Falsche Preise geladen
   - Selten, aber möglich

**Was zu tun:** Warten bis Volatilität sinkt, nicht panikverkaufen

---

## 16) Checkliste: Tägliches Monitoring

Nutze diese Punkte zur Überprüfung:

- [ ] Dashboard zeigt `CONNECTED`?
- [ ] `Status=STALE` oder Fehler > 5 min vorhanden?
- [ ] Kerzen sichtbar? (Nicht gelb/leer?)
- [ ] EMA Linien sichtbar und glatt?
- [ ] RSI im Subwindow sichtbar?
- [ ] Großer Docht oder unerwarteter Gap seit gestern?
- [ ] Entry/SL/TP Boxen plausibel positioniert?
- [ ] History-Pfeile vorhanden? (sollten wachsen)
- [ ] Prob und Regime konsistent mit Kerzen/EMA?
- [ ] Keine Fehlerausgabe im Indikator-Log?

Wenn alle ✅ → System läuft nominal.
Falls ❌ irgendwo → Screenshot machen + melden!

---

## 17) Zusammenfassung: Dein Monitoring-Rhythmus

| Zeitfenster | Aktion | Fokus |
|---|---|---|
| **Morgens (09:00-10:00)** | Chart öffnen + Dashboard prüfen | Ausgangslage: Bullisch/Bearisch? |
| **Stündlich** | Kurz prüfen: Status OK? Kerzen plausibl? | Drift/Fehler früh erkennen |
| **Nach News** | Direkt prüfen (Volatilität, Kerzen, RSI) | Markt-Shock? System OK? |
| **Endes Session** | Kurz-Review: Wie viele neue Kerzen/Trades? | Summenwerte für Statistik |
| **Nach jedem Update/Deploy** | Sofort testen: neue Version aktiv? | Version-Control |

---

## TEIL 3: Die mathematische Logik – Wie EMA & RSI Entry/Exit-Signale generieren

Diese Sektion erklärt, **WIE** EMA und RSI funktionieren, **WANN** sie Signale erzeugen und **WARUM** das System diese für Entry/Exit-Erkennung nutzt.

---

## 18) EMA – Die Mathematik dahinter

### Wie wird ein EMA berechnet?

Ein EMA ist ein **gewichteter gleitender Durchschnitt**, bei dem neuere Preise stärker zählen.

**Formel vereinfacht:**

```
EMA_heute = Preis_heute × α + EMA_gestern × (1 - α)

Wobei: α = Gewichtsfaktor = 2 / (Periode + 1)
```

**Beispiel für EMA20 (α = 2/21 ≈ 0.095):**

```
Annahme:
- Preis heute: 1.3650
- EMA gestern: 1.3640
- α ≈ 0.095

EMA_heute = 1.3650 × 0.095 + 1.3640 × 0.905
          = 129.77 + 1,234.82
          = 1.3645
```

Das heißt: Der neue EMA ist nur **9.5% der heutigen Preis + 90.5% des gestrigen EMAs**.

### Warum 3 EMAs? (20, 50, 200)

| EMA | Gewichtsfaktor | Reaktionszeit | Nutzen |
|-----|---|---|---|
| **EMA20** | 2/21 ≈ 9.5% | Schnell (~3-5 Kerzen) | Kurzfristiger Trend (Entry-Filter) |
| **EMA50** | 2/51 ≈ 3.9% | Mittel (~10-15 Kerzen) | Mittelfristiger Support/Resistance |
| **EMA200** | 2/201 ≈ 1% | Langsam (~40-60 Kerzen) | Langfristiger Trend (Marktphasen-Definition) |

**Praktisch:**

- EMA20 ändert sich **schnell** bei neuen Kerzen
- EMA50 ändert sich **moderater**
- EMA200 bleibt **stabil** (langfristige Richtung)

### Wann wird ein EMA-Signal erkannt?

Das System erkennt ein **Entry-Signal**, wenn:

```
BULLISCHES SIGNAL:
├─ EMA20 kreuzt EMA50 von unten nach oben (GoldenCross)
├─ UND Preis liegt über EMA200 (Langfristtrend bullisch)
├─ UND RSI > 40 (Momentum steigt)
└─ → LightGBM-Modell gibt Prob > 0.55 → Entry "Long"

BEARISCHES SIGNAL:
├─ EMA20 kreuzt EMA50 von oben nach unten (DeathCross)
├─ UND Preis liegt unter EMA200 (Langfristtrend bearisch)
├─ UND RSI < 60 (Momentum sinkt)
└─ → LightGBM-Modell gibt Prob < 0.45 → Entry "Short"
```

**Zeitablauf - Praktisches Szenario (USDCAD H1):**

```
Kerze 1:
- Preis: 1.3600
- EMA20: 1.3590
- EMA50: 1.3600
- Status: EMA20 < EMA50 (noch keine Aktion)

Kerze 2:
- Preis: 1.3610 (UP)
- EMA20: 1.3595 (folgt langsam)
- EMA50: 1.3598 (noch nicht vollständig angepasst)
- Status: EMA20 steigt, aber noch < EMA50

Kerze 3:
- Preis: 1.3620 (UP)
- EMA20: 1.3602 (JETZT > EMA50!)  ← SIGNAL!
- EMA50: 1.3600
- RSI: 58 (> 40 ✓), Preis > EMA200 ✓
- LightGBM Prob: 0.72
- Status: **ENTRY SIGNAL ERKANNT!**
→ Dashboard: Dir=Long, Prob=0.72
```

### Exit-Erkennung mit EMA

Exit passiert, wenn:

```
BEI LONG-POSITION:
├─ EMA20 kreuzt EMA50 von oben nach unten  (Position umkehren?)
├─ ODER Preis fällt unter EMA50 (Support-Break)
├─ ODER Modell gibt Short-Signal (höhere Konfidenz in Reversal)
└─ → Entry-Preis + ATR-basierter SL berechnett

BEI SHORT-POSITION:
├─ EMA20 kreuzt EMA50 von unten nach oben
├─ ODER Preis steigt über EMA50
├─ ODER Modell gibt Long-Signal
└─ → Exit + Neue Position oder Halt bis Klarheit
```

---

## 19) RSI14 – Die Mathematik dahinter

### Wie wird RSI berechnet?

RSI misst das **Verhältnis von Aufwärts- zu Abwärtsbewegungen** in den letzten 14 Kerzen.

**Formel:**

```
Veränderungen über 14 Kerzen:
- Aufwärts-Änderungen (Gains) = fasse alle positiven Close-Änderungen zusammen
- Abwärts-Änderungen (Losses) = fasse alle negativen Close-Änderungen zusammen (Absolutwert)

Durchschnitt_Gewinne = Summe_Gains / 14
Durchschnitt_Verluste = Summe_Losses / 14

RS (Relative Strength) = Durchschnitt_Gewinne / Durchschnitt_Verluste

RSI = 100 - (100 / (1 + RS))
```

**Praktisches Beispiel (14 Kerzen USDCAD):**

```
Kerzen-Close: 1.3600→1.3605→1.3608→1.3603→1.3610→1.3615→...

Gewinne:    +5, +3, 0, +7, +5, ... = 20 Pips Summe → Ø 1.43 Pips
Verluste:   0, 0, -5, 0, 0, ... = 5 Pips Summe → Ø 0.36 Pips

RS = 1.43 / 0.36 ≈ 3.97
RSI = 100 - (100 / (1 + 3.97)) = 100 - 20 = **80**

→ Interpretation: Starke Aufwärtsbewegung!
```

### RSI-Levels und ihre Bedeutung

```
RSI > 80  = Extrem Overbought (9x mehr Up-Moves als Down-Moves)
RSI 70    = Overbought-Schwelle (klassische Umkehrzone)
RSI 50    = Neutral (gleich viele Up und Down)
RSI 30    = Oversold-Schwelle (klassische Bounce-Zone)
RSI < 20  = Extrem Oversold (9x mehr Down-Moves als Up-Moves)
```

### Wann wird RSI als Entry/Exit-Filter genutzt?

Das System **filtert Signale mithilfe von RSI**:

```
ENTRY-FILTER für LONG:
├─ EMA-Signal sagt: Bullisch
├─ RSI 30-70 (nicht extreme overbought)
├─ RSI > 40 (Momentum steigt)
└─ → Probab > 0.55 bestätigt Long-Entry

ENTRY-FILTER für SHORT:
├─ EMA-Signal sagt: Bearisch
├─ RSI 30-70 (nicht extreme oversold)
├─ RSI < 60 (Momentum sinkt)
└─ → Prob < 0.45 bestätigt Short-Entry
```

### Divergenzen – Die wichtigsten Entry/Exit-Signale

Eine **Divergenz** entsteht, wenn Preis + RSI nicht übereinstimmen.

#### Bullische Divergenz (Kaufsignal)

```
Preis macht neues Tief:      1.3500 → 1.3490 (neuer Low)
RSI aber NICHT:              24 → 28 (nicht niedriger)

Interpretation:
→ Preis fällt, aber Verkaufsdruck nimmt ab
→ Bären verlieren Kraft → Bounce sehr wahrscheinlich
→ LightGBM erkennt dies und erhöht Prob auf 0.65+

Aktion: LONG Entry + Stop unter dem Tief
```

**Live-Szenario:**

```
15:00 UTC: Preis 1.3485, RSI=26
15:30 UTC: Preis 1.3480 (Tief), RSI=28 (höher!)
          → Divergenz erkannt!
15:40 UTC: Dashboard Prob springt von 0.40 → 0.62
           → Dir=Long Signal gesendet
15:45 UTC: Neue Kerze öffnet grün (Bounce!)
          → Entry genommen
```

#### Bearische Divergenz (Verkaufssignal)

```
Preis macht neues Hoch:      1.3700 → 1.3710
RSI aber NICHT:              75 → 72 (sinkt!)

Interpretation:
→ Preis steigt, aber Kaufdruck sinkt
→ Käufer verlieren Kraft → Pullback/Reversal wahrscheinlich
→ LightGBM erkennt und sendet Short-Signal

Aktion: SHORT Entry oder Exit aus Long
```

---

## 20) Praktische Signallogik: Wie das System Entry/Exit generiert

### Step-by-Step Signalgenerierung

Das System folgt **jedem Kerzen-Abschluss** diesem Workflow:

```
1. PREIS AUSLESEN (H1 17:00 Close)
   ├─ Close: 1.3650, High: 1.3655, Low: 1.3645

2. EMA AKTUALISIEREN
   ├─ EMA20 alt: 1.3640 → neu: 1.3643
   ├─ EMA50 alt: 1.3630 → neu: 1.3632
   ├─ EMA200 alt: 1.3600 → neu: 1.3601

3. RSI14 BERECHNEN
   ├─ Letzte 14 Closes analysieren
   ├─ Gewinne/Verluste trennen
   ├─ RSI = 62 ausgeben

4. FEATURE ENGINEERING (LightGBM Input)
   ├─ EMA-Struktur: 1.3643 > 1.3632 > 1.3601 (Trend?)
   ├─ Divergenzen: Preis-RSI Check
   ├─ Volatilität: High-Low Spread
   ├─ Regime: Trend vs. Seitwärts
   └─ ... 50+ weitere Features

5. LIGHTGBM-MODELL
   ├─ Input: 56 Features
   ├─ Output: Prob ∈ [0, 1]
   ├─ Long wenn Prob > 0.55
   ├─ Short wenn Prob < 0.45
   ├─ No-Signal sonst (0.45 ≤ Prob ≤ 0.55)

6. ENTRY/EXIT ENTSCHEIDUNG
   ├─ Wenn Signal ≠ alte Position:
   │  ├─ Neuer Trade
   │  ├─ Entry-Preis = Close
   │  ├─ SL = Close ± 1.5 × ATR14
   │  ├─ TP = Close ± 2.0 × ATR14
   └─ CSVs aktualisieren (`*_signals.csv`, optional `*_closes.csv`)

7. DASHBOARD AKTUALISIEREN
   ├─ "Dir=Long | Prob=0.62 | Regime=Trend"
   └─ MT5 Dashboard zeigt neue Entry/SL/TP Boxen
```

### Timing – Wann passiert was?

```
H1 16:59:00 → Kerze schließt sich
H1 17:00:00 → Python-Engine:
              1. Daten laden
              2. Features kalkulieren
              3. Modell-Vorhersage (< 100ms)
              4. CSV schreiben

H1 17:01:00 → MT5 Dashboard:
              1. CSV einlesen
              2. EMAs/RSI neu zeichnen
              3. Dashboard-Text anpassen
              4. Auf dem Chart sichtbar
```

**Verzögerung: ~1-3 Sekunden von Engine zu Chart**

---

## 21) Entry/Exit-Logik im Detail

### ENTRY-Bedingungen

```
LONG Entry wird ausgelöst wenn:

┌─ Prob > 0.55 UND
├─ EMA20 > EMA50 UND                    (Trend-Bestätigung)
├─ Preis > EMA200 UND                   (Langfristig bullisch)
├─ RSI nicht > 75 (nicht zu overbought) UND
├─ ATR14 > 30 Pips (mind. Volatilität)  (Nicht in Range-Markt)
└─ Regime ≠ Sideways                    (Nicht in Seitwärts)

→ CSV: time | Long | 0.62 | ... | 1.3650 | SL | TP
→ MT5 Dashboard aktualisiert sich sofort
```

```
SHORT Entry wird ausgelöst wenn:

┌─ Prob < 0.45 UND
├─ EMA20 < EMA50 UND                    (Trend-Bestätigung)
├─ Preis < EMA200 UND                   (Langfristig bearisch)
├─ RSI nicht < 25 (nicht zu oversold) UND
├─ ATR14 > 30 Pips UND
└─ Regime ≠ Sideways

→ CSV: time | Short | 0.38 | ...
```

### EXIT-Bedingungen

```
EXIT aus LONG wenn:

┌─ Prob < 0.45 (Modell dreht) ODER
├─ EMA20 fällt unter EMA50 (Trend break) ODER
├─ Preis fällt unter SL (Stop Hit) ODER
├─ RSI > 85 (Extreme Overbought - übertrieben) ODER
└─ 4+ Stunden ohne Bewegung (Timeout)

→ CSV wird aktualisiert mit Exit-Zeit
→ MT5 löscht alte Boxen

EXIT aus SHORT wenn:

Entsprechend in andere Richtung
```

---

## 22) Das Gesamtsystem: Von EMA/RSI zum Trade

### Ablauf eines vollständigen Zyklus

**Beispiel: USDCAD H1 - Von Eröffnung bis Trade**

```
08:00 UTC: Neue H1-Kerze öffnet
├─ Preis: 1.3600 (Open)
├─ EMA20/50/200 von gestern laden
├─ RSI startet bei 45

08:05 UTC: Kerze läuft...
├─ Preis steigt auf 1.3610
├─ EMA20 beginnt zu steigen (langsam, da α=9.5%)
├─ RSI stabil bei 45 (noch zu früh für Umrechnung)

09:00 UTC: Neue H1-Kerze schließt
├─ Close: 1.3615, High: 1.3618, Low: 1.3602
├─ EMA20 wird aktualisiert: 1.3605 (steigt weiter)
├─ EMA50: 1.3600 (EMA50 > EMA20 noch, aber enge)
├─ EMA200: 1.3580 (Preis weit oben)
├─ RSI14: 58 (jetzt auf Basis von 14 Closes mit Aufwärts-Neigung)

Python-Engine prüft (09:00:01):
├─ EMA20 (1.3605) > EMA50 (1.3600)? JA ✓
├─ Preis (1.3615) > EMA200 (1.3580)? JA ✓
├─ RSI 58 im guten Range (40-70)? JA ✓
├─ LightGBM-Modell mit 56 Features → Prob = 0.68 ✓
├─ ENTRY DECISION: Long Entry!
├─ SL = 1.3615 - (1.5 × 15 Pips ATR) = 1.3592
├─ TP = 1.3615 + (2.0 × 15 Pips ATR) = 1.3645

09:00:15: CSV aktualisiert
├─ Neue Zeile: Long | Prob 0.68 | Entry 1.3615 | SL 1.3592 | TP 1.3645

09:00:45: MT5 Dashboard aktualisiert
├─ Liest CSV
├─ Zeichnet Entry-Pfeil um 1.3615
├─ Zeichnet SL-Box + TP-Box
├─ Dashboard-Text: "Dir=Long | P=0.68 | Regime=Trend"

10:00 UTC: Preis bei 1.3630
├─ Trade läuft im Gewinn (~15 Pips)
├─ EMA20 folgt bei ~1.3620 (weiter oben, Trend intact)
├─ RSI 72 (jetzt overbought, aber Trend noch stark)

11:00 UTC: Neue H1-Kerze
├─ Close: 1.3625 (leichte Konsolidierung)
├─ EMA20: 1.3618 (sinkt leicht von Spitze)
├─ EMA50: ebenso
├─ Aber EMA20 > EMA50 noch ✓
├─ LightGBM: Prob sinkt auf 0.56 (immer noch Long)
├─ Keine Exit-Decision

13:00 UTC: WENDEPUNKT
├─ Neue H1-Kerze
├─ Close: 1.3610 (abgesackt!)
├─ EMA20: 1.3612 (folgt nach unten)
├─ EMA50: 1.3613 (EMA20 fällt unter EMA50!) ⚠️
├─ RSI: 42 (sinkt aus overbought)
├─ LightGBM: Prob = 0.48 (unter 0.55, aber noch kein Short)

14:00 UTC: BESTÄTIGUNG
├─ Neue H1-Kerze
├─ Close: 1.3598 (noch tiefer!)
├─ EMA20: 1.3607 < EMA50: 1.3610 ✓ CROSS!
├─ Preis: 1.3598 < EMA50 auch ✓
├─ RSI: 35 (sinkt weiter)
├─ LightGBM: Prob = 0.42 (Short-Signal!) → CLOSEOUT/REVERSAL

Python-Engine EXIT Decision:
├─ Alte Position: Long @ 1.3615
├─ Neue Signal: Short @ 0.42
├─ Close Long Trade bei 1.3598 = -17 Pips Verlust
├─ TP setzt wenn Preis auf 1.3598 fällt → Hit!

14:15 UTC: MT5 Dashboard aktualisiert
├─ Alte Long-Boxen gelöscht
├─ Trade als "geschlossen" markiert
├─ Neue Short Entry-Box Zeichnet sich ab
└─ Dashboard: "Dir=Short | P=0.42 | Regime=Trend"
```

**Aber:** Die tatsächliche **Ausführung** passiert im `live_trader.py` und wird in der CSV protokolliert.

---

## 23) Häufig gestellte Fragen

### Q: Warum gibt EMA20 nicht immer das Signal?

A: EMA20 ist zu schnell und erzeugt zu viele falsche Signale. Das System nutzt:

- **EMA20 als Einstiegs-Filter** (wo liegt der Preis relativ?)
- **EMA50 als Signal-Linie** (Kreuzung = Wendepunkt)
- **EMA200 als Trend-Bestätigung** (muss stimmen, sonst Seitwärts)

### Q: Kann RSI allein Entry/Exit geben?

A: Nein! RSI allein erzeugt zu viele Whipsaw-Signale. System nutzt:

- **RSI als Filter** (nicht traden wenn RSI > 80 oder < 20 = zu extrem)
- **Divergenzen als Warnung** (Divergenz-Pattern)
- **Mit EMA kombiniert** = zuverlässiger

### Q: Was ist ATR14 und warum nutzt das System es?

A: **ATR = Average True Range** = durchschnittliche Preisvolatilität der letzten 14 Kerzen.

```
ATR misst: Wie vip bewegt sich der Preis typischerweise?

ATR14 = 15 Pips auf USDCAD → Volatile Markt
ATR14 = 3 Pips auf USDCAD → Ruhiger Markt

System nutzt ATR für:
- SL-Berechnung: SL = Entry ± 1.5×ATR (dynamisch)
- TP-Berechnung: TP = Entry ± 2.0×ATR
- Filter: "Nicht traden wenn ATR < 25 Pips" (zu ruhig)
```

### Q: Wann ist Regime "Trend" vs. "Sideways"?

A: Das Modell klassifiziert Regime basierend auf:

- **EMA-Struktur** (sind sie hierarchisch sortiert?)
- **Kerzen-Volumen** (große oder kleine Kerzen-Körper?)
- **Volatilität** (ATR hoch oder niedrig?)
- **RSI-Bereich** (zentriert bei 50 oder an Extrem?)

```
Trend = EMA20>50>200 (oder 20<50<200) + konstante Richtung
Sideways = EMAs kreuzen oder chaotisch + RSI bei 50 + volatile Kerzen
```

---

Ende der technischen Deep-Dive Anleitung. Du kennst jetzt die **komplette Logik** von EMA, RSI, Entry- und Exit-Erkennung!
