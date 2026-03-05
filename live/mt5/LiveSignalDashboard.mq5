// ╔══════════════════════════════════════════════════════════════╗
// ║   LiveSignalDashboard.mq5 – MT5 Indikator v2.22            ║
// ║                                                              ║
// ║   ZWECK:                                                     ║
// ║   Dieses Programm ist ein MT5-Chart-Indikator, der die       ║
// ║   Trading-Signale unseres Python-Systems (live_trader.py)    ║
// ║   visuell im Chart darstellt.                                ║
// ║                                                              ║
// ║   WIE ES FUNKTIONIERT (Gesamtbild):                         ║
// ║   1. Python (live_trader.py) laeuft auf dem Windows-Laptop   ║
// ║      und schreibt alle 5 Min eine CSV-Datei ins MT5-         ║
// ║      Common/Files-Verzeichnis (z.B. USDJPY_live_trades.csv) ║
// ║   2. Dieser Indikator liest diese CSV-Datei alle 5 Sekunden ║
// ║   3. Er zeigt die Daten als:                                 ║
// ║      - Dashboard-Textbox (oben links)                        ║
// ║      - Ampel-System (unten links: Gruen/Gelb/Rot)           ║
// ║      - Trade-Pfeile + SL/TP-Zonen im Chart                 ║
// ║      - EMA-Hilfslinien (blau/rot/gruen)                     ║
// ║      - EMA-Struktur-Label (oben rechts)                     ║
// ║                                                              ║
// ║   DATEIPFAD DER CSV:                                         ║
// ║   C:\Users\...\AppData\Roaming\MetaQuotes\Terminal\          ║
// ║   Common\Files\USDJPY_live_trades.csv                       ║
// ║                                                              ║
// ║   WICHTIG:                                                   ║
// ║   - Indikator laeuft NUR im Chart-Fenster (kein eigenes)    ║
// ║   - Timer-basiert: alle 5 Sek. wird alles neu gelesen       ║
// ║   - Braucht KEINE Ticks – funktioniert auch am Wochenende   ║
// ╚══════════════════════════════════════════════════════════════╝
#property strict                    // Strenger Compiler-Modus (faengt mehr Fehler ab)
#property version "2.21"            // Versionsnummer – erhoehe bei jeder Aenderung
#property description "MT5 Dashboard fuer Python Live-Signale (USDCAD/USDJPY)"
#property description "Liest CSV aus Common/Files und zeigt Status, Alerts + Chart-Zeichnungen."
#property description "v2.21: +Countdown, Spread, ATR/Vola, Session, Regime, Modus"
#property indicator_chart_window    // Zeigt den Indikator IM Chart (nicht in eigenem Subfenster)


// ============================================================
// 1. EINGABE-PARAMETER (Input-Variablen)
// ============================================================
// Diese Werte kann der Nutzer beim Hinzufuegen des Indikators
// im MT5-Einstellungsdialog aendern. "input" bedeutet:
// Der Wert wird EINMAL beim Laden gesetzt und bleibt dann fest.
// Um ihn zu aendern, muss man den Indikator entfernen + neu laden.

// --- Grundeinstellungen ---
input string InpSymbol1 = "USDCAD";        // Erstes Waehrungspaar das wir ueberwachen
input string InpSymbol2 = "USDJPY";        // Zweites Waehrungspaar das wir ueberwachen
input string InpFileSuffix = "_live_trades.csv"; // Dateiendung – Symbol + dies = Dateiname
                                                  // z.B. "USDJPY" + "_live_trades.csv" = "USDJPY_live_trades.csv"
input bool InpUseCommonFiles = true;        // true = lese aus MT5 Common/Files (alle Terminals teilen sich diesen Ordner)
                                            // false = lese aus MQL5/Files (nur dieses Terminal)
input int InpRefreshSeconds = 5;            // Wie oft (in Sekunden) die CSV neu gelesen wird
                                            // 5 = alle 5 Sekunden aktualisieren
input bool InpEnableAlerts = true;          // true = Popup-Alert wenn ein neues Long/Short-Signal kommt

// --- Freshness / "Wie alt duerfen die Daten sein?" ---
// Wenn die CSV-Daten aelter als X Minuten sind, gelten sie als "STALE" (veraltet).
// Bei H1-Signalen: Die CSV wird nur 1x pro Stunde aktualisiert, daher ist 70 min sinnvoll.
// Freshness-Schwelle in Minuten:
// H1: 70 (empfohlen), M30: 40, M15: 25, M5: 15
// Hintergrund: Bei H1 kommt ein neues Update typischerweise nur pro neuer Kerze.
input int InpStaleMinutes = 70;             // Fester Stale-Schwellenwert (Fallback)
input bool InpAutoStaleByTimeframe = true;  // true = Schwelle automatisch aus Zeitrahmen berechnen
                                            // (ueberschreibt InpStaleMinutes)
input bool InpUseSignalTimeframeForStale = true; // true = nutze den Signal-Zeitrahmen (H1=60 Min)
                                                  // Dies ist wichtig wenn du einen M5-Chart hast
                                                  // aber H1-Signale empfaengst!
input int InpSignalTimeframeMinutes = 60;   // Signal-Zeitrahmen in Minuten (H1=60, M30=30, M15=15, M5=5)
input int InpStaleBufferMinutes = 10;       // Zusaetzlicher Puffer: 60 + 10 = 70 min Grenze
input int InpMissingFileLogEverySec = 300;  // Wenn CSV fehlt: alle 300 Sek (5 Min) eine Warnung ins Log schreiben

// --- Chart-Zeichnungen ---
input bool InpDrawTrades = true;            // Soll der Indikator Entry-Pfeile, SL-/TP-Linien zeichnen?
input bool InpDrawEntryHistory = true;      // Soll er auch VERGANGENE Signale als kleine Pfeile einzeichnen?
input int InpMaxTradesOnChart = 10;         // Maximal so viele historische Trade-Marker anzeigen
input bool InpDrawTechnicalOverlay = true;  // EMA20/EMA50/EMA200 Linien + RSI14 zum Chart hinzufuegen?

// --- Dashboard-Aussehen ---
input bool InpDebugHistoryInfo = true;      // Zeigt "Hist: USDCAD=0 | USDJPY=0" im Dashboard
input bool InpUseLargeDashboardText = true; // true = grosses Label (gut lesbar), false = kleiner Comment-Text
input int InpDashboardFontSize = 10;        // Schriftgroesse des Dashboards (Standard: 10, groesser = besser lesbar)
input color InpDashboardTextColor = clrWhite; // Textfarbe des Dashboards
input bool InpDashboardTextBackground = true; // Schwarzer Hintergrund hinter dem Dashboard?
input color InpDashboardBgColor = clrBlack;   // Hintergrundfarbe der Dashboard-Box
input int InpDashboardBgAlpha = 140;          // Transparenz: 0=durchsichtig, 255=vollstaendig deckend
input bool InpDashboardCompactMode = true;    // true = kurze Zeilen (empfohlen), false = ausfuehrlich

// --- Farben fuer Trade-Zeichnungen ---
input color InpColorLong = clrDodgerBlue;   // Farbe fuer Long-Trades (Kauf) = Blau
input color InpColorShort = clrOrangeRed;   // Farbe fuer Short-Trades (Verkauf) = Orange-Rot
input color InpColorSL = clrRed;            // Farbe fuer Stop-Loss-Linien = Rot
input color InpColorTP = clrLimeGreen;      // Farbe fuer Take-Profit-Linien = Gruen

// --- EMA-Hilfslinien (zusaetzlich zu den normalen EMA-Kurven) ---
input bool InpDrawEmaGuides = true;              // Zusaetzliche horizontale Linien bei aktuellem EMA-Wert
input color InpEma20GuideColor = clrDodgerBlue;  // EMA20-Linie: Blau (schnell, kurzfristig)
input color InpEma50GuideColor = clrRed;         // EMA50-Linie: Rot (mittel)
input color InpEma200GuideColor = clrLimeGreen;  // EMA200-Linie: Gruen (langsam, langfristig)
input int InpEmaGuideWidth = 2;                  // Dicke der EMA-Hilfslinien (1=duenn, 3=dick)
input bool InpShowEmaStructureLabel = true;      // Text oben rechts: "BULL: EMA20>EMA50>EMA200"

// --- Ampel-System ---
input bool InpShowAmpel = true;                  // Ampel-Box unten links anzeigen?
input int InpAmpelFontSize = 13;                 // Schriftgroesse der Ampel (11-15 empfohlen)


// ============================================================
// 2. DATENSTRUKTUREN (Structs)
// ============================================================
// In MQL5 sind "structs" Datenpakete – wie ein Formular mit
// mehreren Feldern. Wir packen alle Infos ueber ein Symbol
// in eine Struktur, damit wir sie leicht herumreichen koennen.

// SignalSnapshot = Ein "Foto" des aktuellen Zustands eines Symbols.
// Wird alle 5 Sekunden aus der CSV-Datei neu befuellt.
struct SignalSnapshot
{
    string symbol;         // z.B. "USDJPY"
    datetime ts;           // Zeitstempel der letzten CSV-Zeile (wann wurde sie geschrieben?)
    string richtung;       // "Long", "Short" oder "Kein" (Handelsrichtung)
    double prob;           // Wahrscheinlichkeit des Signals (0.0 bis 1.0, z.B. 0.37 = 37%)
    int regime;            // Markt-Regime als Zahl: 0=Seitwaerts, 1=Aufwaerts, 2=Abwaerts
    string regime_name;    // Regime als Text: "Seitwärts", "Aufwärtstrend" etc.
    bool paper;            // true = Paper-Trading (kein echtes Geld), false = Live
    string modus;          // "PAPER" oder "LIVE"
    long rows;             // Anzahl Datenzeilen in der CSV (waechst mit jeder neuen Kerze)
    bool valid;            // true = CSV wurde erfolgreich gelesen, false = Fehler/nicht gefunden
    double entry_price;    // Einstiegspreis (nur bei aktivem Signal, sonst 0)
    double sl_price;       // Stop-Loss Preis (nur bei aktivem Signal, sonst 0)
    double tp_price;       // Take-Profit Preis (nur bei aktivem Signal, sonst 0)
};

// EntryPoint = Ein einzelner historischer Trade-Eintrag.
// Wird genutzt, um vergangene Signale als kleine Pfeile im Chart zu zeigen.
struct EntryPoint
{
    datetime ts;           // Wann wurde das Signal ausgeloest?
    string richtung;       // "Long" oder "Short"
    double prob;           // Wie sicher war das Signal?
    string regime_name;    // In welchem Markt-Regime war es?
    double entry_price;    // Zu welchem Preis?
};


// ============================================================
// 3. GLOBALE VARIABLEN
// ============================================================
// Globale Variablen behalten ihren Wert solange der Indikator laeuft.
// Sie werden in OnInit() initialisiert und in OnTimer()/RefreshAll() aktualisiert.

SignalSnapshot g_snap1;         // Aktueller Zustand von Symbol 1 (USDCAD)
SignalSnapshot g_snap2;         // Aktueller Zustand von Symbol 2 (USDJPY)
datetime g_last_alert_ts_1 = 0; // Zeitstempel des letzten Alerts fuer Symbol 1
                                 // (verhindert Doppel-Alerts fuer dasselbe Signal)
datetime g_last_alert_ts_2 = 0; // Zeitstempel des letzten Alerts fuer Symbol 2
int g_refresh_seconds = 5;      // Tatsaechliches Refresh-Intervall (= InpRefreshSeconds, min. 1)
bool g_missing_1 = false;       // true = CSV-Datei fuer Symbol 1 fehlt aktuell
bool g_missing_2 = false;       // true = CSV-Datei fuer Symbol 2 fehlt aktuell
datetime g_missing_log_ts_1 = 0; // Wann haben wir zuletzt "Datei fehlt" geloggt? (Anti-Spam)
datetime g_missing_log_ts_2 = 0;

// Indikator-Handles: Das sind "Referenzen" auf technische Indikatoren.
// MT5 berechnet EMAs, RSI, ATR im Hintergrund. Wir holen die Werte ueber
// diese Handles ab. INVALID_HANDLE = noch nicht erstellt.
int g_h_ema20 = INVALID_HANDLE;   // Handle fuer EMA mit Periode 20 (schnell, reagiert schnell auf Preisaenderungen)
int g_h_ema50 = INVALID_HANDLE;   // Handle fuer EMA mit Periode 50 (mittel)
int g_h_ema200 = INVALID_HANDLE;  // Handle fuer EMA mit Periode 200 (langsam, zeigt langfristigen Trend)
int g_h_rsi14 = INVALID_HANDLE;   // Handle fuer RSI mit Periode 14 (Momentum-Indikator, 0-100)
int g_h_atr14 = INVALID_HANDLE;   // Handle fuer ATR mit Periode 14 auf H1 (misst Volatilitaet in Pips)
int g_hist_count_1 = 0;           // Anzahl historischer Trade-Signale fuer Symbol 1
int g_hist_count_2 = 0;           // Anzahl historischer Trade-Signale fuer Symbol 2

// ============================================================
// 4. STALE-BERECHNUNG (Sind die Daten noch frisch?)
// ============================================================
// "Stale" = veraltet. Wenn die CSV-Daten zu alt sind, warnt uns der
// Indikator. Die Grenze haengt vom Signal-Zeitrahmen ab:
// - H1-Signal: CSV darf max. 70 Min alt sein (60 + 10 Buffer)
// - M5-Signal: CSV darf max. 15 Min alt sein (5 + 10 Buffer)

// StaleBaseMinutes() – Ermittelt den Basis-Zeitrahmen in Minuten.
// Entweder aus dem Signal-TF (empfohlen) oder dem aktuellen Chart-TF.
int StaleBaseMinutes()
{
    // Option 1: Signal-TF nutzen (empfohlen wenn H1-Signale auf M5-Chart laufen)
    if (InpUseSignalTimeframeForStale)
    {
        if (InpSignalTimeframeMinutes > 0)
            return InpSignalTimeframeMinutes;  // z.B. 60 fuer H1
        return 60;  // Fallback: 60 Minuten (H1)
    }

    // Option 2: Chart-TF nutzen (z.B. M5 = 5 Minuten)
    int sec = PeriodSeconds(PERIOD_CURRENT);  // PERIOD_CURRENT = Zeitrahmen des aktuellen Charts
    if (sec <= 0)
        return 60;

    int tf_minutes = sec / 60;  // Sekunden → Minuten umrechnen
    if (tf_minutes <= 0)
        return 60;

    return tf_minutes;
}

// EffectiveStaleMinutes() – Die tatsaechliche Stale-Grenze (Basis + Buffer).
// Beispiel: H1 (60) + Buffer (10) = 70 Minuten. Erst danach zeigt das Dashboard "STALE".
int EffectiveStaleMinutes()
{
    // Manuelle Einstellung verwenden wenn Auto-Modus deaktiviert
    if (!InpAutoStaleByTimeframe)
        return InpStaleMinutes;

    int base_minutes = StaleBaseMinutes();
    if (base_minutes <= 0)
        return InpStaleMinutes;

    // Buffer addieren: Basis + Puffer, aber mindestens 5 Minuten
    int buffer = MathMax(InpStaleBufferMinutes, 0);
    return MathMax(5, base_minutes + buffer);
}


// ============================================================
// 5. TECHNISCHES OVERLAY (EMA + RSI + ATR auf den Chart legen)
// ============================================================
// Diese Funktion erstellt die technischen Indikatoren (EMA20/50/200,
// RSI14, ATR14) und fuegt sie dem Chart hinzu.
// iMA() = Moving Average erstellen, iRSI() = RSI erstellen, iATR() = ATR erstellen
// ChartIndicatorAdd() = den Indikator sichtbar auf den Chart legen
// Subfenster 0 = Hauptchart, Subfenster 1 = erstes Unterfenster (fuer RSI)

void SetupTechnicalOverlay()
{
    // Wenn technisches Overlay deaktiviert ist, nichts tun
    if (!InpDrawTechnicalOverlay)
        return;

    // EMA-Indikatoren erstellen: iMA(Symbol, Zeitrahmen, Periode, Versatz, Methode, Preis)
    // PERIOD_CURRENT = der Zeitrahmen vom aktuellen Chart (z.B. M5, H1)
    // MODE_EMA = Exponential Moving Average (gewichtet neuere Daten staerker)
    // PRICE_CLOSE = berechnet auf Schlusskursen
    g_h_ema20 = iMA(Symbol(), PERIOD_CURRENT, 20, 0, MODE_EMA, PRICE_CLOSE);   // Kurzfristiger Trend
    g_h_ema50 = iMA(Symbol(), PERIOD_CURRENT, 50, 0, MODE_EMA, PRICE_CLOSE);   // Mittelfristiger Trend
    g_h_ema200 = iMA(Symbol(), PERIOD_CURRENT, 200, 0, MODE_EMA, PRICE_CLOSE); // Langfristiger Trend

    // RSI = Relative Strength Index (Ueberkauft/Ueberverkauft-Indikator)
    g_h_rsi14 = iRSI(Symbol(), PERIOD_CURRENT, 14, PRICE_CLOSE);

    // ATR = Average True Range (durchschnittliche Schwankungsbreite pro Kerze)
    // Immer auf H1 berechnet, unabhaengig vom Chart-Zeitrahmen
    g_h_atr14 = iATR(Symbol(), PERIOD_H1, 14);

    // Indikatoren auf den Chart legen (Subfenster 0 = Hauptchart, 1 = erstes Unterfenster)
    if (g_h_ema20 != INVALID_HANDLE)
        ChartIndicatorAdd(0, 0, g_h_ema20);   // EMA20 ins Hauptchart
    else
        Print("[Dashboard] Warnung: EMA20 konnte nicht erstellt werden.");

    if (g_h_ema50 != INVALID_HANDLE)
        ChartIndicatorAdd(0, 0, g_h_ema50);   // EMA50 ins Hauptchart
    else
        Print("[Dashboard] Warnung: EMA50 konnte nicht erstellt werden.");

    if (g_h_ema200 != INVALID_HANDLE)
        ChartIndicatorAdd(0, 0, g_h_ema200);  // EMA200 ins Hauptchart
    else
        Print("[Dashboard] Warnung: EMA200 konnte nicht erstellt werden.");

    if (g_h_rsi14 != INVALID_HANDLE)
        ChartIndicatorAdd(0, 1, g_h_rsi14);   // RSI ins Unterfenster 1
    else
        Print("[Dashboard] Warnung: RSI14 konnte nicht erstellt werden.");

    // ATR wird nicht aufs Chart gelegt – nur fuer Dashboard-Anzeige genutzt
    if (g_h_atr14 == INVALID_HANDLE)
        Print("[Dashboard] Warnung: ATR14 konnte nicht erstellt werden.");
}

// ReleaseTechnicalOverlay() – Gibt alle Indikator-Handles frei.
// Wird in OnDeinit() aufgerufen wenn das Dashboard entfernt wird.
// IndicatorRelease() gibt den Speicher frei und entfernt den Indikator.
void ReleaseTechnicalOverlay()
{
    // Jeden Handle pruefen und freigeben wenn gueltig
    if (g_h_ema20 != INVALID_HANDLE)
        IndicatorRelease(g_h_ema20);
    if (g_h_ema50 != INVALID_HANDLE)
        IndicatorRelease(g_h_ema50);
    if (g_h_ema200 != INVALID_HANDLE)
        IndicatorRelease(g_h_ema200);
    if (g_h_rsi14 != INVALID_HANDLE)
        IndicatorRelease(g_h_rsi14);
    if (g_h_atr14 != INVALID_HANDLE)
        IndicatorRelease(g_h_atr14);

    // Handles zuruecksetzen auf "nicht erstellt"
    g_h_ema20 = INVALID_HANDLE;
    g_h_ema50 = INVALID_HANDLE;
    g_h_ema200 = INVALID_HANDLE;
    g_h_rsi14 = INVALID_HANDLE;
    g_h_atr14 = INVALID_HANDLE;
}

// ============================================================
// 6. CSV-LESE-INFRASTRUKTUR
// ============================================================
// Die Kommunikation mit Python erfolgt ueber CSV-Dateien.
// Python (live_trader.py) schreibt Signale in:
//   C:\Users\...\AppData\Roaming\MetaQuotes\Terminal\Common\Files\SYMBOL_live_trades.csv
// Dieses MQL5-Skript liest die CSV alle 5 Sekunden aus und
// zeigt die Daten visuell auf dem Chart an.
//
// CSV-Format (Beispiel-Header):
//   time,richtung,prob,regime,regime_name,paper_trading,modus,entry_price,sl_price,tp_price
//   2025.06.26 14:00,Long,0.67,1,Trending,true,PAPER,1.38450,1.38200,1.38900

// FindHeaderIndex() – Findet den Spalten-Index eines Headers in der CSV.
// Beispiel: FindHeaderIndex(headers, "prob") → 2  (wenn prob die 3. Spalte ist)
// Case-insensitive Suche ("Prob" == "prob" == "PROB")
int FindHeaderIndex(string &headers[], const string key)
{
    string key_l = key;
    StringToLower(key_l);  // Suchbegriff in Kleinbuchstaben

    for (int i = 0; i < ArraySize(headers); i++)
    {
        string header_l = headers[i];
        StringToLower(header_l);  // Spaltenname in Kleinbuchstaben

        if (StringCompare(header_l, key_l) == 0)
            return i;  // Spaltenindex gefunden
    }
    return -1;  // Spalte nicht gefunden
}

// ReadCsvLine() – Liest eine komplette Zeile aus der CSV-Datei.
// Jedes Komma-getrennte Feld wird in das Array "fields" geschrieben.
// Beispiel: "2025.06.26,Long,0.67" → fields[0]="2025.06.26", fields[1]="Long", fields[2]="0.67"
void ReadCsvLine(const int handle, string &fields[])
{
    ArrayResize(fields, 0);  // Array leeren
    if (FileIsEnding(handle))  // Dateiende erreicht? → Nichts zu lesen
        return;

    // Felder einzeln lesen bis Zeilenende oder Dateiende
    while (true)
    {
        string value = FileReadString(handle);  // Naechstes Komma-getrenntes Feld lesen
        int n = ArraySize(fields);
        ArrayResize(fields, n + 1);  // Array um 1 vergroessern
        fields[n] = value;           // Feld speichern

        // Zeilenende oder Dateiende → fertig mit dieser Zeile
        if (FileIsLineEnding(handle) || FileIsEnding(handle))
            break;
    }
}

// BuildFileName() – Erzeugt den CSV-Dateinamen fuer ein Symbol.
// Beispiel: BuildFileName("USDCAD") → "USDCAD_live_trades.csv"
string BuildFileName(const string symbol)
{
    return symbol + InpFileSuffix;  // Symbol + Suffix (z.B. "_live_trades.csv")
}

// ParseBool() – Wandelt einen CSV-Text in true/false um.
// Akzeptiert verschiedene Formate: "1", "true", "ja", "yes" → true
// Alles andere ("0", "false", "nein") → false
bool ParseBool(const string raw)
{
    string v = raw;
    StringToLower(v);  // In Kleinbuchstaben fuer Vergleich
    return (v == "1" || v == "true" || v == "ja" || v == "yes");
}

// SafeField() – Holt ein Feld aus dem Array, mit Fallback bei ungueltigem Index.
// Schuetzt vor Absturz wenn eine CSV-Zeile weniger Spalten hat als erwartet.
// Beispiel: SafeField(cols, 5, "0") → cols[5] oder "0" wenn Index ungueltig
string SafeField(string &arr[], const int idx, const string fallback)
{
    if (idx < 0 || idx >= ArraySize(arr))
        return fallback;  // Index ungueltig → Fallback-Wert zurueckgeben
    return arr[idx];       // Normaler Zugriff
}

// ReadLatestSnapshot() – Liest die LETZTE Zeile der CSV-Datei und
// parst sie in einen SignalSnapshot (unsere Signal-Datenstruktur).
// Das ist der KERN des Dashboards: Hier werden die Python-Signale eingelesen.
//
// Ablauf:
//   1. CSV-Datei oeffnen
//   2. Header-Zeile lesen → Spaltenindizes merken
//   3. Alle Zeilen durchlaufen → nur die LETZTE behalten
//   4. Letzte Zeile in SignalSnapshot-Felder umwandeln
//
// Returns: true wenn erfolgreich, false wenn Datei fehlt oder leer
bool ReadLatestSnapshot(const string symbol, SignalSnapshot &out)
{
    // Datei-Flags: Lesen + CSV-Format + ANSI-Text
    int flags = FILE_READ | FILE_CSV | FILE_ANSI;
    if (InpUseCommonFiles)
        flags |= FILE_COMMON;  // Common-Ordner (fuer alle MT5-Terminals zugaenglich)

    string file_name = BuildFileName(symbol);  // z.B. "USDCAD_live_trades.csv"
    int h = FileOpen(file_name, flags, ',');    // Komma als Trennzeichen
    if (h == INVALID_HANDLE)  // Datei konnte nicht geoeffnet werden
    {
        out.symbol = symbol;
        out.valid = false;
        return false;
    }

    // SCHRITT 1: Header-Zeile einlesen und Spalten-Positionen merken
    string headers[];
    ReadCsvLine(h, headers);
    if (ArraySize(headers) == 0)  // Leere Datei oder defekte Header
    {
        FileClose(h);
        out.symbol = symbol;
        out.valid = false;
        return false;
    }

    // SCHRITT 2: Spalten-Positionen fuer alle benoetigten Felder suchen
    // z.B. idx_time = 0, idx_richtung = 1, idx_prob = 2, ...
    int idx_time = FindHeaderIndex(headers, "time");
    int idx_richtung = FindHeaderIndex(headers, "richtung");   // "Long", "Short" oder "Kein"
    int idx_prob = FindHeaderIndex(headers, "prob");            // Wahrscheinlichkeit 0.0 bis 1.0
    int idx_regime = FindHeaderIndex(headers, "regime");        // Marktphase 0, 1, 2
    int idx_regime_nm = FindHeaderIndex(headers, "regime_name"); // z.B. "Trending", "Range"
    int idx_paper = FindHeaderIndex(headers, "paper_trading");  // true/false
    int idx_modus = FindHeaderIndex(headers, "modus");          // "PAPER" oder "LIVE"
    int idx_entry = FindHeaderIndex(headers, "entry_price");    // Einstiegspreis
    int idx_sl = FindHeaderIndex(headers, "sl_price");          // Stop-Loss Preis
    int idx_tp = FindHeaderIndex(headers, "tp_price");          // Take-Profit Preis

    // SCHRITT 3: Alle Zeilen durchlaufen, nur die LETZTE Zeile behalten
    // (Wir brauchen immer das aktuellste Signal)
    string last[];     // Speichert die Felder der letzten gelesenen Zeile
    long row_count = 0; // Zaehlt die Datenzeilen (ohne Header)

    while (!FileIsEnding(h))
    {
        string cols[];
        ReadCsvLine(h, cols);  // Naechste Zeile lesen
        if (ArraySize(cols) == 0)  // Leere Zeile ueberspringen
            continue;

        row_count++;               // Zeilenzaehler erhoehen
        ArrayCopy(last, cols);     // Letzte Zeile merken (ueberschreibt vorherige)
    }

    FileClose(h);  // Datei schliessen (wichtig: Ressource freigeben!)

    // Keine Datenzeilen gefunden?
    if (row_count == 0 || ArraySize(last) == 0)
    {
        out.symbol = symbol;
        out.valid = false;
        return false;
    }

    // SCHRITT 4: Letzte Zeile in SignalSnapshot-Felder umwandeln
    // SafeField schuetzt vor fehlenden Spalten (gibt Fallback zurueck)
    out.symbol = symbol;
    out.rows = row_count;  // Gesamtzahl der Signal-Zeilen in der CSV
    out.valid = true;      // Snapshot ist gueltig

    // Zeitstempel parsen: "2025.06.26 14:00" → datetime
    out.ts = StringToTime(SafeField(last, idx_time, "1970.01.01 00:00"));

    // Trading-Signal: "Long", "Short" oder "Kein"
    out.richtung = SafeField(last, idx_richtung, "Unbekannt");

    // Modell-Wahrscheinlichkeit: 0.0 bis 1.0 (z.B. 0.67 = 67%)
    out.prob = StringToDouble(SafeField(last, idx_prob, "0"));

    // Marktphase: 0=Range, 1=Trending, 2=Volatile
    out.regime = (int)StringToInteger(SafeField(last, idx_regime, "-1"));
    out.regime_name = SafeField(last, idx_regime_nm, "?");

    // Paper-Trading Flag und Modus
    out.paper = ParseBool(SafeField(last, idx_paper, "true"));
    out.modus = SafeField(last, idx_modus, "PAPER");

    // Preislevels fuer Entry, Stop-Loss und Take-Profit
    out.entry_price = StringToDouble(SafeField(last, idx_entry, "0"));
    out.sl_price = StringToDouble(SafeField(last, idx_sl, "0"));
    out.tp_price = StringToDouble(SafeField(last, idx_tp, "0"));

    return true;
}

// ============================================================
// 7. STATUS & LOGGING (Zustand des Systems bestimmen)
// ============================================================

// ExpectedFolderText() – Gibt den erwarteten Dateipfad zurueck fuer Debug-Ausgabe.
// Hilft beim Troubleshooting wenn die CSV-Datei nicht gefunden wird.
string ExpectedFolderText()
{
    string base;
    if (InpUseCommonFiles)
        base = TerminalInfoString(TERMINAL_COMMONDATA_PATH) + "\\Files\\"; // Common-Ordner
    else
        base = TerminalInfoString(TERMINAL_DATA_PATH) + "\\MQL5\\Files\\"; // Lokaler Ordner
    return base;
}

// LogFileState() – Protokolliert den Dateistatus im MT5-Log.
// Warnt wenn eine CSV fehlt, meldet wenn sie wieder da ist.
// Verwendet Ratelimiting: loggt nicht bei jedem Timer-Tick, sondern
// nur alle InpMissingFileLogEverySec Sekunden (verhindert Log-Spam).
void LogFileState(
    const string symbol,
    const string file_name,
    const bool ok,         // true = Datei gefunden, false = Datei fehlt
    bool &was_missing,     // Referenz: merkt sich ob Datei vorher fehlte
    datetime &last_log_ts) // Referenz: Zeitpunkt der letzten Log-Nachricht
{
    datetime now = TimeCurrent();

    if (!ok)  // Datei fehlt
    {
        // Pruefen ob wir jetzt loggen sollen (Ratelimiting)
        bool should_log = (!was_missing);  // Erster Fehler: immer loggen
        if (!should_log && InpMissingFileLogEverySec > 0)
            should_log = ((now - last_log_ts) >= InpMissingFileLogEverySec);  // Zeitabstand einhalten

        if (should_log)
        {
            PrintFormat(
                "[Dashboard] %s Datei fehlt: %s | Erwartet in: %s",
                symbol,
                file_name,
                ExpectedFolderText());
            last_log_ts = now;
        }

        was_missing = true;  // Status merken: Datei fehlt
        return;
    }

    // Datei ist wieder da → Entwarnung loggen
    if (was_missing)
        PrintFormat("[Dashboard] %s Datei erkannt: %s", symbol, file_name);

    was_missing = false;  // Status zuruecksetzen
}

// FreshnessText() – Erzeugt einen lesbaren Text ueber die Aktualitaet der Daten.
// Beispiel: "OK (12 min)" oder "STALE (85 min)"
// "OK" = Daten sind frisch, "STALE" = Daten sind veraltet
string FreshnessText(const SignalSnapshot &snap)
{
    if (!snap.valid || snap.ts <= 0)
        return "keine Daten";  // Kein gueltiger Zeitstempel vorhanden

    int stale_minutes = EffectiveStaleMinutes();              // Grenzwert (z.B. 70 Min)
    int age_min = (int)((TimeCurrent() - snap.ts) / 60);     // Alter in Minuten
    string tag = (age_min <= stale_minutes) ? "OK" : "STALE"; // Frisch oder veraltet?
    return StringFormat("%s (%d min)", tag, age_min);
}

// SnapshotState() – Bestimmt den Status-Code eines Snapshots.
// Moegliche Rueckgabewerte:
//   "MISSING"        – CSV-Datei fehlt komplett
//   "NO_TS"          – Datei da, aber kein Zeitstempel
//   "STALE"          – Daten zu alt (ueber Grenzwert)
//   "LIVE_NO_SIGNAL" – Daten frisch, aber Dir=Kein
//   "LIVE_SIGNAL"    – Daten frisch UND aktives Signal (Long/Short)
string SnapshotState(const SignalSnapshot &snap)
{
    if (!snap.valid)
        return "MISSING";  // CSV nicht gefunden oder leer

    if (snap.ts <= 0)
        return "NO_TS";    // Kein Zeitstempel in der letzten Zeile

    int stale_minutes = EffectiveStaleMinutes();
    int age_min = (int)((TimeCurrent() - snap.ts) / 60);
    if (age_min > stale_minutes)
        return "STALE";    // Daten aelter als Grenzwert

    // Daten sind frisch – hat das Modell ein Signal generiert?
    if (StringCompare(snap.richtung, "Kein") == 0)
        return "LIVE_NO_SIGNAL";  // Modell sagt: kein Trade

    return "LIVE_SIGNAL";  // Modell sagt: Long oder Short!
}

// ============================================================
// 8. ENTRY-HISTORY (Vergangene Signale aus CSV lesen)
// ============================================================
// Diese Funktionen lesen die letzten N Signale aus der CSV,
// damit sie als Pfeile auf dem Chart gezeichnet werden koennen.

// ReadRecentEntries() – Liest die letzten max_entries Eintraege
// mit echtem Signal (Long/Short) aus der CSV-Datei.
// Eintraege mit Dir=Kein werden uebersprungen.
bool ReadRecentEntries(const string symbol, EntryPoint &entries[], const int max_entries)
{
    ArrayResize(entries, 0);  // Array leeren
    if (max_entries <= 0)
        return false;

    // CSV-Datei oeffnen (gleich wie bei ReadLatestSnapshot)
    int flags = FILE_READ | FILE_CSV | FILE_ANSI;
    if (InpUseCommonFiles)
        flags |= FILE_COMMON;

    string file_name = BuildFileName(symbol);
    int h = FileOpen(file_name, flags, ',');
    if (h == INVALID_HANDLE)
        return false;

    // Header lesen und Spaltenindizes merken
    string headers[];
    ReadCsvLine(h, headers);
    if (ArraySize(headers) == 0)
    {
        FileClose(h);
        return false;
    }

    int idx_time = FindHeaderIndex(headers, "time");
    int idx_richtung = FindHeaderIndex(headers, "richtung");
    int idx_prob = FindHeaderIndex(headers, "prob");
    int idx_regime_nm = FindHeaderIndex(headers, "regime_name");
    int idx_entry = FindHeaderIndex(headers, "entry_price");

    // Alle Zeilen mit Signal (Long/Short) sammeln
    while (!FileIsEnding(h))
    {
        string cols[];
        ReadCsvLine(h, cols);
        if (ArraySize(cols) == 0)
            continue;

        // Nur Long und Short behalten, "Kein" ueberspringen
        string dir = SafeField(cols, idx_richtung, "Kein");
        bool is_long = (StringCompare(dir, "Long") == 0);
        bool is_short = (StringCompare(dir, "Short") == 0);
        if (!is_long && !is_short)
            continue;  // Kein Signal → ueberspringen

        // Zeitstempel und Entry-Preis validieren
        datetime ts = StringToTime(SafeField(cols, idx_time, "1970.01.01 00:00"));
        double entry = StringToDouble(SafeField(cols, idx_entry, "0"));
        if (ts <= 0 || entry <= 0)
            continue;  // Ungueltiger Datensatz → ueberspringen

        // Neuen EntryPoint ans Array anhaengen
        int n = ArraySize(entries);
        ArrayResize(entries, n + 1);
        entries[n].ts = ts;
        entries[n].richtung = dir;
        entries[n].prob = StringToDouble(SafeField(cols, idx_prob, "0"));
        entries[n].regime_name = SafeField(cols, idx_regime_nm, "?");
        entries[n].entry_price = entry;
    }

    FileClose(h);

    int total = ArraySize(entries);
    if (total <= 0)
        return false;  // Keine Eintraege mit Signal gefunden

    // Wenn wir mehr Eintraege haben als gewuenscht:
    // nur die letzten max_entries behalten (neueste zuerst)
    if (total <= max_entries)
        return true;  // Passt, alle behalten

    // Array auf die letzten max_entries kuerzen
    EntryPoint last_entries[];
    ArrayResize(last_entries, max_entries);
    int start = total - max_entries;  // Start-Index fuer die letzten N
    for (int i = 0; i < max_entries; i++)
        last_entries[i] = entries[start + i];

    // Ergebnis zurueckschreiben
    ArrayResize(entries, max_entries);
    for (int i = 0; i < max_entries; i++)
        entries[i] = last_entries[i];

    return true;
}

// CountRecentEntries() – Zaehlt wie viele Signal-Eintraege in der CSV sind.
// Wird fuer die Debug-Anzeige "Hist: USDCAD=5" im Dashboard genutzt.
int CountRecentEntries(const string symbol, const int max_entries)
{
    EntryPoint entries[];
    if (!ReadRecentEntries(symbol, entries, max_entries))
        return 0;
    return ArraySize(entries);  // Anzahl der Eintraege mit Signal
}

// OverallState() – Kombiniert den Status beider Symbole zu einem Gesamtstatus.
// Wird oben im Dashboard angezeigt: "CONNECTED", "PARTIAL", "WAITING_FOR_CSV"
//
// Logik:
//   Beide LIVE (frisch)  → "CONNECTED"       (alles OK)
//   Beide MISSING        → "WAITING_FOR_CSV" (Python laeuft nicht?)
//   Einer STALE          → "PARTIAL_STALE"   (teilweise veraltet)
//   Sonst                → "PARTIAL"         (gemischter Status)
string OverallState(const string st1, const string st2)
{
    // Pruefen ob beide Symbole "LIVE" im Status haben
    bool ok1 = (StringFind(st1, "LIVE") == 0);  // Beginnt mit "LIVE"?
    bool ok2 = (StringFind(st2, "LIVE") == 0);

    if (ok1 && ok2)
        return "CONNECTED";  // Beide Symbole liefern frische Daten

    if (StringCompare(st1, "MISSING") == 0 && StringCompare(st2, "MISSING") == 0)
        return "WAITING_FOR_CSV";  // Keine CSV-Dateien da

    if (StringCompare(st1, "STALE") == 0 || StringCompare(st2, "STALE") == 0)
        return "PARTIAL_STALE";  // Mindestens ein Symbol veraltet

    return "PARTIAL";  // Gemischter Status
}

// MaybeAlert() – Sendet einen MT5-Alert wenn ein NEUES Signal erkannt wird.
// Alerts werden nur einmal pro Signal gesendet (nicht bei jeder Aktualisierung).
// Der Zeitstempel last_alert_ts merkt sich das letzte alerte Signal.
void MaybeAlert(const SignalSnapshot &snap, datetime &last_alert_ts)
{
    // Alerts deaktiviert oder keine gueltigen Daten?
    if (!InpEnableAlerts || !snap.valid || snap.ts <= 0)
        return;

    // Nur bei NEUEN Zeilen alerten (Zeitstempel muss neuer sein als letzter Alert)
    if (snap.ts <= last_alert_ts)
        return;

    // Dir=Kein → kein Alert, aber Zeitstempel trotzdem merken
    if (StringCompare(snap.richtung, "Kein") == 0)
    {
        last_alert_ts = snap.ts;
        return;
    }

    // Alert-Nachricht zusammenbauen und senden
    string msg = StringFormat(
        "[MT5 Dashboard] %s | %s | Prob=%.3f | Regime=%s | %s",
        snap.symbol,
        snap.richtung,
        snap.prob,
        snap.regime_name,
        TimeToString(snap.ts, TIME_DATE | TIME_MINUTES));

    Alert(msg);   // MT5-Popup (und Sound wenn aktiviert)
    Print(msg);   // Ins Journal schreiben
    last_alert_ts = snap.ts;  // Zeitstempel merken (verhindert doppelte Alerts)
}

// ============================================================
// 9. CHART-ZEICHNUNGEN (Objekte auf den Chart malen)
// ============================================================
// MT5 verwendet Chart-Objekte (OBJ_HLINE, OBJ_ARROW, OBJ_RECTANGLE, OBJ_LABEL)
// um visuelle Elemente auf dem Chart darzustellen.
// Jedes Objekt hat einen eindeutigen Namen (String) und Eigenschaften
// wie Farbe, Position, Stil etc.
//
// Namenskonvention: Alle unsere Objekte beginnen mit "PYML_"
// (Python ML), damit sie beim Aufraemen leicht identifiziert werden.

// DeleteObjPrefix() – Loescht ALLE Chart-Objekte die mit dem Prefix beginnen.
// Beispiel: DeleteObjPrefix("PYML_USDCAD_") loescht alle USDCAD-bezogenen Objekte.
// Wichtig: Rueckwaerts iterieren weil ObjectDelete den Index aendert!
void DeleteObjPrefix(const string prefix)
{
    int total = ObjectsTotal(0, 0, -1);  // Alle Objekte auf Chart 0 zaehlen
    // Rueckwaerts iterieren (sonst ueberspringen wir Objekte nach dem Loeschen)
    for (int i = total - 1; i >= 0; i--)
    {
        string name = ObjectName(0, i, 0, -1);  // Name des Objekts holen
        if (StringFind(name, prefix) == 0)       // Beginnt mit unserem Prefix?
            ObjectDelete(0, name);               // Ja → loeschen
    }
}

// DrawHLine() – Zeichnet eine horizontale Linie auf den Chart.
// Wird verwendet fuer: Entry-Linie, SL-Linie, TP-Linie, EMA-Guides.
// Parameter:
//   name    – Eindeutiger Objektname (z.B. "PYML_USDCAD_SL_LINE")
//   price   – Preis an dem die Linie gezeichnet wird
//   clr     – Farbe der Linie
//   style   – Linienstil: STYLE_SOLID (durchgezogen), STYLE_DASH, STYLE_DOT
//   width   – Linienstaerke in Pixeln
//   tooltip – Tooltip-Text beim Ueberfahren mit der Maus
void DrawHLine(const string name, const double price, const color clr,
               const ENUM_LINE_STYLE style, const int width, const string tooltip)
{
    if (price <= 0)  // Ungueltiger Preis → nichts zeichnen
        return;

    // Objekt erstellen wenn es noch nicht existiert, sonst Preis aktualisieren
    if (ObjectFind(0, name) < 0)
        ObjectCreate(0, name, OBJ_HLINE, 0, 0, price);  // Neues Objekt: OBJ_HLINE = horizontale Linie
    else
        ObjectSetDouble(0, name, OBJPROP_PRICE, price);  // Preis aktualisieren

    // Eigenschaften setzen
    ObjectSetInteger(0, name, OBJPROP_COLOR, clr);        // Farbe
    ObjectSetInteger(0, name, OBJPROP_STYLE, style);      // Linienstil
    ObjectSetInteger(0, name, OBJPROP_WIDTH, width);      // Linienstaerke
    ObjectSetInteger(0, name, OBJPROP_BACK, true);        // Hinter der Kerze zeichnen
    ObjectSetInteger(0, name, OBJPROP_SELECTABLE, false); // Nicht anklickbar
    ObjectSetString(0, name, OBJPROP_TOOLTIP, tooltip);   // Maus-Tooltip
}

// DrawArrow() – Zeichnet einen Pfeil auf den Chart (fuer Entry-Signale).
// Arrow-Codes: 233 = Pfeil hoch (Long), 234 = Pfeil runter (Short)
//              241/242 = kleinere Marker fuer Historie-Eintraege
void DrawArrow(const string name, const datetime time_val, const double price,
               const int arrow_code, const color clr, const string tooltip)
{
    if (price <= 0)  // Kein gueltiger Preis
        return;

    // Objekt erstellen oder Position aktualisieren
    if (ObjectFind(0, name) < 0)
        ObjectCreate(0, name, OBJ_ARROW, 0, time_val, price);  // OBJ_ARROW = Pfeil-Objekt
    else
    {
        ObjectSetInteger(0, name, OBJPROP_TIME, time_val);   // Zeitpunkt aktualisieren
        ObjectSetDouble(0, name, OBJPROP_PRICE, price);      // Preis aktualisieren
    }

    ObjectSetInteger(0, name, OBJPROP_ARROWCODE, arrow_code); // Pfeil-Symbol (Wingdings-Font)
    ObjectSetInteger(0, name, OBJPROP_COLOR, clr);            // Farbe
    ObjectSetInteger(0, name, OBJPROP_WIDTH, 2);              // Groesse
    ObjectSetInteger(0, name, OBJPROP_SELECTABLE, false);     // Nicht anklickbar
    ObjectSetString(0, name, OBJPROP_TOOLTIP, tooltip);       // Maus-Tooltip
}

// DrawRectangle() – Zeichnet ein halbtransparentes Rechteck auf den Chart.
// Wird verwendet fuer SL-Zone (rot) und TP-Zone (gruen) zwischen Entry und Zielpreis.
// Parameter:
//   t1, p1 – Ecke 1 (Zeitpunkt + Preis, meistens Entry-Zeitpunkt + Entry-Preis)
//   t2, p2 – Ecke 2 (Zeitpunkt + Preis, meistens Zukunft + SL/TP-Preis)
void DrawRectangle(const string name, const datetime t1, const double p1,
                   const datetime t2, const double p2, const color clr,
                   const string tooltip)
{
    if (p1 <= 0 || p2 <= 0)  // Ungueltige Preise
        return;

    // Objekt erstellen oder Eckpunkte aktualisieren
    if (ObjectFind(0, name) < 0)
        ObjectCreate(0, name, OBJ_RECTANGLE, 0, t1, p1, t2, p2);  // OBJ_RECTANGLE = Rechteck
    else
    {
        // Eckpunkt 0 (links oben) und Eckpunkt 1 (rechts unten) setzen
        ObjectSetInteger(0, name, OBJPROP_TIME, 0, t1);   // Zeit Ecke 1
        ObjectSetDouble(0, name, OBJPROP_PRICE, 0, p1);   // Preis Ecke 1
        ObjectSetInteger(0, name, OBJPROP_TIME, 1, t2);   // Zeit Ecke 2
        ObjectSetDouble(0, name, OBJPROP_PRICE, 1, p2);   // Preis Ecke 2
    }

    ObjectSetInteger(0, name, OBJPROP_COLOR, clr);        // Randfarbe
    ObjectSetInteger(0, name, OBJPROP_BACK, true);        // Im Hintergrund (hinter den Kerzen)
    ObjectSetInteger(0, name, OBJPROP_FILL, true);        // Ausgefuellt (nicht nur Rahmen)
    ObjectSetInteger(0, name, OBJPROP_SELECTABLE, false); // Nicht anklickbar
    ObjectSetString(0, name, OBJPROP_TOOLTIP, tooltip);   // Maus-Tooltip
}

// ============================================================
// 10. TRADE-VISUALISIERUNG (Entry/SL/TP auf dem Chart)
// ============================================================
// Wenn Python ein Signal sendet (Long/Short mit Entry, SL, TP),
// zeichnet DrawTradeOnChart() diese Levels visuell auf den Chart:
//   - Entry-Pfeil (hoch/runter je nach Richtung)
//   - Entry-Linie (gestrichelt)
//   - SL-Linie + SL-Zone (rotes Rechteck)
//   - TP-Linie + TP-Zone (gruenes Rechteck)
//   - Info-Label oben rechts mit allen Details

void DrawTradeOnChart(const SignalSnapshot &snap)
{
    // Nur zeichnen wenn das Symbol zum aktuellen Chart passt
    // (z.B. USDCAD-Signal nur auf USDCAD-Chart zeichnen)
    if (StringCompare(snap.symbol, Symbol()) != 0)
        return;

    if (!InpDrawTrades || !snap.valid)  // Zeichnung deaktiviert oder keine Daten
        return;

    // Prefix fuer alle Objekte dieses Symbols (z.B. "PYML_USDCAD_")
    string pfx = "PYML_" + snap.symbol + "_";

    // Zuerst ALLE alten Zeichnungen loeschen (clean slate)
    DeleteObjPrefix(pfx);

    // Kein Signal oder kein Entry-Preis → nichts zu zeichnen
    if (StringCompare(snap.richtung, "Kein") == 0 || snap.entry_price <= 0)
        return;

    // Long = Blau/Gruen, Short = Rot/Orange (konfigurierbar)
    bool is_long = (StringCompare(snap.richtung, "Long") == 0);
    color trade_clr = is_long ? InpColorLong : InpColorShort;

    // Zeitrahmen: Entry-Zeitpunkt bis 4 Stunden in die Zukunft (fuer Rechtecke)
    datetime entry_time = snap.ts;
    datetime future_time = entry_time + 4 * 3600;  // +4 Stunden

    // --- 1) Entry-Pfeil ---
    // Arrow 233 = grosser Pfeil hoch (Long), 234 = grosser Pfeil runter (Short)
    int arrow = is_long ? 233 : 234;
    string tip_entry = StringFormat(
        "%s Entry @ %.5f | Prob=%.1f%% | %s",
        snap.richtung, snap.entry_price, snap.prob * 100.0, snap.regime_name);
    DrawArrow(pfx + "ENTRY", entry_time, snap.entry_price, arrow, trade_clr, tip_entry);

    // --- 2) Entry-Linie (horizontal, gestrichelt) ---
    string tip_line = StringFormat("Entry: %.5f", snap.entry_price);
    DrawHLine(pfx + "ENTRY_LINE", snap.entry_price, trade_clr, STYLE_DASH, 1, tip_line);

    // --- 3) Stop-Loss: Linie + farbige Zone ---
    if (snap.sl_price > 0)
    {
        // Pips-Abstand berechnen: |Entry - SL| / Point / 10
        // Point = kleinste Preisaenderung, /10 = Points zu Pips
        string tip_sl = StringFormat("SL: %.5f (%.1f Pips)",
            snap.sl_price,
            MathAbs(snap.entry_price - snap.sl_price) / SymbolInfoDouble(Symbol(), SYMBOL_POINT) / 10.0);
        DrawHLine(pfx + "SL_LINE", snap.sl_price, InpColorSL, STYLE_DOT, 2, tip_sl);

        // SL-Zone: rotes Rechteck von Entry-Preis bis SL-Preis
        DrawRectangle(pfx + "SL_ZONE", entry_time, snap.entry_price,
                      future_time, snap.sl_price, InpColorSL, tip_sl);
    }

    // --- 4) Take-Profit: Linie + farbige Zone ---
    if (snap.tp_price > 0)
    {
        // TP-Abstand in Pips
        string tip_tp = StringFormat("TP: %.5f (%.1f Pips)",
            snap.tp_price,
            MathAbs(snap.tp_price - snap.entry_price) / SymbolInfoDouble(Symbol(), SYMBOL_POINT) / 10.0);
        DrawHLine(pfx + "TP_LINE", snap.tp_price, InpColorTP, STYLE_DOT, 2, tip_tp);

        // TP-Zone: gruenes Rechteck von Entry-Preis bis TP-Preis
        DrawRectangle(pfx + "TP_ZONE", entry_time, snap.entry_price,
                      future_time, snap.tp_price, InpColorTP, tip_tp);
    }

    // --- 5) Info-Label oben rechts ---
    // Zeigt alle Trade-Details kompakt in einer Zeile an
    string label_name = pfx + "INFO";
    string info_text = StringFormat(
        "%s %s @ %.5f | SL=%.5f | TP=%.5f | Prob=%.0f%% | %s",
        snap.symbol, snap.richtung, snap.entry_price,
        snap.sl_price, snap.tp_price, snap.prob * 100.0, snap.regime_name);

    // Label positionieren: rechts oben, Consolas-Font fuer gleichmaessige Breite
    if (ObjectFind(0, label_name) < 0)
        ObjectCreate(0, label_name, OBJ_LABEL, 0, 0, 0);  // OBJ_LABEL = Text-Objekt

    ObjectSetInteger(0, label_name, OBJPROP_XDISTANCE, 10);          // 10px vom rechten Rand
    ObjectSetInteger(0, label_name, OBJPROP_YDISTANCE, 50);          // 50px von oben
    ObjectSetInteger(0, label_name, OBJPROP_CORNER, CORNER_RIGHT_UPPER); // Ecke: Rechts oben
    ObjectSetInteger(0, label_name, OBJPROP_ANCHOR, ANCHOR_RIGHT_UPPER); // Anker: Rechts oben
    ObjectSetString(0, label_name, OBJPROP_TEXT, info_text);          // Text setzen
    ObjectSetString(0, label_name, OBJPROP_FONT, "Consolas");        // Monospace-Font
    ObjectSetInteger(0, label_name, OBJPROP_FONTSIZE, 10);
    ObjectSetInteger(0, label_name, OBJPROP_COLOR, trade_clr);       // Farbe = Trade-Farbe

    ChartRedraw(0);  // Chart sofort neu zeichnen (damit alles sichtbar wird)
}

// DrawEntryHistory() – Zeichnet die letzten N Signale als kleine Pfeile auf den Chart.
// Gibt eine visuelle Historie: Wo hat das Modell frueher Signale generiert?
// Verwendet kleinere Marker (241/242) als das aktuelle Signal (233/234).
void DrawEntryHistory(const SignalSnapshot &snap)
{
    // Nur fuer das aktive Chart-Symbol zeichnen
    if (StringCompare(snap.symbol, Symbol()) != 0)
        return;

    // Beide Optionen muessen aktiviert sein
    if (!InpDrawTrades || !InpDrawEntryHistory)
        return;

    string pfx = "PYML_" + snap.symbol + "_";
    DeleteObjPrefix(pfx + "HIST_");  // Alte Historie-Pfeile loeschen

    // Die letzten N Signale aus der CSV lesen
    EntryPoint entries[];
    if (!ReadRecentEntries(snap.symbol, entries, InpMaxTradesOnChart))
        return;

    // Jeden historischen Entry als kleinen Pfeil zeichnen
    for (int i = 0; i < ArraySize(entries); i++)
    {
        bool is_long = (StringCompare(entries[i].richtung, "Long") == 0);
        int arrow = is_long ? 241 : 242;  // Kleinere Symbole: 241 = Dreieck hoch, 242 = Dreieck runter
        color clr = is_long ? InpColorLong : InpColorShort;

        // Eindeutiger Name: PYML_USDCAD_HIST_0_1719403200
        string name = pfx + "HIST_" + IntegerToString(i) + "_" + IntegerToString((int)entries[i].ts);
        // Tooltip mit allen Details
        string tip = StringFormat(
            "History %s @ %.5f | Prob=%.1f%% | %s | %s",
            entries[i].richtung,
            entries[i].entry_price,
            entries[i].prob * 100.0,
            entries[i].regime_name,
            TimeToString(entries[i].ts, TIME_DATE | TIME_MINUTES));

        DrawArrow(name, entries[i].ts, entries[i].entry_price, arrow, clr, tip);
        ObjectSetInteger(0, name, OBJPROP_WIDTH, 1);  // Duenner als aktueller Entry
    }

    ChartRedraw(0);
}

// ============================================================
// 11. TECHNISCHE INDIKATOREN AUSLESEN
// ============================================================

// GetLatestBufferValue() – Liest den AKTUELLEN Wert eines Indikators aus.
// CopyBuffer() kopiert Indikator-Daten in unser Array.
// ArraySetAsSeries(true) = Index 0 ist der NEUESTE Wert (nicht der aelteste).
// Wird verwendet fuer: EMA20/50/200-Werte und ATR14.
bool GetLatestBufferValue(const int handle, double &out_value)
{
    out_value = 0.0;
    if (handle == INVALID_HANDLE)  // Indikator nicht erstellt
        return false;

    double buff[];               // Temporaerer Puffer fuer Indikator-Daten
    ArraySetAsSeries(buff, true); // Neuester Wert = Index 0
    int copied = CopyBuffer(handle, 0, 0, 1, buff); // 1 Wert kopieren (den aktuellen)
    if (copied < 1)              // Kopieren fehlgeschlagen
        return false;

    out_value = buff[0];          // Aktuellen Wert zurueckgeben
    return (out_value > 0);       // true wenn Wert gueltig (> 0)
}

// DrawEmaGuides() – Zeichnet horizontale Linien fuer EMA20/50/200 auf den Chart.
// Zusaetzlich ein Status-Label das die EMA-Struktur anzeigt:
//   BULL: EMA20 > EMA50 > EMA200 (Aufwaertstrend)
//   BEAR: EMA20 < EMA50 < EMA200 (Abwaertstrend)
//   MIXED: EMAs nicht in Reihenfolge (Seitwaertsmarkt / Uebergang)
void DrawEmaGuides()
{
    string pfx = "PYML_EMA_GUIDE_";

    // Wenn deaktiviert: vorhandene Linien loeschen und raus
    if (!InpDrawTechnicalOverlay || !InpDrawEmaGuides)
    {
        DeleteObjPrefix(pfx);
        return;
    }

    // Aktuelle EMA-Werte vom Indikator holen
    double ema20 = 0.0;
    double ema50 = 0.0;
    double ema200 = 0.0;
    if (!GetLatestBufferValue(g_h_ema20, ema20) ||
        !GetLatestBufferValue(g_h_ema50, ema50) ||
        !GetLatestBufferValue(g_h_ema200, ema200))
    {
        return;  // Nicht alle EMA-Werte verfuegbar
    }

    // EMA-Linien zeichnen (Farben konfigurierbar ueber Input-Parameter)
    int width = MathMax(1, InpEmaGuideWidth);
    DrawHLine(pfx + "20", ema20, InpEma20GuideColor, STYLE_SOLID, width,
              StringFormat("EMA20 (blau): %.5f", ema20));
    DrawHLine(pfx + "50", ema50, InpEma50GuideColor, STYLE_SOLID, width,
              StringFormat("EMA50 (rot): %.5f", ema50));
    DrawHLine(pfx + "200", ema200, InpEma200GuideColor, STYLE_SOLID, width,
              StringFormat("EMA200 (gruen): %.5f", ema200));

    // EMA-Struktur-Label anzeigen? (Optional)
    if (!InpShowEmaStructureLabel)
    {
        if (ObjectFind(0, pfx + "STATUS") >= 0)
            ObjectDelete(0, pfx + "STATUS");  // Label entfernen wenn deaktiviert
        return;
    }

    // Aktuellen Preis holen (Schlusskurs der letzten Kerze oder Bid)
    double price = iClose(Symbol(), PERIOD_CURRENT, 0);
    if (price <= 0)
        price = SymbolInfoDouble(Symbol(), SYMBOL_BID);

    // EMA-Struktur bestimmen
    bool bull_stack = (ema20 > ema50 && ema50 > ema200); // Bullish: 20 > 50 > 200
    bool bear_stack = (ema20 < ema50 && ema50 < ema200); // Bearish: 20 < 50 < 200
    bool price_above_all = (price > ema20 && price > ema50 && price > ema200); // Preis ueber allen EMAs
    bool price_below_all = (price < ema20 && price < ema50 && price < ema200); // Preis unter allen EMAs

    // Status-Text und Farbe bestimmen
    string structure = "MIXED";       // Standard: keine klare Struktur
    color state_color = clrSilver;    // Grau fuer unklare Situation
    // Optimaler Bullish: EMAs gestapelt UND Preis darueber
    if (bull_stack && price_above_all)
    {
        structure = "BULL: EMA20>EMA50>EMA200 | Preis ueber allen";
        state_color = InpEma20GuideColor;  // Blau (EMA20-Farbe)
    }
    // Optimaler Bearish: EMAs gestapelt UND Preis darunter
    else if (bear_stack && price_below_all)
    {
        structure = "BEAR: EMA20<EMA50<EMA200 | Preis unter allen";
        state_color = InpColorShort;  // Rot
    }
    // Bull Stack, aber Preis zwischen den EMAs (Pullback-Zone)
    else if (bull_stack)
    {
        structure = "BULL Stack, Preis in Pullback-Zone";
        state_color = InpEma50GuideColor;  // Orange (EMA50-Farbe)
    }
    // Bear Stack, aber Preis zwischen den EMAs (Pullback-Zone)
    else if (bear_stack)
    {
        structure = "BEAR Stack, Preis in Pullback-Zone";
        state_color = InpEma50GuideColor;
    }
    // Sonst: "MIXED" bleibt als Default

    // Status-Label oben rechts positionieren
    string name = pfx + "STATUS";
    // Label erstellen und positionieren
    if (ObjectFind(0, name) < 0)
        ObjectCreate(0, name, OBJ_LABEL, 0, 0, 0);

    ObjectSetInteger(0, name, OBJPROP_XDISTANCE, 10);                // 10px vom rechten Rand
    ObjectSetInteger(0, name, OBJPROP_YDISTANCE, 42);                // 42px von oben
    ObjectSetInteger(0, name, OBJPROP_CORNER, CORNER_RIGHT_UPPER);   // Rechts oben
    ObjectSetInteger(0, name, OBJPROP_ANCHOR, ANCHOR_RIGHT_UPPER);
    ObjectSetInteger(0, name, OBJPROP_SELECTABLE, false);            // Nicht anklickbar
    ObjectSetInteger(0, name, OBJPROP_HIDDEN, true);                 // Nicht in Objektliste
    ObjectSetString(0, name, OBJPROP_FONT, "Consolas");              // Monospace-Font
    ObjectSetInteger(0, name, OBJPROP_FONTSIZE, 9);
    ObjectSetInteger(0, name, OBJPROP_COLOR, state_color);           // Farbe = Struktur-Farbe
    ObjectSetString(0, name, OBJPROP_TEXT, structure);                // z.B. "BULL: EMA20>EMA50>EMA200"
}

// ============================================================
// 12. AMPEL-SYSTEM (Schnelle visuelle Uebersicht)
// ============================================================
// Die Ampel gibt auf einen Blick Auskunft:
//   GRUEN  → Signal aktiv + hohe Wahrscheinlichkeit (>55%) → HANDELN
//   GELB   → Daten frisch, aber kein/schwaches Signal   → BEOBACHTEN
//   ROT    → Daten veraltet oder keine CSV vorhanden     → FINGER WEG
//
// Die Ampel besteht aus 4 Chart-Objekten:
//   1) Hintergrund-Box (schwarz mit farbigem Rand)
//   2) Farbiges Quadrat (das "Licht" der Ampel)
//   3) Hauptlabel (z.B. "GRUEN: HANDELN")
//   4) Detail-Zeile (z.B. "Long | Prob=67% | Signal aktiv!")

void DrawAmpel()
{
    string pfx = "PYML_AMPEL_";  // Prefix fuer alle Ampel-Objekte

    // Ampel deaktiviert → alle Objekte loeschen
    if (!InpShowAmpel)
    {
        DeleteObjPrefix(pfx);
        return;
    }

    // === Schritt 1: Das passende Snapshot fuer diesen Chart finden ===
    // Wenn wir auf einem USDCAD-Chart sind, nehmen wir g_snap1 (USDCAD)
    // Wenn wir auf einem USDJPY-Chart sind, nehmen wir g_snap2 (USDJPY)
    string active_state = "";
    double active_prob = 0.0;
    string active_dir = "Kein";
    bool found_match = false;

    // Pruefen ob Symbol 1 zum aktuellen Chart passt
    if (StringCompare(g_snap1.symbol, Symbol()) == 0 && g_snap1.valid)
    {
        active_state = SnapshotState(g_snap1);
        active_prob = g_snap1.prob;
        active_dir = g_snap1.richtung;
        found_match = true;
    }
    // Pruefen ob Symbol 2 zum aktuellen Chart passt
    else if (StringCompare(g_snap2.symbol, Symbol()) == 0 && g_snap2.valid)
    {
        active_state = SnapshotState(g_snap2);
        active_prob = g_snap2.prob;
        active_dir = g_snap2.richtung;
        found_match = true;
    }

    // Kein passendes Symbol gefunden (z.B. wir sind auf EURUSD-Chart)
    // In diesem Fall: den schlechteren Status beider Symbole nehmen
    if (!found_match)
    {
        string st1 = SnapshotState(g_snap1);
        string st2 = SnapshotState(g_snap2);

        // Wenn eines STALE oder MISSING ist → ROT
        if (StringCompare(st1, "STALE") == 0 || StringCompare(st2, "STALE") == 0 ||
            StringCompare(st1, "MISSING") == 0 || StringCompare(st2, "MISSING") == 0)
        {
            active_state = "STALE";
        }
        // Wenn Symbol 1 ein Signal hat → dessen Daten nehmen
        else if (StringCompare(st1, "LIVE_SIGNAL") == 0)
        {
            active_state = st1;
            active_prob = g_snap1.prob;
            active_dir = g_snap1.richtung;
        }
        // Wenn Symbol 2 ein Signal hat → dessen Daten nehmen
        else if (StringCompare(st2, "LIVE_SIGNAL") == 0)
        {
            active_state = st2;
            active_prob = g_snap2.prob;
            active_dir = g_snap2.richtung;
        }
        // Sonst: kein Signal bei beiden
        else
        {
            active_state = "LIVE_NO_SIGNAL";
            active_prob = MathMax(g_snap1.prob, g_snap2.prob);
            active_dir = "Kein";
        }
    }

    // === Schritt 2: Ampel-Farbe und Text bestimmen ===
    color ampel_color;    // Farbe der Ampel (Gruen/Gelb/Rot)
    string ampel_label;   // Haupttext (z.B. "GRUEN: HANDELN")
    string ampel_detail;  // Detailtext mit mehr Infos

    // Status-Flags fuer die Ampel-Logik
    bool is_stale = (StringCompare(active_state, "STALE") == 0);
    bool is_missing = (StringCompare(active_state, "MISSING") == 0 ||
                       StringCompare(active_state, "NO_TS") == 0);
    bool has_signal = (StringCompare(active_state, "LIVE_SIGNAL") == 0);

    if (is_stale || is_missing)
    {
        // ROT: Daten veraltet oder nicht vorhanden
        ampel_color = clrRed;
        ampel_label = "ROT: WARTEN";
        if (is_stale)
            ampel_detail = "Signal veraltet (STALE) - Finger weg!";
        else
            ampel_detail = "Keine Daten - System pruefen!";
    }
    else if (has_signal && active_prob > 0.55)
    {
        // GRUEN: Aktives Signal mit hoher Wahrscheinlichkeit
        ampel_color = clrLimeGreen;
        ampel_label = "GRUEN: HANDELN";
        ampel_detail = StringFormat("%s | Prob=%.0f%% | Signal aktiv!",
            active_dir, active_prob * 100.0);
    }
    else
    {
        // GELB: Daten frisch, aber kein oder schwaches Signal
        ampel_color = clrGold;
        ampel_label = "GELB: BEOBACHTEN";
        if (has_signal)
            ampel_detail = StringFormat("%s Prob=%.0f%% (unter 55%%) - noch warten",
                active_dir, active_prob * 100.0);
        else
            ampel_detail = StringFormat("Dir=Kein | Prob=%.0f%% - naechsten Candle abwarten",
                active_prob * 100.0);
    }

    // === Schritt 3: Ampel auf dem Chart positionieren ===
    // Position: Unten links im Chart
    int chart_h = (int)ChartGetInteger(0, CHART_HEIGHT_IN_PIXELS, 0);  // Chart-Hoehe in Pixeln
    int font_size = MathMax(10, InpAmpelFontSize);  // Minimale Schriftgroesse: 10
    int light_size = font_size + 10;  // Groesse des farbigen Quadrats
    int box_height = 60;   // Hoehe der Hintergrund-Box
    int box_width = 400;   // Breite der Hintergrund-Box
    int x_start = 15;      // Abstand vom linken Rand
    int y_top = chart_h - box_height - 25;  // Position von oben (= unten im Chart)

    // --- Objekt 1: Hintergrund-Box (schwarzer Kasten mit farbigem Rand) ---
    string bg_name = pfx + "BG";
    if (ObjectFind(0, bg_name) < 0)
        ObjectCreate(0, bg_name, OBJ_RECTANGLE_LABEL, 0, 0, 0);  // OBJ_RECTANGLE_LABEL = Pixel-basiertes Rechteck

    ObjectSetInteger(0, bg_name, OBJPROP_CORNER, CORNER_LEFT_UPPER);
    ObjectSetInteger(0, bg_name, OBJPROP_XDISTANCE, x_start - 8);     // Links mit etwas Padding
    ObjectSetInteger(0, bg_name, OBJPROP_YDISTANCE, y_top - 5);       // Oben mit etwas Padding
    ObjectSetInteger(0, bg_name, OBJPROP_XSIZE, box_width);           // Breite in Pixeln
    ObjectSetInteger(0, bg_name, OBJPROP_YSIZE, box_height);          // Hoehe in Pixeln
    ObjectSetInteger(0, bg_name, OBJPROP_BGCOLOR, ColorToARGB(clrBlack, 190)); // Schwarz, 75% deckend
    ObjectSetInteger(0, bg_name, OBJPROP_COLOR, ampel_color);         // Rand = Ampelfarbe
    ObjectSetInteger(0, bg_name, OBJPROP_BORDER_TYPE, BORDER_FLAT);   // Flacher Rand
    ObjectSetInteger(0, bg_name, OBJPROP_BACK, false);                // VOR den Kerzen (sichtbar)
    ObjectSetInteger(0, bg_name, OBJPROP_SELECTABLE, false);
    ObjectSetInteger(0, bg_name, OBJPROP_HIDDEN, true);               // Nicht in Objektliste

    // --- Objekt 2: Farbiges Quadrat (das "Ampel-Licht") ---
    // Ein kleines Quadrat links in der Box, eingefaerbt in der Ampelfarbe
    string light_name = pfx + "LIGHT";
    if (ObjectFind(0, light_name) < 0)
        ObjectCreate(0, light_name, OBJ_RECTANGLE_LABEL, 0, 0, 0);

    ObjectSetInteger(0, light_name, OBJPROP_CORNER, CORNER_LEFT_UPPER);
    ObjectSetInteger(0, light_name, OBJPROP_XDISTANCE, x_start);
    ObjectSetInteger(0, light_name, OBJPROP_YDISTANCE, y_top + (box_height - light_size) / 2 - 5); // Vertikal zentriert
    ObjectSetInteger(0, light_name, OBJPROP_XSIZE, light_size);
    ObjectSetInteger(0, light_name, OBJPROP_YSIZE, light_size);
    ObjectSetInteger(0, light_name, OBJPROP_BGCOLOR, ampel_color);  // Hintergrundfarbe = Ampelfarbe
    ObjectSetInteger(0, light_name, OBJPROP_COLOR, ampel_color);    // Randfarbe = Ampelfarbe
    ObjectSetInteger(0, light_name, OBJPROP_BORDER_TYPE, BORDER_FLAT);
    ObjectSetInteger(0, light_name, OBJPROP_BACK, false);
    ObjectSetInteger(0, light_name, OBJPROP_SELECTABLE, false);
    ObjectSetInteger(0, light_name, OBJPROP_HIDDEN, true);

    // --- Objekt 3: Hauptlabel (z.B. "GRUEN: HANDELN") ---
    // Grosser Text rechts neben dem Ampel-Licht
    string label_name = pfx + "LABEL";
    if (ObjectFind(0, label_name) < 0)
        ObjectCreate(0, label_name, OBJ_LABEL, 0, 0, 0);

    ObjectSetInteger(0, label_name, OBJPROP_CORNER, CORNER_LEFT_UPPER);
    ObjectSetInteger(0, label_name, OBJPROP_ANCHOR, ANCHOR_LEFT_UPPER);
    ObjectSetInteger(0, label_name, OBJPROP_XDISTANCE, x_start + light_size + 12);  // Rechts neben dem Licht
    ObjectSetInteger(0, label_name, OBJPROP_YDISTANCE, y_top + 4);
    ObjectSetString(0, label_name, OBJPROP_FONT, "Consolas");
    ObjectSetInteger(0, label_name, OBJPROP_FONTSIZE, font_size);
    ObjectSetInteger(0, label_name, OBJPROP_COLOR, ampel_color);  // Farbe = Ampelfarbe
    ObjectSetString(0, label_name, OBJPROP_TEXT, ampel_label);    // z.B. "GELB: BEOBACHTEN"
    ObjectSetInteger(0, label_name, OBJPROP_SELECTABLE, false);
    ObjectSetInteger(0, label_name, OBJPROP_HIDDEN, true);

    // --- Objekt 4: Detail-Zeile (kleinerer Text unter dem Hauptlabel) ---
    // z.B. "Long | Prob=67% | Signal aktiv!" oder "Dir=Kein | Prob=37% - naechsten Candle abwarten"
    string detail_name = pfx + "DETAIL";
    if (ObjectFind(0, detail_name) < 0)
        ObjectCreate(0, detail_name, OBJ_LABEL, 0, 0, 0);

    ObjectSetInteger(0, detail_name, OBJPROP_CORNER, CORNER_LEFT_UPPER);
    ObjectSetInteger(0, detail_name, OBJPROP_ANCHOR, ANCHOR_LEFT_UPPER);
    ObjectSetInteger(0, detail_name, OBJPROP_XDISTANCE, x_start + light_size + 12);
    ObjectSetInteger(0, detail_name, OBJPROP_YDISTANCE, y_top + font_size + 14);  // Unterhalb des Hauptlabels
    ObjectSetString(0, detail_name, OBJPROP_FONT, "Consolas");
    ObjectSetInteger(0, detail_name, OBJPROP_FONTSIZE, font_size - 2);  // Etwas kleiner
    ObjectSetInteger(0, detail_name, OBJPROP_COLOR, clrSilver);         // Grau fuer Details
    ObjectSetString(0, detail_name, OBJPROP_TEXT, ampel_detail);
    ObjectSetInteger(0, detail_name, OBJPROP_SELECTABLE, false);
    ObjectSetInteger(0, detail_name, OBJPROP_HIDDEN, true);
}

// ============================================================
// 13. DASHBOARD-TEXT-RENDERING
// ============================================================
// Das Dashboard zeigt alle Infos als Text oben links im Chart.
// Es gibt zwei Modi:
//   1) Comment()-Modus: MT5 Standard-Kommentar (einfach aber klein)
//   2) Label-Modus: Eigene OBJ_LABEL-Objekte (groesser, mit Hintergrund)
//
// DrawDashboardLabel() wird im Label-Modus verwendet und zeichnet:
//   - Einen halbtransparenten Hintergrund-Kasten
//   - Jede Text-Zeile als eigenes OBJ_LABEL (weil MQL5 kein mehrzeiliges Label kann)

void DrawDashboardLabel(const string text)
{
    string text_prefix = "PYML_DASH_TEXT_";  // Prefix fuer Text-Zeilen
    string bg_name = "PYML_DASH_BG";         // Hintergrund-Objekt
    int font_size = MathMax(8, InpDashboardFontSize);
    int x = 14;   // X-Position (vom linken Rand)
    int y = 20;   // Y-Position (vom oberen Rand)
    int pad = 14;  // Padding um den Text herum

    // Text in einzelne Zeilen aufteilen (getrennt durch \n)
    string lines[];
    int line_count = StringSplit(text, '\n', lines);
    if (line_count <= 0)
        line_count = 1;

    // Maximale Zeilenlaenge ermitteln (fuer Hintergrund-Breite)
    int max_chars = 0;
    for (int i = 0; i < line_count; i++)
    {
        int len = StringLen(lines[i]);
        if (len > max_chars)
            max_chars = len;
    }

    // Textflaeche berechnen (Consolas = Monospace, jedes Zeichen gleich breit)
    int char_width = (int)(font_size * 0.72);           // Geschaetzte Zeichenbreite
    int line_height_est = font_size + 10;                // Geschaetzte Zeilenhoehe
    int text_width = max_chars * char_width + 2 * pad;   // Gesamtbreite
    int text_height = line_count * line_height_est + 2 * pad; // Gesamthoehe

    // Hintergrund mindestens 45% der Chart-Breite und 40% der Chart-Hoehe
    int chart_w = (int)ChartGetInteger(0, CHART_WIDTH_IN_PIXELS);
    int chart_h = (int)ChartGetInteger(0, CHART_HEIGHT_IN_PIXELS);
    int width = MathMax(text_width, (int)(chart_w * 0.45));
    int height = MathMax(text_height, (int)(chart_h * 0.40));
    width = MathMax(width, 300);    // Mindestens 300px breit
    height = MathMax(height, 250);  // Mindestens 250px hoch

    // Hintergrund-Box zeichnen (halbtransparent, konfigurierbar)
    if (InpDashboardTextBackground)
    {
        if (ObjectFind(0, bg_name) < 0)
            ObjectCreate(0, bg_name, OBJ_RECTANGLE_LABEL, 0, 0, 0);

        // Alpha-Transparenz: 0 = voll transparent, 255 = voll deckend
        uint alpha_color = ColorToARGB(InpDashboardBgColor, (uchar)MathMax(0, MathMin(255, InpDashboardBgAlpha)));
        ObjectSetInteger(0, bg_name, OBJPROP_XDISTANCE, x - pad);   // Links mit Padding
        ObjectSetInteger(0, bg_name, OBJPROP_YDISTANCE, y - pad);   // Oben mit Padding
        ObjectSetInteger(0, bg_name, OBJPROP_XSIZE, width);          // Breite
        ObjectSetInteger(0, bg_name, OBJPROP_YSIZE, height);         // Hoehe
        ObjectSetInteger(0, bg_name, OBJPROP_CORNER, CORNER_LEFT_UPPER);
        ObjectSetInteger(0, bg_name, OBJPROP_COLOR, clrDimGray);     // Randfarbe
        ObjectSetInteger(0, bg_name, OBJPROP_BGCOLOR, alpha_color);  // Hintergrundfarbe mit Transparenz
        ObjectSetInteger(0, bg_name, OBJPROP_BORDER_TYPE, BORDER_FLAT);
        ObjectSetInteger(0, bg_name, OBJPROP_BACK, false);           // VOR den Kerzen
        ObjectSetInteger(0, bg_name, OBJPROP_SELECTABLE, false);
        ObjectSetInteger(0, bg_name, OBJPROP_HIDDEN, true);
    }
    else
    {
        // Hintergrund deaktiviert → Objekt loeschen falls vorhanden
        if (ObjectFind(0, bg_name) >= 0)
            ObjectDelete(0, bg_name);
    }

    // Alte Text-Zeilen loeschen und neu aufbauen
    // (wichtig: bei wechselnder Zeilenanzahl muessen ueberschuessige Zeilen weg)
    DeleteObjPrefix(text_prefix);

    // Jede Zeile als eigenes OBJ_LABEL zeichnen (MQL5 kann kein mehrzeiliges Label)
    int line_height = font_size + 12;  // Vertikaler Abstand zwischen Zeilen
    for (int i = 0; i < line_count; i++)
    {
        string name = text_prefix + IntegerToString(i);  // z.B. "PYML_DASH_TEXT_0"
        if (ObjectFind(0, name) < 0)
            ObjectCreate(0, name, OBJ_LABEL, 0, 0, 0);

        ObjectSetInteger(0, name, OBJPROP_XDISTANCE, x);                    // X-Position
        ObjectSetInteger(0, name, OBJPROP_YDISTANCE, y + i * line_height);  // Y-Position (jede Zeile weiter unten)
        ObjectSetInteger(0, name, OBJPROP_CORNER, CORNER_LEFT_UPPER);
        ObjectSetInteger(0, name, OBJPROP_ANCHOR, ANCHOR_LEFT_UPPER);
        ObjectSetInteger(0, name, OBJPROP_SELECTABLE, false);
        ObjectSetInteger(0, name, OBJPROP_BACK, false);
        ObjectSetInteger(0, name, OBJPROP_HIDDEN, true);
        ObjectSetString(0, name, OBJPROP_FONT, "Consolas");                 // Monospace fuer gleichmaessige Breite
        ObjectSetInteger(0, name, OBJPROP_FONTSIZE, font_size);
        ObjectSetInteger(0, name, OBJPROP_COLOR, InpDashboardTextColor);    // Text-Farbe (konfigurierbar)
        ObjectSetString(0, name, OBJPROP_TEXT, lines[i]);                   // Zeilen-Text
    }
}

// ============================================================
// 14. HILFS-TEXT-FUNKTIONEN (Infos fuer das Dashboard)
// ============================================================
// Jede Funktion erzeugt einen formatierten Text-String fuer eine
// bestimmte Information, die im Dashboard angezeigt wird.

// CountdownText() – Zeigt an wie viele Minuten/Sekunden bis zum naechsten Signal-Update.
// Das Modell generiert Signale nur bei neuen Kerzen (z.B. alle 60 Min bei H1).
// Berechnung: Wie viele Sekunden seit der letzten vollen Stunde? Rest = Countdown.
string CountdownText()
{
    int signal_tf = MathMax(1, InpSignalTimeframeMinutes);  // Signal-Zeitrahmen in Minuten
    datetime now = TimeCurrent();         // Aktuelle Server-Zeit
    int sec_since_epoch = (int)now;       // Sekunden seit 1.1.1970
    int tf_seconds = signal_tf * 60;      // Zeitrahmen in Sekunden (z.B. H1 = 3600)
    int elapsed = sec_since_epoch % tf_seconds;  // Bereits verstrichene Zeit in der aktuellen Kerze
    int remaining = tf_seconds - elapsed;        // Verbleibende Zeit bis naechste Kerze
    int min_left = remaining / 60;        // Ganzzahlige Minuten
    int sec_left = remaining % 60;        // Restliche Sekunden
    return StringFormat("Update in %d:%02d", min_left, sec_left);  // z.B. "Update in 23:45"
}

// SpreadText() – Zeigt den aktuellen Spread in Pips an.
// Spread = Differenz zwischen Ask und Bid (Kosten fuer den Trader).
// Bei 5/3-stelligen Brokern: Points / 10 = Pips (z.B. 15 Points = 1.5 Pips)
// Warnung: "!" bei >2 Pips, "!!" bei >3 Pips (zu teuer zum Handeln)
string SpreadText()
{
    string sym = Symbol();
    double spread_points = (double)SymbolInfoInteger(sym, SYMBOL_SPREAD); // Spread in Points
    double point = SymbolInfoDouble(sym, SYMBOL_POINT);    // Kleinste Preisaenderung
    int digits = (int)SymbolInfoInteger(sym, SYMBOL_DIGITS); // Nachkommastellen

    // Points zu Pips umrechnen (bei 5/3-stelligen Brokern teilen durch 10)
    double spread_pips;
    if (digits == 5 || digits == 3)
        spread_pips = spread_points / 10.0;  // z.B. EURUSD: 5 Nachkommastellen
    else
        spread_pips = spread_points;  // z.B. USDJPY alt: 3 Nachkommastellen

    // Warnung bei hohem Spread (= teuer)
    string warn = "";
    if (spread_pips > 3.0)
        warn = " !!";   // Sehr hoher Spread – nicht handeln!
    else if (spread_pips > 2.0)
        warn = " !";    // Erhoehter Spread – Vorsicht

    return StringFormat("Spread: %.1f Pips%s", spread_pips, warn);
}

// AtrVolaText() – Zeigt ATR14 in Pips + Volatilitaets-Stufe an.
// ATR = Average True Range (durchschnittliche Schwankung pro H1-Kerze)
// Stufen: NIEDRIG (<30 Pips), NORMAL (30-70 Pips), HOCH (>70 Pips)
// Hilft bei der Einschaetzung ob der Markt gerade ruhig oder wild ist.
string AtrVolaText()
{
    double atr_val = 0.0;
    // ATR-Wert vom Indikator holen (immer H1, unabhaengig vom Chart-TF)
    if (!GetLatestBufferValue(g_h_atr14, atr_val) || atr_val <= 0)
        return "ATR: n/a";  // Indikator nicht verfuegbar

    // ATR-Wert in Pips umrechnen
    double point = SymbolInfoDouble(Symbol(), SYMBOL_POINT);
    int digits = (int)SymbolInfoInteger(Symbol(), SYMBOL_DIGITS);
    double atr_pips;
    if (digits == 5 || digits == 3)
        atr_pips = atr_val / (point * 10.0);  // 5/3-stelliger Broker
    else
        atr_pips = atr_val / point;

    // Volatilitaets-Stufe bestimmen
    string level;
    if (atr_pips < 30)
        level = "NIEDRIG";  // Ruhiger Markt (enge Ranges)
    else if (atr_pips < 70)
        level = "NORMAL";   // Normale Marktbedingungen
    else
        level = "HOCH";     // Hohe Volatilitaet (z.B. News-Events)

    return StringFormat("ATR: %.0f Pips (%s)", atr_pips, level);
}

// SessionText() – Zeigt die aktive Handels-Session basierend auf UTC-Stunde an.
// Wichtig: Verschiedene Sessions haben unterschiedliche Liquiditaet und Volatilitaet.
// Session-Zeiten (UTC):
//   Asia:   00:00 - 08:00 (ruhiger, enge Ranges)
//   London: 07:00 - 16:00 (hohe Liquiditaet)
//   NY:     12:00 - 21:00 (hohe Volatilitaet)
//   London+NY Overlap: 12:00-16:00 (hoechste Liquiditaet des Tages!)
string SessionText()
{
    MqlDateTime dt;
    TimeGMT(dt);    // Aktuelle UTC-Zeit holen
    int h = dt.hour; // Stunde (0-23)

    // Session-Zuordnung (Ueberlappungen beachten!)
    bool asia = (h >= 0 && h < 8);      // Asia-Session
    bool london = (h >= 7 && h < 16);   // London-Session
    bool ny = (h >= 12 && h < 21);      // New York-Session

    // Ueberlappungen zuerst pruefen (wichtigste Handelszeit)
    if (london && ny)
        return "Session: London+NY";    // Hoechste Liquiditaet!
    else if (london)
        return "Session: London";
    else if (ny)
        return "Session: New York";
    else if (asia)
        return "Session: Asia";
    else
        return "Session: Off-Hours";    // Wochenende oder spaeter Abend
}

// RegimeModusText() – Zeigt das aktuelle Markt-Regime und den Handels-Modus.
// Regime: Kommt aus dem Python-Modell (z.B. "Trending", "Range", "Volatile")
// Modus: "PAPER" (Papierhandel) oder "LIVE" (echtes Geld)
// Beispiel: "Regime: Trending | PAPER"
string RegimeModusText()
{
    string regime_name = "?";   // Fallback wenn kein Regime bekannt
    string modus = "PAPER";     // Fallback: immer Paper-Trading annehmen

    // Zuerst pruefen ob eins der Symbole zum aktuellen Chart passt
    if (StringCompare(g_snap1.symbol, Symbol()) == 0 && g_snap1.valid)
    {
        regime_name = g_snap1.regime_name;
        modus = g_snap1.modus;
    }
    else if (StringCompare(g_snap2.symbol, Symbol()) == 0 && g_snap2.valid)
    {
        regime_name = g_snap2.regime_name;
        modus = g_snap2.modus;
    }
    // Wenn kein Match: Symbol 1 als Fallback nehmen
    else if (g_snap1.valid)
    {
        regime_name = g_snap1.regime_name;
        modus = g_snap1.modus;
    }

    return StringFormat("Regime: %s | %s", regime_name, modus);
}

// ShortState() – Wandelt lange Status-Bezeichnungen in kurze Abkuerzungen.
// Wird im Kompaktmodus verwendet um Platz zu sparen.
// Beispiel: "LIVE_SIGNAL" → "SIGNAL", "LIVE_NO_SIGNAL" → "IDLE"
string ShortState(const string state)
{
    if (StringCompare(state, "LIVE_SIGNAL") == 0)     return "SIGNAL";  // Aktives Signal
    if (StringCompare(state, "LIVE_NO_SIGNAL") == 0)  return "IDLE";    // Kein Signal, aber verbunden
    if (StringCompare(state, "MISSING") == 0)          return "MISS";   // Datei fehlt
    if (StringCompare(state, "STALE") == 0)             return "STALE";  // Daten veraltet
    if (StringCompare(state, "NO_TS") == 0)             return "NO_TS";  // Kein Zeitstempel
    if (StringCompare(state, "WAITING_FOR_CSV") == 0)   return "WAIT";   // Warte auf CSV
    return state;  // Unbekannter Status → unveraendert zurueckgeben
}

// ============================================================
// 15. DASHBOARD-TEXT ZUSAMMENBAUEN
// ============================================================
// DrawDashboard() baut den gesamten Dashboard-Text zusammen.
// Es gibt zwei Modi:
//   KOMPAKTMODUS (InpDashboardCompactMode = true):
//     Kurze Zeilen, wenig Platz, ideal fuer kleine Charts
//   AUSFUEHRLICHER MODUS (InpDashboardCompactMode = false):
//     Alle Details, inkl. CSV-Pfad und Debug-Info

void DrawDashboard()
{
    // Status beider Symbole bestimmen
    string st1 = SnapshotState(g_snap1);           // z.B. "LIVE_SIGNAL"
    string st2 = SnapshotState(g_snap2);           // z.B. "LIVE_NO_SIGNAL"
    string overall = OverallState(st1, st2);       // z.B. "CONNECTED"
    string stale_src = InpUseSignalTimeframeForStale ? "SignalTF" : "ChartTF";
    string sep = "-------------------------";       // Trennlinie
    string dashboard_text;

    if (InpDashboardCompactMode)
    {
        // === KOMPAKTMODUS ===
        // Header: Gesamtstatus auf einen Blick
        string header = StringFormat("Live Dashboard | %s", overall);
        // Stale-Grenze und Countdown
        string stale_info = StringFormat(
            "Grenze: %dmin (%s)",
            EffectiveStaleMinutes(), stale_src);

        // Symbol 1: Status + Frische in einer Zeile, Details in der naechsten
        string s1a = StringFormat("%s | %s | %s",
            g_snap1.symbol, ShortState(st1),       // z.B. "USDCAD | IDLE"
            FreshnessText(g_snap1));                 // z.B. "OK (12 min)"
        string s1b = StringFormat("  %s | P=%.2f | R=%d",
            g_snap1.richtung, g_snap1.prob,         // z.B. "  Kein | P=0.37"
            (int)g_snap1.rows);                      // Anzahl Zeilen in CSV

        // Symbol 2: gleiches Format
        string s2a = StringFormat("%s | %s | %s",
            g_snap2.symbol, ShortState(st2),
            FreshnessText(g_snap2));
        string s2b = StringFormat("  %s | P=%.2f | R=%d",
            g_snap2.richtung, g_snap2.prob,
            (int)g_snap2.rows);

        // Zusatz-Infos: Spread + ATR + Regime + Session + Countdown
        string spread_atr = SpreadText() + " | " + AtrVolaText();
        string regime_session = RegimeModusText() + " | " + SessionText();
        string countdown = CountdownText();

        // Alles zusammenbauen mit Trennlinien
        dashboard_text = header + "\n" +
            stale_info + " | " + countdown + "\n" +
            sep + "\n" +
            s1a + "\n" + s1b + "\n" +
            s2a + "\n" + s2b + "\n" +
            sep + "\n" +
            spread_atr + "\n" +
            regime_session;

        // Debug: Historie-Zaehler anzeigen (nur wenn aktiviert)
        if (InpDebugHistoryInfo)
        {
            string hist = StringFormat(
                "Hist: %s=%d | %s=%d",
                InpSymbol1, g_hist_count_1,
                InpSymbol2, g_hist_count_2);
            dashboard_text += "\n" + sep + "\n" + hist;
        }
    }
    else
    {
        // === AUSFUEHRLICHER MODUS ===
        // Alle Details mit langen Bezeichnungen
        string header = "=== Live Signal Dashboard ===";
        string status_line = StringFormat(
            "STATUS=%s | STALE>%dmin (%s)",
            overall, EffectiveStaleMinutes(), stale_src);

        // Symbol 1: drei Zeilen (Status, Signal, Regime)
        string s1a = StringFormat("%s | %s | %s",
            g_snap1.symbol, st1, FreshnessText(g_snap1));
        string s1b = StringFormat("  Dir=%s | Prob=%.3f",
            g_snap1.richtung, g_snap1.prob);              // Volle Genauigkeit
        string s1c = StringFormat("  Regime=%s | Mode=%s | R=%d",
            g_snap1.regime_name, g_snap1.modus, (int)g_snap1.rows);

        // Symbol 2: gleiches Format
        string s2a = StringFormat("%s | %s | %s",
            g_snap2.symbol, st2, FreshnessText(g_snap2));
        string s2b = StringFormat("  Dir=%s | Prob=%.3f",
            g_snap2.richtung, g_snap2.prob);
        string s2c = StringFormat("  Regime=%s | Mode=%s | R=%d",
            g_snap2.regime_name, g_snap2.modus, (int)g_snap2.rows);

        // Dashboard zusammenbauen
        dashboard_text = header + "\n" + status_line + "\n" +
            sep + "\n" +
            s1a + "\n" + s1b + "\n" + s1c + "\n" +
            s2a + "\n" + s2b + "\n" + s2c;

        // Debug: Historie-Zaehler (nur wenn aktiviert)
        if (InpDebugHistoryInfo)
        {
            string hist = StringFormat(
                "History: %s=%d | %s=%d (max=%d)",
                InpSymbol1, g_hist_count_1,
                InpSymbol2, g_hist_count_2,
                InpMaxTradesOnChart);
            dashboard_text += "\n" + sep + "\n" + hist;
        }

        // CSV-Pfad-Hinweis (hilfreich beim Debugging)
        string hint = "CSV: " + ExpectedFolderText();
        dashboard_text += "\n" + hint;

        // Auch im ausfuehrlichen Modus: Spread, ATR, Session, Countdown
        string spread_atr_v = SpreadText() + " | " + AtrVolaText();
        string regime_session_v = RegimeModusText() + " | " + SessionText();
        string countdown_v = CountdownText();
        dashboard_text += "\n" + sep + "\n" +
            "Countdown: " + countdown_v + "\n" +
            spread_atr_v + "\n" +
            regime_session_v;
    }

    // === Text-Ausgabe: Label-Modus oder Comment()-Modus ===
    if (InpUseLargeDashboardText)
    {
        // Label-Modus: Grosse Schrift mit Hintergrund (empfohlen)
        Comment("");  // Standard-Kommentar loeschen
        DrawDashboardLabel(dashboard_text);  // Eigene Labels zeichnen
    }
    else
    {
        // Comment-Modus: MT5 Standard (klein, ohne Hintergrund)
        DeleteObjPrefix("PYML_DASH_");  // Eigene Labels loeschen
        Comment(dashboard_text);         // MT5-Kommentar verwenden
    }
}

// ============================================================
// 16. HAUPTSTEUERUNG (RefreshAll + MT5-Event-Handler)
// ============================================================
// RefreshAll() ist die zentrale Funktion die ALLES aktualisiert.
// Sie wird alle N Sekunden vom Timer aufgerufen und:
//   1. Liest beide CSV-Dateien neu ein
//   2. Protokolliert Datei-Status
//   3. Sendet Alerts bei neuen Signalen
//   4. Zaehlt Historie-Eintraege
//   5. Zeichnet Dashboard, Trades, EMAs und Ampel

void RefreshAll()
{
    // CSV-Dateinamen zusammenbauen
    string file_1 = BuildFileName(InpSymbol1);  // z.B. "USDCAD_live_trades.csv"
    string file_2 = BuildFileName(InpSymbol2);  // z.B. "USDJPY_live_trades.csv"

    // 1. CSV-Dateien lesen und in Snapshots parsen
    bool ok_1 = ReadLatestSnapshot(InpSymbol1, g_snap1);  // USDCAD
    bool ok_2 = ReadLatestSnapshot(InpSymbol2, g_snap2);  // USDJPY

    // 2. Datei-Status ins Log schreiben (Warnungen bei fehlenden Dateien)
    LogFileState(InpSymbol1, file_1, ok_1, g_missing_1, g_missing_log_ts_1);
    LogFileState(InpSymbol2, file_2, ok_2, g_missing_2, g_missing_log_ts_2);

    // 3. Alerts senden wenn neue Signale erkannt wurden
    MaybeAlert(g_snap1, g_last_alert_ts_1);
    MaybeAlert(g_snap2, g_last_alert_ts_2);

    // 4. Historie-Eintraege zaehlen (fuer Debug-Anzeige)
    g_hist_count_1 = CountRecentEntries(InpSymbol1, InpMaxTradesOnChart);
    g_hist_count_2 = CountRecentEntries(InpSymbol2, InpMaxTradesOnChart);

    // 5. Alles zeichnen
    DrawDashboard();              // Dashboard-Text aktualisieren
    DrawTradeOnChart(g_snap1);    // Trade-Levels fuer Symbol 1
    DrawTradeOnChart(g_snap2);    // Trade-Levels fuer Symbol 2
    DrawEntryHistory(g_snap1);    // Historie-Pfeile fuer Symbol 1
    DrawEntryHistory(g_snap2);    // Historie-Pfeile fuer Symbol 2
    DrawEmaGuides();              // EMA-Linien + Struktur-Label
    DrawAmpel();                  // Ampel-System aktualisieren
}

// OnInit() – Wird EINMAL aufgerufen wenn der Indikator auf den Chart gezogen wird.
// Initialisiert Timer, Snapshots und technische Indikatoren.
// Gibt INIT_SUCCEEDED zurueck wenn alles OK ist.
int OnInit()
{
    // Timer starten: RefreshAll() wird alle N Sekunden aufgerufen
    g_refresh_seconds = (InpRefreshSeconds < 1) ? 1 : InpRefreshSeconds;
    EventSetTimer(g_refresh_seconds);  // z.B. alle 5 Sekunden

    // Snapshots initialisieren (leer, noch keine Daten)
    g_snap1.symbol = InpSymbol1;  // z.B. "USDCAD"
    g_snap2.symbol = InpSymbol2;  // z.B. "USDJPY"
    g_snap1.valid = false;
    g_snap2.valid = false;

    // Technische Indikatoren erstellen (EMA, RSI, ATR)
    SetupTechnicalOverlay();

    Print("[Dashboard] Initialisiert. Warte auf CSV-Signale...");
    RefreshAll();  // Sofort einmal alles laden und anzeigen
    return (INIT_SUCCEEDED);  // Erfolg melden
}

// OnDeinit() – Wird aufgerufen wenn der Indikator entfernt wird.
// Raumt alles auf: Timer stoppen, Kommentar loeschen, Indikatoren freigeben,
// alle Chart-Objekte mit "PYML_" Prefix loeschen.
void OnDeinit(const int reason)
{
    EventKillTimer();               // Timer stoppen (kein RefreshAll mehr)
    Comment("");                     // Standard-Kommentar loeschen
    ReleaseTechnicalOverlay();       // EMA/RSI/ATR Handles freigeben
    DeleteObjPrefix("PYML_");        // ALLE unsere Chart-Objekte loeschen
    ChartRedraw(0);                  // Chart neu zeichnen (sauber hinterlassen)
}

// OnTick() – Wird bei jedem neuen Tick (Preisaenderung) aufgerufen.
// Wir nutzen dies NICHT – unser Refresh erfolgt ueber den Timer.
// Grund: Ticks kommen zu haeufig (hunderte pro Sekunde bei hoher Volatilitaet)
void OnTick()
{
    // Keine Tick-Logik noetig. Refresh erfolgt ueber Timer.
}

// OnTimer() – Wird alle g_refresh_seconds Sekunden aufgerufen.
// Startet den kompletten Refresh-Zyklus (CSV lesen + Dashboard zeichnen).
void OnTimer()
{
    RefreshAll();  // Alles neu laden und zeichnen
}
