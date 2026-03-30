// ╔══════════════════════════════════════════════════════════════╗
// ║   PythonSignalExecutor.mq5 – MT5 Expert Advisor v1.10        ║
// ║                                                              ║
// ║   ZWECK:                                                     ║
// ║   Dieser EA liest die Signal-CSV-Dateien, die von Python     ║
// ║   (live_trader.py) geschrieben werden, und fuehrt die        ║
// ║   Trades TATSAECHLICH im MT5-Terminal aus.                   ║
// ║                                                              ║
// ║   ERGEBNIS:                                                  ║
// ║   Alle Trades erscheinen im MT5-Terminal:                    ║
// ║     - Toolbox → Handel (offene Positionen)                   ║
// ║     - Toolbox → Historie (geschlossene Trades)               ║
// ║     - Toolbox → Orders (pending/ausgefuehrte Orders)         ║
// ║     - Magic Number korrekt zugewiesen (20260101)             ║
// ║     - SL/TP nativ von MT5 verwaltet                          ║
// ║                                                              ║
// ║   WIE ES FUNKTIONIERT:                                       ║
// ║   1. Python schreibt {SYMBOL}_signals.csv ins Common/Files   ║
// ║   2. Dieser EA liest die CSV alle N Sekunden (Timer)         ║
// ║   3. Erkennt neue Long/Short-Signale (Zeitstempel-Vergleich) ║
// ║   4. Fuehrt Market-Order mit SL/TP aus                       ║
// ║   5. Position wird von MT5 nativ verwaltet (SL/TP-Ausloesung)║
// ║                                                              ║
// ║   SICHERHEIT:                                                ║
// ║   - Maximal 1 Position pro Symbol (kein Stacking)            ║
// ║   - SL ist PFLICHT – kein Trade ohne Stop-Loss               ║
// ║   - Gegenposition wird geschlossen bevor neue oeffnet        ║
// ║   - DryRun-Modus verfuegbar (nur loggen, nicht handeln)      ║
// ║                                                              ║
// ║   INSTALLATION:                                              ║
// ║   1. Datei nach MQL5/Experts/ kopieren                       ║
// ║   2. Im MetaEditor kompilieren (F7)                          ║
// ║   3. Im Chart auf USDCAD oder USDJPY ziehen                  ║
// ║   4. Algo-Trading Button aktivieren (oben in MT5)            ║
// ║   5. Sicherstellen dass live_trader.py laeuft und CSV        ║
// ║      ins Common/Files-Verzeichnis schreibt                   ║
// ╚══════════════════════════════════════════════════════════════╝
#property strict
#property version "1.10"
#property description "Fuehrt Python-ML-Signale als echte MT5-Trades aus."
#property description "Liest CSV aus Common/Files und sendet Market-Orders."
#property description "Alle Trades erscheinen in Toolbox: Handel, Historie, Orders."
#property description "v1.10: +CSV-Trade-Log +OnTradeTransaction +Chart-Comment"


// ============================================================
// 1. EINGABE-PARAMETER
// ============================================================

// --- Grundeinstellungen ---
input string InpSymbol = "";                    // Symbol (leer = Chart-Symbol verwenden)
input string InpFileSuffix = "_signals.csv";    // CSV-Dateiendung (Standard: _signals.csv)
input bool   InpUseCommonFiles = true;          // true = Common/Files, false = MQL5/Files
input int    InpRefreshSeconds = 5;             // CSV-Abfrage-Intervall in Sekunden

// --- Trade-Ausfuehrung ---
input double InpLot = 0.01;                     // Lot-Groesse pro Trade (0.01 = Micro-Lot)
input int    InpMagicNumber = 20260101;          // Magic Number (muss mit Python uebereinstimmen!)
input int    InpSlippage = 20;                   // Maximaler Slippage in Points
input string InpComment = "ML-Python-Signal";    // Order-Kommentar im MT5-Terminal
input bool   InpDryRun = false;                  // true = nur loggen, KEINE echten Trades

// --- Positionsmanagement ---
input bool   InpCloseOpposite = true;           // Gegenposition schliessen bevor neue oeffnet?
input bool   InpUseCsvPrices = true;             // SL/TP aus CSV verwenden (empfohlen!)
input double InpFallbackSlPct = 0.003;           // Fallback-SL in Dezimal (0.3%) wenn CSV kein SL hat
input double InpFallbackTpPct = 0.006;           // Fallback-TP in Dezimal (0.6%) wenn CSV kein TP hat

// --- Sicherheit ---
input int    InpMaxStaleMinutes = 25;            // Signal aelter als X Minuten → ignorieren (M15-Default + Puffer)
input bool   InpRequireSlTp = true;              // Trade nur wenn SL UND TP vorhanden sind

// --- CSV-Trade-Log (fuer Auswertung) ---
input bool   InpWriteTradeLog = true;            // true = CSV-Datei mit allen Trades schreiben
input string InpTradeLogSuffix = "_ea_trades.csv"; // Dateiendung fuer Trade-Log
input bool   InpShowChartComment = true;          // true = Status-Anzeige direkt im Chart
input int    InpChartCommentMode = 1;             // 0 = Compact (kleiner Chart), 1 = Detailed (Standard)


// ============================================================
// 2. GLOBALE VARIABLEN
// ============================================================

// Effektives Symbol (bestimmt in OnInit)
string g_symbol = "";

// Zeitstempel des zuletzt verarbeiteten Signals (verhindert Doppel-Trades)
datetime g_last_processed_signal_time = 0;

// Richtung des zuletzt verarbeiteten Signals ("Long", "Short", "Kein")
string g_last_processed_direction = "";

// Zaehler fuer ausgefuehrte Trades
int g_trade_count = 0;

// Zaehler fuer geschlossene Trades (SL/TP/manuell)
int g_close_count = 0;

// Kumulierter PnL aus geschlossenen Trades
double g_total_pnl = 0.0;

// Letztes Signal (fuer Chart-Comment)
string g_last_signal_info = "Warte auf Signal...";
datetime g_last_signal_time_display = 0;

// Ticket der aktuell offenen Position (fuer OnTradeTransaction-Tracking)
ulong g_open_position_ticket = 0;

// Letzte Fehlermeldung (fuer Journal-Deduplizierung)
string g_last_error_msg = "";


// ============================================================
// 3. SIGNAL-DATENSTRUKTUR (identisch mit Dashboard)
// ============================================================

struct SignalData
{
    datetime ts;            // Zeitstempel des Signals
    string   richtung;      // "Long", "Short" oder "Kein"
    double   prob;          // Modell-Wahrscheinlichkeit (0.0 bis 1.0)
    int      regime;        // Marktphase (0=Seitwaerts, 1=Auf, 2=Ab, 3=HighVola)
    double   entry_price;   // Entry-Preis aus Python
    double   sl_price;      // Stop-Loss-Preis aus Python
    double   tp_price;      // Take-Profit-Preis aus Python
    string   modus;         // "PAPER" oder "LIVE"
    bool     valid;         // true wenn erfolgreich geparst
};


// ============================================================
// 4. INITIALISIERUNG
// ============================================================

int OnInit()
{
    // Symbol bestimmen: Input-Parameter oder Chart-Symbol
    g_symbol = InpSymbol;
    StringTrimLeft(g_symbol);
    StringTrimRight(g_symbol);
    if (StringLen(g_symbol) == 0)
        g_symbol = _Symbol;  // Chart-Symbol verwenden
    StringToUpper(g_symbol);

    // Symbol pruefen
    if (!SymbolSelect(g_symbol, true))
    {
        PrintFormat("[%s] Symbol nicht verfuegbar im Market Watch!", g_symbol);
        return INIT_FAILED;
    }

    // Timer starten (CSV wird alle N Sekunden gelesen)
    if (!EventSetTimer(InpRefreshSeconds))
    {
        PrintFormat("[%s] Timer konnte nicht gestartet werden!", g_symbol);
        return INIT_FAILED;
    }

    // Status ausgeben
    PrintFormat("═══════════════════════════════════════════════════");
    PrintFormat("PythonSignalExecutor v1.10 gestartet");
    PrintFormat("Symbol:       %s", g_symbol);
    PrintFormat("Magic:        %d", InpMagicNumber);
    PrintFormat("Lot:          %.2f", InpLot);
    PrintFormat("Modus:        %s", InpDryRun ? "DRY-RUN (nur loggen)" : "LIVE-AUSFUEHRUNG");
    PrintFormat("CSV-Quelle:   %s%s", g_symbol, InpFileSuffix);
    PrintFormat("CSV-Ordner:   %s", InpUseCommonFiles ? "Common/Files" : "MQL5/Files");
    PrintFormat("Intervall:    %d Sekunden", InpRefreshSeconds);
    PrintFormat("Stale-Limit:  %d Minuten", InpMaxStaleMinutes);
    PrintFormat("Trade-Log:    %s", InpWriteTradeLog ? "aktiv (CSV)" : "aus");
    PrintFormat("Chart-Status: %s", InpShowChartComment ? "aktiv" : "aus");
    PrintFormat("Chart-Modus:  %s", InpChartCommentMode == 0 ? "Compact" : "Detailed");
    PrintFormat("═══════════════════════════════════════════════════");

    // CSV-Trade-Log Header schreiben (falls Datei neu)
    if (InpWriteTradeLog)
        WriteTradeLogHeader();

    // Initiale Chart-Anzeige
    if (InpShowChartComment)
        UpdateChartComment("Initialisiert – warte auf Signal...");

    return INIT_SUCCEEDED;
}


// ============================================================
// 5. DEINITIALISIERUNG
// ============================================================

void OnDeinit(const int reason)
{
    EventKillTimer();
    Comment("");  // Chart-Kommentar entfernen
    PrintFormat("[%s] PythonSignalExecutor gestoppt | Trades: %d geoeffnet, %d geschlossen | PnL: %.2f | Grund: %d",
                g_symbol, g_trade_count, g_close_count, g_total_pnl, reason);
}


// ============================================================
// 6. TIMER-EVENT (Haupt-Logik, wird alle N Sekunden ausgefuehrt)
// ============================================================

void OnTimer()
{
    // CSV-Signal lesen
    SignalData signal;
    if (!ReadLatestSignal(g_symbol, signal))
        return;  // Datei fehlt oder ist leer → naechsten Timer abwarten

    // Signal veraltet?
    if (IsSignalStale(signal.ts))
    {
        // Stale-Warnung nur einmal pro Minute loggen
        static datetime last_stale_warn = 0;
        if (TimeCurrent() - last_stale_warn > 60)
        {
            PrintFormat("[%s] Signal veraltet (%s) – ignoriert. Stale-Limit: %d Min.",
                        g_symbol, TimeToString(signal.ts), InpMaxStaleMinutes);
            last_stale_warn = TimeCurrent();
        }
        return;
    }

    // Kein aktives Signal ("Kein")?
    if (StringCompare(signal.richtung, "Kein") == 0)
        return;  // Heartbeat – nichts zu tun

    // Signal bereits verarbeitet? (gleicher Zeitstempel UND gleiche Richtung)
    if (signal.ts == g_last_processed_signal_time &&
        StringCompare(signal.richtung, g_last_processed_direction) == 0)
        return;  // Schon ausgefuehrt → ueberspringen

    // NEUES SIGNAL ERKANNT!
    PrintFormat("[%s] *** NEUES SIGNAL: %s | Prob=%.1f%% | Regime=%d | Entry=%.5f | SL=%.5f | TP=%.5f ***",
                g_symbol, signal.richtung, signal.prob * 100.0, signal.regime,
                signal.entry_price, signal.sl_price, signal.tp_price);

    // Trade ausfuehren
    bool success = ExecuteSignal(signal);

    if (success)
    {
        // Signal als verarbeitet markieren
        g_last_processed_signal_time = signal.ts;
        g_last_processed_direction = signal.richtung;
        g_trade_count++;

        g_last_signal_info = StringFormat("%s | Prob=%.0f%% | #%d",
                                          signal.richtung, signal.prob * 100.0, g_trade_count);
        g_last_signal_time_display = signal.ts;

        PrintFormat("[%s] Trade #%d erfolgreich ausgefuehrt: %s",
                    g_symbol, g_trade_count, signal.richtung);
    }
    else
    {
        PrintFormat("[%s] Trade-Ausfuehrung fehlgeschlagen: %s", g_symbol, signal.richtung);
    }

    // Chart-Comment aktualisieren (bei jedem Timer-Durchlauf)
    if (InpShowChartComment)
        UpdateChartComment(g_last_signal_info);
}


// ============================================================
// 7. TICK-EVENT (fuer EA-Aktivierung noetig, Logik laeuft per Timer)
// ============================================================

void OnTick()
{
    // Tick-Event wird nicht aktiv genutzt.
    // Die gesamte Logik laeuft timer-basiert (OnTimer).
    // OnTick ist nur noetig damit MT5 den EA als "aktiv" betrachtet.
}


// ============================================================
// 8. CSV LESEN
// ============================================================

// ReadCsvLine() – Liest eine komplette CSV-Zeile in ein String-Array.
void ReadCsvLine(const int handle, string &fields[])
{
    ArrayResize(fields, 0);
    if (FileIsEnding(handle))
        return;

    while (true)
    {
        string value = FileReadString(handle);
        int n = ArraySize(fields);
        ArrayResize(fields, n + 1);
        fields[n] = value;

        if (FileIsLineEnding(handle) || FileIsEnding(handle))
            break;
    }
}

// FindHeaderIndex() – Findet den Index einer Spalte im Header (case-insensitive).
int FindHeaderIndex(string &headers[], const string key)
{
    string key_l = key;
    StringToLower(key_l);

    for (int i = 0; i < ArraySize(headers); i++)
    {
        string header_l = headers[i];
        StringTrimLeft(header_l);
        StringTrimRight(header_l);
        StringToLower(header_l);
        if (StringCompare(header_l, key_l) == 0)
            return i;
    }
    return -1;
}

// SafeField() – Sicherer Array-Zugriff mit Fallback.
string SafeField(string &arr[], const int idx, const string fallback)
{
    if (idx < 0 || idx >= ArraySize(arr))
        return fallback;
    return arr[idx];
}

// ParseCsvTime() – Robustes Zeit-Parsing fuer Python-CSV-Formate.
datetime ParseCsvTime(const string raw)
{
    string ts = raw;
    StringTrimLeft(ts);
    StringTrimRight(ts);
    if (StringLen(ts) == 0)
        return 0;

    // ISO-8601 und Python-Formate normalisieren
    StringReplace(ts, "T", " ");
    StringReplace(ts, "Z", "");
    StringReplace(ts, "-", ".");

    // Offset abschneiden (z.B. "+00:00")
    int plus_pos = StringFind(ts, "+");
    if (plus_pos > 0)
        ts = StringSubstr(ts, 0, plus_pos);

    return StringToTime(ts);
}

// ParseSignalDirection() – Normalisiert Signal-Richtung.
string ParseSignalDirection(const string raw)
{
    string v = raw;
    StringToLower(v);
    StringTrimLeft(v);
    StringTrimRight(v);

    if (v == "long" || v == "buy" || v == "2")
        return "Long";
    if (v == "short" || v == "sell" || v == "-1")
        return "Short";
    return "Kein";
}

// ReadLatestSignal() – Liest das letzte Signal aus der CSV-Datei.
bool ReadLatestSignal(const string symbol, SignalData &out)
{
    // Dateiflags setzen
    int flags = FILE_READ | FILE_CSV | FILE_ANSI;
    if (InpUseCommonFiles)
        flags |= FILE_COMMON;

    string file_name = symbol + InpFileSuffix;
    int h = FileOpen(file_name, flags, ',');
    if (h == INVALID_HANDLE)
    {
        // Fehler nur selten loggen (alle 5 Minuten)
        string err = StringFormat("[%s] CSV nicht gefunden: %s", symbol, file_name);
        if (StringCompare(err, g_last_error_msg) != 0)
        {
            Print(err);
            g_last_error_msg = err;
        }
        out.valid = false;
        return false;
    }

    // Header lesen
    string headers[];
    ReadCsvLine(h, headers);
    if (ArraySize(headers) == 0)
    {
        FileClose(h);
        out.valid = false;
        return false;
    }

    // Spaltenindizes finden
    int idx_time     = FindHeaderIndex(headers, "time");
    int idx_richtung = FindHeaderIndex(headers, "richtung");
    int idx_signal   = FindHeaderIndex(headers, "signal");
    int idx_prob     = FindHeaderIndex(headers, "prob");
    int idx_regime   = FindHeaderIndex(headers, "regime");
    int idx_entry    = FindHeaderIndex(headers, "entry_price");
    int idx_sl       = FindHeaderIndex(headers, "sl_price");
    int idx_tp       = FindHeaderIndex(headers, "tp_price");
    int idx_modus    = FindHeaderIndex(headers, "modus");

    // Alle Zeilen lesen, nur die letzte behalten
    string last[];
    long row_count = 0;

    while (!FileIsEnding(h))
    {
        string cols[];
        ReadCsvLine(h, cols);
        if (ArraySize(cols) == 0)
            continue;
        row_count++;
        ArrayCopy(last, cols);
    }
    FileClose(h);

    if (row_count == 0 || ArraySize(last) == 0)
    {
        out.valid = false;
        return false;
    }

    // Letzte Zeile parsen
    out.ts          = ParseCsvTime(SafeField(last, idx_time, ""));
    out.richtung    = ParseSignalDirection(SafeField(last, idx_richtung, ""));
    if (StringCompare(out.richtung, "Kein") == 0)
        out.richtung = ParseSignalDirection(SafeField(last, idx_signal, "0"));
    out.prob        = StringToDouble(SafeField(last, idx_prob, "0"));
    out.regime      = (int)StringToInteger(SafeField(last, idx_regime, "-1"));
    out.entry_price = StringToDouble(SafeField(last, idx_entry, "0"));
    out.sl_price    = StringToDouble(SafeField(last, idx_sl, "0"));
    out.tp_price    = StringToDouble(SafeField(last, idx_tp, "0"));
    out.modus       = SafeField(last, idx_modus, "PAPER");
    out.valid       = (out.ts > 0);

    // Fehlermeldung zuruecksetzen nach erfolgreichem Lesen
    g_last_error_msg = "";

    return out.valid;
}


// ============================================================
// 9. SIGNAL-AUSFUEHRUNG
// ============================================================

// IsSignalStale() – Prueft ob das Signal zu alt ist.
bool IsSignalStale(const datetime signal_time)
{
    if (signal_time == 0)
        return true;

    // MT5 TimeCurrent() kann Broker-Zeit sein → nutze TimeGMT fuer UTC-Vergleich
    datetime now_gmt = TimeGMT();
    long diff_sec = (long)(now_gmt - signal_time);

    // Signal liegt in der Zukunft (Uhrzeit-Offset) → als frisch betrachten
    if (diff_sec < 0)
        return false;

    return (diff_sec > InpMaxStaleMinutes * 60);
}

// HasOpenPosition() – Prueft ob bereits eine Position mit unserer Magic Number offen ist.
bool HasOpenPosition(const string symbol)
{
    for (int i = PositionsTotal() - 1; i >= 0; i--)
    {
        ulong ticket = PositionGetTicket(i);
        if (ticket == 0)
            continue;
        if (PositionGetString(POSITION_SYMBOL) == symbol &&
            PositionGetInteger(POSITION_MAGIC) == InpMagicNumber)
            return true;
    }
    return false;
}

// GetOpenPositionType() – Gibt den Typ der offenen Position zurueck.
// Returns: POSITION_TYPE_BUY, POSITION_TYPE_SELL, oder -1 wenn keine offen.
int GetOpenPositionType(const string symbol)
{
    for (int i = PositionsTotal() - 1; i >= 0; i--)
    {
        ulong ticket = PositionGetTicket(i);
        if (ticket == 0)
            continue;
        if (PositionGetString(POSITION_SYMBOL) == symbol &&
            PositionGetInteger(POSITION_MAGIC) == InpMagicNumber)
            return (int)PositionGetInteger(POSITION_TYPE);
    }
    return -1;
}

// GetOpenPositionTicket() – Gibt das Ticket der offenen Position zurueck (oder 0 wenn keine).
ulong GetOpenPositionTicket(const string symbol)
{
    for (int i = PositionsTotal() - 1; i >= 0; i--)
    {
        ulong ticket = PositionGetTicket(i);
        if (ticket == 0)
            continue;
        if (PositionGetString(POSITION_SYMBOL) == symbol &&
            PositionGetInteger(POSITION_MAGIC) == InpMagicNumber)
            return ticket;
    }
    return 0;
}

// ClosePosition() – Schliesst eine offene Position (fuer Gegensignal).
bool ClosePosition(const string symbol)
{
    for (int i = PositionsTotal() - 1; i >= 0; i--)
    {
        ulong ticket = PositionGetTicket(i);
        if (ticket == 0)
            continue;
        if (PositionGetString(POSITION_SYMBOL) != symbol)
            continue;
        if (PositionGetInteger(POSITION_MAGIC) != InpMagicNumber)
            continue;

        // Position gefunden → schliessen
        int pos_type = (int)PositionGetInteger(POSITION_TYPE);
        double volume = PositionGetDouble(POSITION_VOLUME);

        MqlTick tick;
        if (!SymbolInfoTick(symbol, tick))
        {
            PrintFormat("[%s] Tick-Daten nicht verfuegbar fuer Close!", symbol);
            return false;
        }

        MqlTradeRequest request = {};
        MqlTradeResult result = {};

        request.action   = TRADE_ACTION_DEAL;
        request.symbol   = symbol;
        request.volume   = volume;
        request.position = ticket;
        request.magic    = InpMagicNumber;
        request.comment  = "ML-Close-Gegensignal";
        request.deviation = (ulong)InpSlippage;

        if (pos_type == POSITION_TYPE_BUY)
        {
            request.type  = ORDER_TYPE_SELL;
            request.price = tick.bid;
        }
        else
        {
            request.type  = ORDER_TYPE_BUY;
            request.price = tick.ask;
        }

        // Filling-Mode bestimmen
        request.type_filling = GetFillingMode(symbol);
        request.type_time    = ORDER_TIME_GTC;

        if (!OrderSend(request, result))
        {
            PrintFormat("[%s] Position-Close FEHLGESCHLAGEN: Error=%d | Retcode=%u | %s",
                        symbol, GetLastError(), result.retcode, result.comment);
            return false;
        }

        if (result.retcode == TRADE_RETCODE_DONE || result.retcode == TRADE_RETCODE_DONE_PARTIAL)
        {
            PrintFormat("[%s] Position %llu geschlossen (Gegensignal) | Preis=%.5f",
                        symbol, ticket, result.price);
            return true;
        }
        else
        {
            PrintFormat("[%s] Position-Close unerwarteter Retcode: %u | %s",
                        symbol, result.retcode, result.comment);
            return false;
        }
    }
    return false;  // Keine passende Position gefunden
}

// GetFillingMode() – Ermittelt den unterstuetzten Filling-Modus des Brokers.
ENUM_ORDER_TYPE_FILLING GetFillingMode(const string symbol)
{
    long filling_mode = 0;
    if (!SymbolInfoInteger(symbol, SYMBOL_FILLING_MODE, filling_mode))
        return ORDER_FILLING_IOC;  // Fallback

    // IOC bevorzugen (am weitesten verbreitet)
    if ((filling_mode & SYMBOL_FILLING_IOC) != 0)
        return ORDER_FILLING_IOC;

    // FOK als zweitbeste Option
    if ((filling_mode & SYMBOL_FILLING_FOK) != 0)
        return ORDER_FILLING_FOK;

    // Return oder Book → ORDER_FILLING_RETURN
    return ORDER_FILLING_RETURN;
}

// ExecuteSignal() – Fuehrt das erkannte Signal als echten Trade aus.
bool ExecuteSignal(const SignalData &signal)
{
    // SL/TP bestimmen: aus CSV oder Fallback berechnen
    double sl_price = signal.sl_price;
    double tp_price = signal.tp_price;

    // Aktuelle Tick-Daten holen
    MqlTick tick;
    if (!SymbolInfoTick(g_symbol, tick))
    {
        PrintFormat("[%s] Tick-Daten nicht verfuegbar!", g_symbol);
        return false;
    }

    // Symbol-Info fuer Digits und Normalisierung
    int digits = (int)SymbolInfoInteger(g_symbol, SYMBOL_DIGITS);
    double point = SymbolInfoDouble(g_symbol, SYMBOL_POINT);

    // Bestimme Trade-Richtung und Preis
    ENUM_ORDER_TYPE order_type;
    double entry_price;
    bool is_long = (StringCompare(signal.richtung, "Long") == 0);

    if (is_long)
    {
        order_type = ORDER_TYPE_BUY;
        entry_price = tick.ask;

        // Fallback SL/TP berechnen wenn CSV keine liefert
        if (sl_price <= 0.0 || !InpUseCsvPrices)
            sl_price = NormalizeDouble(entry_price * (1.0 - InpFallbackSlPct), digits);
        if (tp_price <= 0.0 || !InpUseCsvPrices)
            tp_price = NormalizeDouble(entry_price * (1.0 + InpFallbackTpPct), digits);
    }
    else  // Short
    {
        order_type = ORDER_TYPE_SELL;
        entry_price = tick.bid;

        if (sl_price <= 0.0 || !InpUseCsvPrices)
            sl_price = NormalizeDouble(entry_price * (1.0 + InpFallbackSlPct), digits);
        if (tp_price <= 0.0 || !InpUseCsvPrices)
            tp_price = NormalizeDouble(entry_price * (1.0 - InpFallbackTpPct), digits);
    }

    // SL/TP normalisieren
    sl_price = NormalizeDouble(sl_price, digits);
    tp_price = NormalizeDouble(tp_price, digits);

    // Sicherheitscheck: SL muss vorhanden sein
    if (InpRequireSlTp && (sl_price <= 0.0 || tp_price <= 0.0))
    {
        PrintFormat("[%s] Trade ABGEBROCHEN: SL=%.5f TP=%.5f – SL/TP fehlt!",
                    g_symbol, sl_price, tp_price);
        return false;
    }

    // SL-Abstand validieren (muss mindestens SYMBOL_TRADE_STOPS_LEVEL entfernt sein)
    long stops_level = 0;
    SymbolInfoInteger(g_symbol, SYMBOL_TRADE_STOPS_LEVEL, stops_level);
    double min_stop_distance = stops_level * point;

    if (is_long)
    {
        if (entry_price - sl_price < min_stop_distance)
        {
            sl_price = NormalizeDouble(entry_price - min_stop_distance - point, digits);
            PrintFormat("[%s] SL angepasst (Broker Minimum): %.5f", g_symbol, sl_price);
        }
        if (tp_price - entry_price < min_stop_distance)
        {
            tp_price = NormalizeDouble(entry_price + min_stop_distance + point, digits);
            PrintFormat("[%s] TP angepasst (Broker Minimum): %.5f", g_symbol, tp_price);
        }
    }
    else
    {
        if (sl_price - entry_price < min_stop_distance)
        {
            sl_price = NormalizeDouble(entry_price + min_stop_distance + point, digits);
            PrintFormat("[%s] SL angepasst (Broker Minimum): %.5f", g_symbol, sl_price);
        }
        if (entry_price - tp_price < min_stop_distance)
        {
            tp_price = NormalizeDouble(entry_price - min_stop_distance - point, digits);
            PrintFormat("[%s] TP angepasst (Broker Minimum): %.5f", g_symbol, tp_price);
        }
    }

    // Bereits offene Position pruefen
    if (HasOpenPosition(g_symbol))
    {
        int current_type = GetOpenPositionType(g_symbol);
        bool same_direction = (is_long && current_type == POSITION_TYPE_BUY) ||
                              (!is_long && current_type == POSITION_TYPE_SELL);

        if (same_direction)
        {
            PrintFormat("[%s] Bereits offene %s-Position – kein neuer Trade",
                        g_symbol, is_long ? "Long" : "Short");
            // Signal trotzdem als verarbeitet markieren (Duplikat verhindern)
            g_last_processed_signal_time = signal.ts;
            g_last_processed_direction = signal.richtung;
            return false;
        }

        // Gegenposition: erst schliessen
        if (InpCloseOpposite)
        {
            PrintFormat("[%s] Gegensignal erkannt – schliesse bestehende Position", g_symbol);
            if (!ClosePosition(g_symbol))
            {
                PrintFormat("[%s] Konnte bestehende Position nicht schliessen – Trade abgebrochen",
                            g_symbol);
                return false;
            }
            Sleep(500);  // Kurz warten bis Position geschlossen ist
        }
        else
        {
            PrintFormat("[%s] Offene Position vorhanden, Gegenposition-Close deaktiviert – kein Trade",
                        g_symbol);
            return false;
        }
    }

    // ---- DRY-RUN-MODUS: Nur loggen, nicht ausfuehren ----
    if (InpDryRun)
    {
        PrintFormat("[%s] [DRY-RUN] Wuerde ausfuehren: %s | Lot=%.2f | Entry=%.5f | SL=%.5f | TP=%.5f",
                    g_symbol, signal.richtung, InpLot, entry_price, sl_price, tp_price);
        return true;  // Als "erfolgreich" markieren damit Signal nicht wiederholt wird
    }

    // ---- ECHTE ORDER SENDEN ----
    MqlTradeRequest request = {};
    MqlTradeCheckResult check_result = {};
    MqlTradeResult result = {};

    request.action       = TRADE_ACTION_DEAL;
    request.symbol       = g_symbol;
    request.volume       = InpLot;
    request.type         = order_type;
    request.price        = entry_price;
    request.sl           = sl_price;
    request.tp           = tp_price;
    request.deviation    = (ulong)InpSlippage;
    request.magic        = InpMagicNumber;
    request.comment      = InpComment;
    request.type_filling = GetFillingMode(g_symbol);
    request.type_time    = ORDER_TIME_GTC;

    // Order Pre-Check (optional, gibt detailliertere Fehlermeldung)
    if (!OrderCheck(request, check_result))
    {
        PrintFormat("[%s] OrderCheck FEHLGESCHLAGEN: Retcode=%u | %s",
                    g_symbol, check_result.retcode, check_result.comment);
        // Trotzdem OrderSend versuchen (OrderCheck kann false-negatives liefern)
    }

    // Order senden
    if (!OrderSend(request, result))
    {
        PrintFormat("[%s] OrderSend FEHLGESCHLAGEN: Error=%d | Retcode=%u | %s",
                    g_symbol, GetLastError(), result.retcode, result.comment);
        return false;
    }

    // Ergebnis pruefen
    if (result.retcode == TRADE_RETCODE_DONE || result.retcode == TRADE_RETCODE_DONE_PARTIAL)
    {
        PrintFormat("[%s] ✓ ORDER AUSGEFUEHRT: %s | Lot=%.2f | Preis=%.5f | SL=%.5f | TP=%.5f | "
                    "Deal=%llu | Order=%llu | Magic=%d",
                    g_symbol, signal.richtung, InpLot, result.price, sl_price, tp_price,
                    result.deal, result.order, InpMagicNumber);

        // Position-Ticket robust merken (nicht Order-Ticket)
        g_open_position_ticket = GetOpenPositionTicket(g_symbol);

        // CSV-Trade-Log schreiben (OPEN-Event)
        if (InpWriteTradeLog)
            WriteTradeLogEntry("OPEN", signal.richtung, InpLot, result.price,
                               sl_price, tp_price, 0.0, (long)result.deal,
                               (long)g_open_position_ticket, signal.prob, signal.regime);

        return true;
    }
    else
    {
        PrintFormat("[%s] Order unerwarteter Retcode: %u | %s",
                    g_symbol, result.retcode, result.comment);
        return false;
    }
}


// ============================================================
// 10. TRADE-TRANSACTION-HANDLER (faengt SL/TP/Close-Events ab)
// ============================================================

void OnTradeTransaction(
    const MqlTradeTransaction &trans,
    const MqlTradeRequest &request,
    const MqlTradeResult &result)
{
    // Nur Deal-Events interessieren uns (tatsaechliche Ausfuehrungen)
    if (trans.type != TRADE_TRANSACTION_DEAL_ADD)
        return;

    // Nur unsere Magic Number beachten
    // Deal-Details laden
    if (!HistoryDealSelect(trans.deal))
        return;

    long deal_magic = HistoryDealGetInteger(trans.deal, DEAL_MAGIC);
    if (deal_magic != InpMagicNumber)
        return;

    string deal_symbol = HistoryDealGetString(trans.deal, DEAL_SYMBOL);
    if (deal_symbol != g_symbol)
        return;

    long deal_entry = HistoryDealGetInteger(trans.deal, DEAL_ENTRY);
    long deal_type  = HistoryDealGetInteger(trans.deal, DEAL_TYPE);

    // DEAL_ENTRY_OUT = Position wurde geschlossen (SL, TP, manuell, Gegensignal)
    if (deal_entry == DEAL_ENTRY_OUT || deal_entry == DEAL_ENTRY_INOUT)
    {
        double deal_profit     = HistoryDealGetDouble(trans.deal, DEAL_PROFIT);
        double deal_commission = HistoryDealGetDouble(trans.deal, DEAL_COMMISSION);
        double deal_swap       = HistoryDealGetDouble(trans.deal, DEAL_SWAP);
        double deal_volume     = HistoryDealGetDouble(trans.deal, DEAL_VOLUME);
        double deal_price      = HistoryDealGetDouble(trans.deal, DEAL_PRICE);
        long   deal_position   = HistoryDealGetInteger(trans.deal, DEAL_POSITION_ID);
        double net_pnl         = deal_profit + deal_commission + deal_swap;
        string deal_comment    = HistoryDealGetString(trans.deal, DEAL_COMMENT);
        string deal_comment_l  = deal_comment;
        StringToLower(deal_comment_l);

        // Bestimme Close-Grund aus dem Kommentar
        string close_grund = "manuell";
        if (StringFind(deal_comment_l, "sl") >= 0)
            close_grund = "SL";
        else if (StringFind(deal_comment_l, "tp") >= 0)
            close_grund = "TP";
        else if (StringFind(deal_comment_l, "gegensignal") >= 0)
            close_grund = "Gegensignal";
        else if (StringFind(deal_comment_l, "kill") >= 0)
            close_grund = "Kill-Switch";

        g_close_count++;
        g_total_pnl += net_pnl;

        string richtung_str = (deal_type == DEAL_TYPE_BUY) ? "CLOSE-Short" : "CLOSE-Long";

        PrintFormat("[%s] CLOSE erkannt: %s | Preis=%.5f | PnL=%.2f (Brutto=%.2f, Komm=%.2f, Swap=%.2f) | "
                    "Grund=%s | Deal=%llu | Position=%lld | Gesamt-PnL=%.2f",
                    g_symbol, richtung_str, deal_price, net_pnl, deal_profit,
                    deal_commission, deal_swap, close_grund, trans.deal,
                    deal_position, g_total_pnl);

        // CSV-Trade-Log schreiben (CLOSE-Event)
        if (InpWriteTradeLog)
            WriteTradeLogEntry("CLOSE", richtung_str, deal_volume, deal_price,
                               0.0, 0.0, net_pnl, (long)trans.deal,
                               deal_position, 0.0, -1);

        // Position-Ticket zuruecksetzen
        if ((ulong)deal_position == g_open_position_ticket)
            g_open_position_ticket = 0;

        // Chart-Comment aktualisieren
        if (InpShowChartComment)
        {
            string close_info = StringFormat("Letzter Close: %s | PnL=%.2f | Gesamt=%.2f",
                                             close_grund, net_pnl, g_total_pnl);
            UpdateChartComment(close_info);
        }
    }
}


// ============================================================
// 11. CSV-TRADE-LOG (strukturierte Ausgabe fuer Auswertung)
// ============================================================

// WriteTradeLogHeader() – Schreibt den CSV-Header wenn die Datei neu ist.
void WriteTradeLogHeader()
{
    string file_name = g_symbol + InpTradeLogSuffix;
    int flags = FILE_READ | FILE_CSV | FILE_ANSI;
    if (InpUseCommonFiles)
        flags |= FILE_COMMON;

    // Pruefen ob Datei bereits existiert und Daten hat
    int h_check = FileOpen(file_name, flags, ',');
    bool file_has_data = false;
    if (h_check != INVALID_HANDLE)
    {
        file_has_data = (FileSize(h_check) > 10);  // Mehr als nur ein paar Bytes
        FileClose(h_check);
    }

    if (file_has_data)
        return;  // Header existiert bereits

    // Header schreiben
    int flags_write = FILE_WRITE | FILE_CSV | FILE_ANSI;
    if (InpUseCommonFiles)
        flags_write |= FILE_COMMON;

    int h = FileOpen(file_name, flags_write, ',');
    if (h == INVALID_HANDLE)
    {
        PrintFormat("[%s] Trade-Log CSV konnte nicht erstellt werden!", g_symbol);
        return;
    }

    FileWrite(h, "time", "symbol", "event", "richtung", "lot", "price",
              "sl", "tp", "pnl_net", "deal_ticket", "position_ticket",
              "prob", "regime", "magic", "trade_nr", "close_nr", "total_pnl");
    FileClose(h);
}

// WriteTradeLogEntry() – Schreibt eine Zeile ins Trade-Log (OPEN oder CLOSE).
void WriteTradeLogEntry(
    const string event,         // "OPEN" oder "CLOSE"
    const string richtung,      // "Long", "Short", "CLOSE-Long", "CLOSE-Short"
    const double lot,
    const double price,
    const double sl,
    const double tp,
    const double pnl_net,
    const long deal_ticket,
    const long position_ticket,
    const double prob,          // Modell-Wahrscheinlichkeit (nur bei OPEN)
    const int regime)           // Marktregime (nur bei OPEN, -1 bei CLOSE)
{
    string file_name = g_symbol + InpTradeLogSuffix;
    int flags = FILE_READ | FILE_WRITE | FILE_CSV | FILE_ANSI;
    if (InpUseCommonFiles)
        flags |= FILE_COMMON;

    int h = FileOpen(file_name, flags, ',');
    if (h == INVALID_HANDLE)
    {
        PrintFormat("[%s] Trade-Log CSV nicht beschreibbar!", g_symbol);
        return;
    }

    // Ans Ende der Datei springen
    FileSeek(h, 0, SEEK_END);

    // Zeitstempel im gleichen Format wie Python (YYYY-MM-DD HH:MM:SS UTC)
    string time_str = TimeToString(TimeGMT(), TIME_DATE | TIME_SECONDS);
    StringReplace(time_str, ".", "-");  // YYYY.MM.DD → YYYY-MM-DD

    FileWrite(h, time_str, g_symbol, event, richtung,
              DoubleToString(lot, 2),
              DoubleToString(price, (int)SymbolInfoInteger(g_symbol, SYMBOL_DIGITS)),
              DoubleToString(sl, (int)SymbolInfoInteger(g_symbol, SYMBOL_DIGITS)),
              DoubleToString(tp, (int)SymbolInfoInteger(g_symbol, SYMBOL_DIGITS)),
              DoubleToString(pnl_net, 2),
              IntegerToString(deal_ticket),
              IntegerToString(position_ticket),
              DoubleToString(prob, 4),
              IntegerToString(regime),
              IntegerToString(InpMagicNumber),
              IntegerToString(g_trade_count),
              IntegerToString(g_close_count),
              DoubleToString(g_total_pnl, 2));
    FileClose(h);
}


// ============================================================
// 12. CHART-KOMMENTAR (Live-Status-Anzeige im Chart)
// ============================================================

// UpdateChartComment() – Zeigt aktuellen EA-Status direkt im Chart an.
void UpdateChartComment(const string extra_info)
{
    // Aktuellen Positionsstatus fuer die Anzeige bestimmen.
    string pos_status = "Keine";
    if (HasOpenPosition(g_symbol))
    {
        int ptype = GetOpenPositionType(g_symbol);
        pos_status = (ptype == POSITION_TYPE_BUY) ? "LONG offen" : "SHORT offen";
    }

    // UTC-Zeit nur einmal erzeugen und fuer beide Layouts verwenden.
    string update_utc = TimeToString(TimeGMT(), TIME_DATE | TIME_SECONDS);
    string comment = "";

    // Compact-Modus: kurze, platzsparende Darstellung fuer kleine Charts.
    if (InpChartCommentMode == 0)
    {
        comment = StringFormat(
            "PyExec v1.10 | %s | %s\n"
            "Pos: %s\n"
            "T: %d/%d | PnL: %.2f\n"
            "Event: %s\n"
            "UTC: %s",
            g_symbol,
            InpDryRun ? "DRY" : "LIVE",
            pos_status,
            g_trade_count, g_close_count,
            g_total_pnl,
            extra_info,
            update_utc
        );
    }
    // Detailed-Modus: vollstaendige, gut gruppierte Darstellung.
    else
    {
        comment = StringFormat(
            "=== PythonSignalExecutor v1.10 ===\n"
            "Symbol / Modus : %s | %s\n"
            "Position       : %s\n"
            "--------------------------------\n"
            "Trades         : Open %d | Close %d\n"
            "PnL (netto)    : %.2f\n"
            "Letztes Event  : %s\n"
            "Update (UTC)   : %s\n"
            "================================",
            g_symbol,
            InpDryRun ? "DRY-RUN" : "LIVE",
            pos_status,
            g_trade_count, g_close_count,
            g_total_pnl,
            extra_info,
            update_utc
        );
    }

    // Kommentar im Chart anzeigen (ersetzt immer den vorherigen Kommentar).
    Comment(comment);
}
