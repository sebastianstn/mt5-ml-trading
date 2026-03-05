#property strict
#property version "2.20"
#property description "MT5 Dashboard fuer Python Live-Signale (USDCAD/USDJPY)"
#property description "Liest CSV aus Common/Files und zeigt Status, Alerts + Chart-Zeichnungen."
#property description "v2.20: Ampel-System (Gruen/Gelb/Rot) fuer schnelle Marktbewertung"
#property indicator_chart_window

input string InpSymbol1 = "USDCAD";
input string InpSymbol2 = "USDJPY";
input string InpFileSuffix = "_live_trades.csv";
input bool InpUseCommonFiles = true;
input int InpRefreshSeconds = 5;
input bool InpEnableAlerts = true;
// Freshness-Schwelle in Minuten:
// H1: 70 (empfohlen), M30: 40, M15: 25, M5: 15
// Hintergrund: Bei H1 kommt ein neues Update typischerweise nur pro neuer Kerze.
input int InpStaleMinutes = 70;
input bool InpAutoStaleByTimeframe = true; // Auto: Schwelle aus Zeitrahmen + Buffer berechnen
input bool InpUseSignalTimeframeForStale = true; // true: Signal-TF nutzen (empfohlen fuer H1-Signale auf M5-Chart)
input int InpSignalTimeframeMinutes = 60;  // Signal-TF in Minuten (H1=60, M30=30, M15=15, M5=5)
input int InpStaleBufferMinutes = 10;      // Buffer fuer Auto-Stale (z.B. H1=60+10 => 70)
input int InpMissingFileLogEverySec = 300;
input bool InpDrawTrades = true;           // Chart-Zeichnungen fuer Trades
input bool InpDrawEntryHistory = true;     // Mehrere historische Entry-Punkte einzeichnen
input int InpMaxTradesOnChart = 10;        // Max. Trades gleichzeitig im Chart
input bool InpDrawTechnicalOverlay = true; // EMA20/EMA50/EMA200 + RSI grafisch einblenden
input bool InpDebugHistoryInfo = true;     // Zeigt im Dashboard die Anzahl gefundener History-Entries
input bool InpUseLargeDashboardText = true; // Dashboard-Text als großes Label statt kleinem Comment
input int InpDashboardFontSize = 10;        // Schriftgröße Dashboard-Text (empfohlen: 9-13)
input color InpDashboardTextColor = clrWhite; // Schriftfarbe Dashboard-Text
input bool InpDashboardTextBackground = true; // Hintergrundkasten hinter Dashboard-Text anzeigen
input color InpDashboardBgColor = clrBlack;   // Hintergrundfarbe
input int InpDashboardBgAlpha = 140;          // Transparenz 0..255 (0=transparent, 255=deckend)
input bool InpDashboardCompactMode = true;    // Kürzere Dashboard-Zeilen für mehr Platz
input color InpColorLong = clrDodgerBlue;  // Farbe fuer Long-Trades
input color InpColorShort = clrOrangeRed;  // Farbe fuer Short-Trades
input color InpColorSL = clrRed;           // Farbe fuer Stop-Loss
input color InpColorTP = clrLimeGreen;     // Farbe fuer Take-Profit
input bool InpDrawEmaGuides = true;        // Zusätzliche EMA-Hilfslinien (visueller Unterschied)
input color InpEma20GuideColor = clrDodgerBlue; // EMA20: Blau
input color InpEma50GuideColor = clrRed;        // EMA50: Rot
input color InpEma200GuideColor = clrLimeGreen; // EMA200: Grün
input int InpEmaGuideWidth = 2;                 // Linienbreite der EMA-Guides
input bool InpShowEmaStructureLabel = true;     // Textlabel mit EMA-Struktur anzeigen
input bool InpShowAmpel = true;                  // Ampel-System im Chart anzeigen (Gruen/Gelb/Rot)
input int InpAmpelFontSize = 13;                 // Ampel Schriftgroesse (empfohlen: 11-15)

struct SignalSnapshot
{
    string symbol;
    datetime ts;
    string richtung;
    double prob;
    int regime;
    string regime_name;
    bool paper;
    string modus;
    long rows;
    bool valid;
    double entry_price;
    double sl_price;
    double tp_price;
};

struct EntryPoint
{
    datetime ts;
    string richtung;
    double prob;
    string regime_name;
    double entry_price;
};

SignalSnapshot g_snap1;
SignalSnapshot g_snap2;
datetime g_last_alert_ts_1 = 0;
datetime g_last_alert_ts_2 = 0;
int g_refresh_seconds = 5;
bool g_missing_1 = false;
bool g_missing_2 = false;
datetime g_missing_log_ts_1 = 0;
datetime g_missing_log_ts_2 = 0;
int g_h_ema20 = INVALID_HANDLE;
int g_h_ema50 = INVALID_HANDLE;
int g_h_ema200 = INVALID_HANDLE;
int g_h_rsi14 = INVALID_HANDLE;
int g_hist_count_1 = 0;
int g_hist_count_2 = 0;

int StaleBaseMinutes()
{
    if (InpUseSignalTimeframeForStale)
    {
        if (InpSignalTimeframeMinutes > 0)
            return InpSignalTimeframeMinutes;
        return 60;
    }

    int sec = PeriodSeconds(PERIOD_CURRENT);
    if (sec <= 0)
        return 60;

    int tf_minutes = sec / 60;
    if (tf_minutes <= 0)
        return 60;

    return tf_minutes;
}

int EffectiveStaleMinutes()
{
    if (!InpAutoStaleByTimeframe)
        return InpStaleMinutes;

    int base_minutes = StaleBaseMinutes();
    if (base_minutes <= 0)
        return InpStaleMinutes;

    int buffer = MathMax(InpStaleBufferMinutes, 0);
    return MathMax(5, base_minutes + buffer);
}

void SetupTechnicalOverlay()
{
    if (!InpDrawTechnicalOverlay)
        return;

    g_h_ema20 = iMA(Symbol(), PERIOD_CURRENT, 20, 0, MODE_EMA, PRICE_CLOSE);
    g_h_ema50 = iMA(Symbol(), PERIOD_CURRENT, 50, 0, MODE_EMA, PRICE_CLOSE);
    g_h_ema200 = iMA(Symbol(), PERIOD_CURRENT, 200, 0, MODE_EMA, PRICE_CLOSE);
    g_h_rsi14 = iRSI(Symbol(), PERIOD_CURRENT, 14, PRICE_CLOSE);

    if (g_h_ema20 != INVALID_HANDLE)
        ChartIndicatorAdd(0, 0, g_h_ema20);
    else
        Print("[Dashboard] Warnung: EMA20 konnte nicht erstellt werden.");

    if (g_h_ema50 != INVALID_HANDLE)
        ChartIndicatorAdd(0, 0, g_h_ema50);
    else
        Print("[Dashboard] Warnung: EMA50 konnte nicht erstellt werden.");

    if (g_h_ema200 != INVALID_HANDLE)
        ChartIndicatorAdd(0, 0, g_h_ema200);
    else
        Print("[Dashboard] Warnung: EMA200 konnte nicht erstellt werden.");

    if (g_h_rsi14 != INVALID_HANDLE)
        ChartIndicatorAdd(0, 1, g_h_rsi14);
    else
        Print("[Dashboard] Warnung: RSI14 konnte nicht erstellt werden.");
}

void ReleaseTechnicalOverlay()
{
    if (g_h_ema20 != INVALID_HANDLE)
        IndicatorRelease(g_h_ema20);
    if (g_h_ema50 != INVALID_HANDLE)
        IndicatorRelease(g_h_ema50);
    if (g_h_ema200 != INVALID_HANDLE)
        IndicatorRelease(g_h_ema200);
    if (g_h_rsi14 != INVALID_HANDLE)
        IndicatorRelease(g_h_rsi14);

    g_h_ema20 = INVALID_HANDLE;
    g_h_ema50 = INVALID_HANDLE;
    g_h_ema200 = INVALID_HANDLE;
    g_h_rsi14 = INVALID_HANDLE;
}

int FindHeaderIndex(string &headers[], const string key)
{
    string key_l = key;
    StringToLower(key_l);

    for (int i = 0; i < ArraySize(headers); i++)
    {
        string header_l = headers[i];
        StringToLower(header_l);

        if (StringCompare(header_l, key_l) == 0)
            return i;
    }
    return -1;
}

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

string BuildFileName(const string symbol)
{
    return symbol + InpFileSuffix;
}

bool ParseBool(const string raw)
{
    string v = raw;
    StringToLower(v);
    return (v == "1" || v == "true" || v == "ja" || v == "yes");
}

string SafeField(string &arr[], const int idx, const string fallback)
{
    if (idx < 0 || idx >= ArraySize(arr))
        return fallback;
    return arr[idx];
}

bool ReadLatestSnapshot(const string symbol, SignalSnapshot &out)
{
    int flags = FILE_READ | FILE_CSV | FILE_ANSI;
    if (InpUseCommonFiles)
        flags |= FILE_COMMON;

    string file_name = BuildFileName(symbol);
    int h = FileOpen(file_name, flags, ',');
    if (h == INVALID_HANDLE)
    {
        out.symbol = symbol;
        out.valid = false;
        return false;
    }

    string headers[];
    ReadCsvLine(h, headers);
    if (ArraySize(headers) == 0)
    {
        FileClose(h);
        out.symbol = symbol;
        out.valid = false;
        return false;
    }

    int idx_time = FindHeaderIndex(headers, "time");
    int idx_richtung = FindHeaderIndex(headers, "richtung");
    int idx_prob = FindHeaderIndex(headers, "prob");
    int idx_regime = FindHeaderIndex(headers, "regime");
    int idx_regime_nm = FindHeaderIndex(headers, "regime_name");
    int idx_paper = FindHeaderIndex(headers, "paper_trading");
    int idx_modus = FindHeaderIndex(headers, "modus");
    int idx_entry = FindHeaderIndex(headers, "entry_price");
    int idx_sl = FindHeaderIndex(headers, "sl_price");
    int idx_tp = FindHeaderIndex(headers, "tp_price");

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
        out.symbol = symbol;
        out.valid = false;
        return false;
    }

    out.symbol = symbol;
    out.rows = row_count;
    out.valid = true;
    out.ts = StringToTime(SafeField(last, idx_time, "1970.01.01 00:00"));
    out.richtung = SafeField(last, idx_richtung, "Unbekannt");
    out.prob = StringToDouble(SafeField(last, idx_prob, "0"));
    out.regime = (int)StringToInteger(SafeField(last, idx_regime, "-1"));
    out.regime_name = SafeField(last, idx_regime_nm, "?");
    out.paper = ParseBool(SafeField(last, idx_paper, "true"));
    out.modus = SafeField(last, idx_modus, "PAPER");
    out.entry_price = StringToDouble(SafeField(last, idx_entry, "0"));
    out.sl_price = StringToDouble(SafeField(last, idx_sl, "0"));
    out.tp_price = StringToDouble(SafeField(last, idx_tp, "0"));

    return true;
}

string ExpectedFolderText()
{
    string base;
    if (InpUseCommonFiles)
        base = TerminalInfoString(TERMINAL_COMMONDATA_PATH) + "\\Files\\";
    else
        base = TerminalInfoString(TERMINAL_DATA_PATH) + "\\MQL5\\Files\\";
    return base;
}

void LogFileState(
    const string symbol,
    const string file_name,
    const bool ok,
    bool &was_missing,
    datetime &last_log_ts)
{
    datetime now = TimeCurrent();

    if (!ok)
    {
        bool should_log = (!was_missing);
        if (!should_log && InpMissingFileLogEverySec > 0)
            should_log = ((now - last_log_ts) >= InpMissingFileLogEverySec);

        if (should_log)
        {
            PrintFormat(
                "[Dashboard] %s Datei fehlt: %s | Erwartet in: %s",
                symbol,
                file_name,
                ExpectedFolderText());
            last_log_ts = now;
        }

        was_missing = true;
        return;
    }

    if (was_missing)
        PrintFormat("[Dashboard] %s Datei erkannt: %s", symbol, file_name);

    was_missing = false;
}

string FreshnessText(const SignalSnapshot &snap)
{
    if (!snap.valid || snap.ts <= 0)
        return "keine Daten";

    int stale_minutes = EffectiveStaleMinutes();
    int age_min = (int)((TimeCurrent() - snap.ts) / 60);
    string tag = (age_min <= stale_minutes) ? "OK" : "STALE";
    return StringFormat("%s (%d min)", tag, age_min);
}

string SnapshotState(const SignalSnapshot &snap)
{
    if (!snap.valid)
        return "MISSING";

    if (snap.ts <= 0)
        return "NO_TS";

    int stale_minutes = EffectiveStaleMinutes();
    int age_min = (int)((TimeCurrent() - snap.ts) / 60);
    if (age_min > stale_minutes)
        return "STALE";

    if (StringCompare(snap.richtung, "Kein") == 0)
        return "LIVE_NO_SIGNAL";

    return "LIVE_SIGNAL";
}

bool ReadRecentEntries(const string symbol, EntryPoint &entries[], const int max_entries)
{
    ArrayResize(entries, 0);
    if (max_entries <= 0)
        return false;

    int flags = FILE_READ | FILE_CSV | FILE_ANSI;
    if (InpUseCommonFiles)
        flags |= FILE_COMMON;

    string file_name = BuildFileName(symbol);
    int h = FileOpen(file_name, flags, ',');
    if (h == INVALID_HANDLE)
        return false;

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

    while (!FileIsEnding(h))
    {
        string cols[];
        ReadCsvLine(h, cols);
        if (ArraySize(cols) == 0)
            continue;

        string dir = SafeField(cols, idx_richtung, "Kein");
        bool is_long = (StringCompare(dir, "Long") == 0);
        bool is_short = (StringCompare(dir, "Short") == 0);
        if (!is_long && !is_short)
            continue;

        datetime ts = StringToTime(SafeField(cols, idx_time, "1970.01.01 00:00"));
        double entry = StringToDouble(SafeField(cols, idx_entry, "0"));
        if (ts <= 0 || entry <= 0)
            continue;

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
        return false;

    if (total <= max_entries)
        return true;

    EntryPoint last_entries[];
    ArrayResize(last_entries, max_entries);
    int start = total - max_entries;
    for (int i = 0; i < max_entries; i++)
        last_entries[i] = entries[start + i];

    ArrayResize(entries, max_entries);
    for (int i = 0; i < max_entries; i++)
        entries[i] = last_entries[i];

    return true;
}

int CountRecentEntries(const string symbol, const int max_entries)
{
    EntryPoint entries[];
    if (!ReadRecentEntries(symbol, entries, max_entries))
        return 0;
    return ArraySize(entries);
}

string OverallState(const string st1, const string st2)
{
    bool ok1 = (StringFind(st1, "LIVE") == 0);
    bool ok2 = (StringFind(st2, "LIVE") == 0);

    if (ok1 && ok2)
        return "CONNECTED";

    if (StringCompare(st1, "MISSING") == 0 && StringCompare(st2, "MISSING") == 0)
        return "WAITING_FOR_CSV";

    if (StringCompare(st1, "STALE") == 0 || StringCompare(st2, "STALE") == 0)
        return "PARTIAL_STALE";

    return "PARTIAL";
}

void MaybeAlert(const SignalSnapshot &snap, datetime &last_alert_ts)
{
    if (!InpEnableAlerts || !snap.valid || snap.ts <= 0)
        return;

    // Nur bei neuen Zeilen und echten Signalen alerten.
    if (snap.ts <= last_alert_ts)
        return;

    if (StringCompare(snap.richtung, "Kein") == 0)
    {
        last_alert_ts = snap.ts;
        return;
    }

    string msg = StringFormat(
        "[MT5 Dashboard] %s | %s | Prob=%.3f | Regime=%s | %s",
        snap.symbol,
        snap.richtung,
        snap.prob,
        snap.regime_name,
        TimeToString(snap.ts, TIME_DATE | TIME_MINUTES));

    Alert(msg);
    Print(msg);
    last_alert_ts = snap.ts;
}

// ============================================================
// Chart-Zeichnungen: Entry-Pfeile, SL/TP-Linien
// ============================================================

void DeleteObjPrefix(const string prefix)
{
    // Alle Objekte mit diesem Prefix loeschen
    int total = ObjectsTotal(0, 0, -1);
    for (int i = total - 1; i >= 0; i--)
    {
        string name = ObjectName(0, i, 0, -1);
        if (StringFind(name, prefix) == 0)
            ObjectDelete(0, name);
    }
}

void DrawHLine(const string name, const double price, const color clr,
               const ENUM_LINE_STYLE style, const int width, const string tooltip)
{
    if (price <= 0)
        return;

    if (ObjectFind(0, name) < 0)
        ObjectCreate(0, name, OBJ_HLINE, 0, 0, price);
    else
        ObjectSetDouble(0, name, OBJPROP_PRICE, price);

    ObjectSetInteger(0, name, OBJPROP_COLOR, clr);
    ObjectSetInteger(0, name, OBJPROP_STYLE, style);
    ObjectSetInteger(0, name, OBJPROP_WIDTH, width);
    ObjectSetInteger(0, name, OBJPROP_BACK, true);
    ObjectSetInteger(0, name, OBJPROP_SELECTABLE, false);
    ObjectSetString(0, name, OBJPROP_TOOLTIP, tooltip);
}

void DrawArrow(const string name, const datetime time_val, const double price,
               const int arrow_code, const color clr, const string tooltip)
{
    if (price <= 0)
        return;

    if (ObjectFind(0, name) < 0)
        ObjectCreate(0, name, OBJ_ARROW, 0, time_val, price);
    else
    {
        ObjectSetInteger(0, name, OBJPROP_TIME, time_val);
        ObjectSetDouble(0, name, OBJPROP_PRICE, price);
    }

    ObjectSetInteger(0, name, OBJPROP_ARROWCODE, arrow_code);
    ObjectSetInteger(0, name, OBJPROP_COLOR, clr);
    ObjectSetInteger(0, name, OBJPROP_WIDTH, 2);
    ObjectSetInteger(0, name, OBJPROP_SELECTABLE, false);
    ObjectSetString(0, name, OBJPROP_TOOLTIP, tooltip);
}

void DrawRectangle(const string name, const datetime t1, const double p1,
                   const datetime t2, const double p2, const color clr,
                   const string tooltip)
{
    if (p1 <= 0 || p2 <= 0)
        return;

    if (ObjectFind(0, name) < 0)
        ObjectCreate(0, name, OBJ_RECTANGLE, 0, t1, p1, t2, p2);
    else
    {
        ObjectSetInteger(0, name, OBJPROP_TIME, 0, t1);
        ObjectSetDouble(0, name, OBJPROP_PRICE, 0, p1);
        ObjectSetInteger(0, name, OBJPROP_TIME, 1, t2);
        ObjectSetDouble(0, name, OBJPROP_PRICE, 1, p2);
    }

    ObjectSetInteger(0, name, OBJPROP_COLOR, clr);
    ObjectSetInteger(0, name, OBJPROP_BACK, true);
    ObjectSetInteger(0, name, OBJPROP_FILL, true);
    ObjectSetInteger(0, name, OBJPROP_SELECTABLE, false);
    ObjectSetString(0, name, OBJPROP_TOOLTIP, tooltip);
}

void DrawTradeOnChart(const SignalSnapshot &snap)
{
    // Nur zeichnen wenn das Symbol zum aktuellen Chart passt
    if (StringCompare(snap.symbol, Symbol()) != 0)
        return;

    if (!InpDrawTrades || !snap.valid)
        return;

    // Prefix fuer alle Objekte dieses Symbols
    string pfx = "PYML_" + snap.symbol + "_";

    // Zuerst alte Zeichnungen loeschen
    DeleteObjPrefix(pfx);

    // Kein Signal → nichts zeichnen
    if (StringCompare(snap.richtung, "Kein") == 0 || snap.entry_price <= 0)
        return;

    bool is_long = (StringCompare(snap.richtung, "Long") == 0);
    color trade_clr = is_long ? InpColorLong : InpColorShort;

    // Entry-Zeitpunkt (aus CSV) und Zukunftslinie (+4 Stunden)
    datetime entry_time = snap.ts;
    datetime future_time = entry_time + 4 * 3600;

    // 1) Entry-Pfeil
    int arrow = is_long ? 233 : 234;  // Pfeil hoch / runter
    string tip_entry = StringFormat(
        "%s Entry @ %.5f | Prob=%.1f%% | %s",
        snap.richtung, snap.entry_price, snap.prob * 100.0, snap.regime_name);
    DrawArrow(pfx + "ENTRY", entry_time, snap.entry_price, arrow, trade_clr, tip_entry);

    // 2) Entry-Linie (horizontal, gestrichelt)
    string tip_line = StringFormat("Entry: %.5f", snap.entry_price);
    DrawHLine(pfx + "ENTRY_LINE", snap.entry_price, trade_clr, STYLE_DASH, 1, tip_line);

    // 3) Stop-Loss-Linie
    if (snap.sl_price > 0)
    {
        string tip_sl = StringFormat("SL: %.5f (%.1f Pips)",
            snap.sl_price,
            MathAbs(snap.entry_price - snap.sl_price) / SymbolInfoDouble(Symbol(), SYMBOL_POINT) / 10.0);
        DrawHLine(pfx + "SL_LINE", snap.sl_price, InpColorSL, STYLE_DOT, 2, tip_sl);

        // SL-Zone (Rechteck Entry bis SL)
        DrawRectangle(pfx + "SL_ZONE", entry_time, snap.entry_price,
                      future_time, snap.sl_price, InpColorSL, tip_sl);
    }

    // 4) Take-Profit-Linie
    if (snap.tp_price > 0)
    {
        string tip_tp = StringFormat("TP: %.5f (%.1f Pips)",
            snap.tp_price,
            MathAbs(snap.tp_price - snap.entry_price) / SymbolInfoDouble(Symbol(), SYMBOL_POINT) / 10.0);
        DrawHLine(pfx + "TP_LINE", snap.tp_price, InpColorTP, STYLE_DOT, 2, tip_tp);

        // TP-Zone (halbtransparentes Rechteck)
        DrawRectangle(pfx + "TP_ZONE", entry_time, snap.entry_price,
                      future_time, snap.tp_price, InpColorTP, tip_tp);
    }

    // 5) Info-Label oben rechts im Chart
    string label_name = pfx + "INFO";
    string info_text = StringFormat(
        "%s %s @ %.5f | SL=%.5f | TP=%.5f | Prob=%.0f%% | %s",
        snap.symbol, snap.richtung, snap.entry_price,
        snap.sl_price, snap.tp_price, snap.prob * 100.0, snap.regime_name);

    if (ObjectFind(0, label_name) < 0)
        ObjectCreate(0, label_name, OBJ_LABEL, 0, 0, 0);

    ObjectSetInteger(0, label_name, OBJPROP_XDISTANCE, 10);
    ObjectSetInteger(0, label_name, OBJPROP_YDISTANCE, 50);
    ObjectSetInteger(0, label_name, OBJPROP_CORNER, CORNER_RIGHT_UPPER);
    ObjectSetInteger(0, label_name, OBJPROP_ANCHOR, ANCHOR_RIGHT_UPPER);
    ObjectSetString(0, label_name, OBJPROP_TEXT, info_text);
    ObjectSetString(0, label_name, OBJPROP_FONT, "Consolas");
    ObjectSetInteger(0, label_name, OBJPROP_FONTSIZE, 10);
    ObjectSetInteger(0, label_name, OBJPROP_COLOR, trade_clr);

    ChartRedraw(0);
}

void DrawEntryHistory(const SignalSnapshot &snap)
{
    // Nur fuer aktives Chart-Symbol zeichnen
    if (StringCompare(snap.symbol, Symbol()) != 0)
        return;

    if (!InpDrawTrades || !InpDrawEntryHistory)
        return;

    string pfx = "PYML_" + snap.symbol + "_";
    DeleteObjPrefix(pfx + "HIST_");

    EntryPoint entries[];
    if (!ReadRecentEntries(snap.symbol, entries, InpMaxTradesOnChart))
        return;

    for (int i = 0; i < ArraySize(entries); i++)
    {
        bool is_long = (StringCompare(entries[i].richtung, "Long") == 0);
        int arrow = is_long ? 241 : 242; // kleinere Marker fuer Historie
        color clr = is_long ? InpColorLong : InpColorShort;

        string name = pfx + "HIST_" + IntegerToString(i) + "_" + IntegerToString((int)entries[i].ts);
        string tip = StringFormat(
            "History %s @ %.5f | Prob=%.1f%% | %s | %s",
            entries[i].richtung,
            entries[i].entry_price,
            entries[i].prob * 100.0,
            entries[i].regime_name,
            TimeToString(entries[i].ts, TIME_DATE | TIME_MINUTES));

        DrawArrow(name, entries[i].ts, entries[i].entry_price, arrow, clr, tip);
        ObjectSetInteger(0, name, OBJPROP_WIDTH, 1);
    }

    ChartRedraw(0);
}

bool GetLatestBufferValue(const int handle, double &out_value)
{
    out_value = 0.0;
    if (handle == INVALID_HANDLE)
        return false;

    double buff[];
    ArraySetAsSeries(buff, true);
    int copied = CopyBuffer(handle, 0, 0, 1, buff);
    if (copied < 1)
        return false;

    out_value = buff[0];
    return (out_value > 0);
}

void DrawEmaGuides()
{
    string pfx = "PYML_EMA_GUIDE_";
    if (!InpDrawTechnicalOverlay || !InpDrawEmaGuides)
    {
        DeleteObjPrefix(pfx);
        return;
    }

    double ema20 = 0.0;
    double ema50 = 0.0;
    double ema200 = 0.0;
    if (!GetLatestBufferValue(g_h_ema20, ema20) ||
        !GetLatestBufferValue(g_h_ema50, ema50) ||
        !GetLatestBufferValue(g_h_ema200, ema200))
    {
        return;
    }

    int width = MathMax(1, InpEmaGuideWidth);
    DrawHLine(pfx + "20", ema20, InpEma20GuideColor, STYLE_SOLID, width,
              StringFormat("EMA20 (blau): %.5f", ema20));
    DrawHLine(pfx + "50", ema50, InpEma50GuideColor, STYLE_SOLID, width,
              StringFormat("EMA50 (rot): %.5f", ema50));
    DrawHLine(pfx + "200", ema200, InpEma200GuideColor, STYLE_SOLID, width,
              StringFormat("EMA200 (gruen): %.5f", ema200));

    if (!InpShowEmaStructureLabel)
    {
        if (ObjectFind(0, pfx + "STATUS") >= 0)
            ObjectDelete(0, pfx + "STATUS");
        return;
    }

    double price = iClose(Symbol(), PERIOD_CURRENT, 0);
    if (price <= 0)
        price = SymbolInfoDouble(Symbol(), SYMBOL_BID);

    bool bull_stack = (ema20 > ema50 && ema50 > ema200);
    bool bear_stack = (ema20 < ema50 && ema50 < ema200);
    bool price_above_all = (price > ema20 && price > ema50 && price > ema200);
    bool price_below_all = (price < ema20 && price < ema50 && price < ema200);

    string structure = "MIXED";
    color state_color = clrSilver;
    if (bull_stack && price_above_all)
    {
        structure = "BULL: EMA20>EMA50>EMA200 | Preis ueber allen";
        state_color = InpEma20GuideColor;
    }
    else if (bear_stack && price_below_all)
    {
        structure = "BEAR: EMA20<EMA50<EMA200 | Preis unter allen";
        state_color = InpColorShort;
    }
    else if (bull_stack)
    {
        structure = "BULL Stack, Preis in Pullback-Zone";
        state_color = InpEma50GuideColor;
    }
    else if (bear_stack)
    {
        structure = "BEAR Stack, Preis in Pullback-Zone";
        state_color = InpEma50GuideColor;
    }

    string name = pfx + "STATUS";
    if (ObjectFind(0, name) < 0)
        ObjectCreate(0, name, OBJ_LABEL, 0, 0, 0);

    ObjectSetInteger(0, name, OBJPROP_XDISTANCE, 10);
    ObjectSetInteger(0, name, OBJPROP_YDISTANCE, 42);
    ObjectSetInteger(0, name, OBJPROP_CORNER, CORNER_RIGHT_UPPER);
    ObjectSetInteger(0, name, OBJPROP_ANCHOR, ANCHOR_RIGHT_UPPER);
    ObjectSetInteger(0, name, OBJPROP_SELECTABLE, false);
    ObjectSetInteger(0, name, OBJPROP_HIDDEN, true);
    ObjectSetString(0, name, OBJPROP_FONT, "Consolas");
    ObjectSetInteger(0, name, OBJPROP_FONTSIZE, 9);
    ObjectSetInteger(0, name, OBJPROP_COLOR, state_color);
    ObjectSetString(0, name, OBJPROP_TEXT, structure);
}

// ============================================================
// Ampel-System: Gruen / Gelb / Rot fuer schnelle Uebersicht
// Gruen:  Dir=Long/Short + Prob>55% + Daten frisch → HANDELN
// Gelb:   Daten frisch, aber kein/schwaches Signal → BEOBACHTEN
// Rot:    STALE oder keine Daten → FINGER WEG
// ============================================================

void DrawAmpel()
{
    string pfx = "PYML_AMPEL_";

    if (!InpShowAmpel)
    {
        DeleteObjPrefix(pfx);
        return;
    }

    // Welches Snapshot passt zum aktuellen Chart-Symbol?
    string active_state = "";
    double active_prob = 0.0;
    string active_dir = "Kein";
    bool found_match = false;

    if (StringCompare(g_snap1.symbol, Symbol()) == 0 && g_snap1.valid)
    {
        active_state = SnapshotState(g_snap1);
        active_prob = g_snap1.prob;
        active_dir = g_snap1.richtung;
        found_match = true;
    }
    else if (StringCompare(g_snap2.symbol, Symbol()) == 0 && g_snap2.valid)
    {
        active_state = SnapshotState(g_snap2);
        active_prob = g_snap2.prob;
        active_dir = g_snap2.richtung;
        found_match = true;
    }

    // Kein passendes Symbol → schlechteren Status von beiden nehmen
    if (!found_match)
    {
        string st1 = SnapshotState(g_snap1);
        string st2 = SnapshotState(g_snap2);

        if (StringCompare(st1, "STALE") == 0 || StringCompare(st2, "STALE") == 0 ||
            StringCompare(st1, "MISSING") == 0 || StringCompare(st2, "MISSING") == 0)
        {
            active_state = "STALE";
        }
        else if (StringCompare(st1, "LIVE_SIGNAL") == 0)
        {
            active_state = st1;
            active_prob = g_snap1.prob;
            active_dir = g_snap1.richtung;
        }
        else if (StringCompare(st2, "LIVE_SIGNAL") == 0)
        {
            active_state = st2;
            active_prob = g_snap2.prob;
            active_dir = g_snap2.richtung;
        }
        else
        {
            active_state = "LIVE_NO_SIGNAL";
            active_prob = MathMax(g_snap1.prob, g_snap2.prob);
            active_dir = "Kein";
        }
    }

    // === Ampel-Logik ===
    color ampel_color;
    string ampel_label;
    string ampel_detail;

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

    // === Position: unten links im Chart ===
    int chart_h = (int)ChartGetInteger(0, CHART_HEIGHT_IN_PIXELS, 0);
    int font_size = MathMax(10, InpAmpelFontSize);
    int light_size = font_size + 10;
    int box_height = 60;
    int box_width = 400;
    int x_start = 15;
    int y_top = chart_h - box_height - 25;

    // 1) Hintergrund-Box mit farbigem Rand
    string bg_name = pfx + "BG";
    if (ObjectFind(0, bg_name) < 0)
        ObjectCreate(0, bg_name, OBJ_RECTANGLE_LABEL, 0, 0, 0);

    ObjectSetInteger(0, bg_name, OBJPROP_CORNER, CORNER_LEFT_UPPER);
    ObjectSetInteger(0, bg_name, OBJPROP_XDISTANCE, x_start - 8);
    ObjectSetInteger(0, bg_name, OBJPROP_YDISTANCE, y_top - 5);
    ObjectSetInteger(0, bg_name, OBJPROP_XSIZE, box_width);
    ObjectSetInteger(0, bg_name, OBJPROP_YSIZE, box_height);
    ObjectSetInteger(0, bg_name, OBJPROP_BGCOLOR, ColorToARGB(clrBlack, 190));
    ObjectSetInteger(0, bg_name, OBJPROP_COLOR, ampel_color);
    ObjectSetInteger(0, bg_name, OBJPROP_BORDER_TYPE, BORDER_FLAT);
    ObjectSetInteger(0, bg_name, OBJPROP_BACK, false);
    ObjectSetInteger(0, bg_name, OBJPROP_SELECTABLE, false);
    ObjectSetInteger(0, bg_name, OBJPROP_HIDDEN, true);

    // 2) Farbiges Quadrat als Ampel-Licht
    string light_name = pfx + "LIGHT";
    if (ObjectFind(0, light_name) < 0)
        ObjectCreate(0, light_name, OBJ_RECTANGLE_LABEL, 0, 0, 0);

    ObjectSetInteger(0, light_name, OBJPROP_CORNER, CORNER_LEFT_UPPER);
    ObjectSetInteger(0, light_name, OBJPROP_XDISTANCE, x_start);
    ObjectSetInteger(0, light_name, OBJPROP_YDISTANCE, y_top + (box_height - light_size) / 2 - 5);
    ObjectSetInteger(0, light_name, OBJPROP_XSIZE, light_size);
    ObjectSetInteger(0, light_name, OBJPROP_YSIZE, light_size);
    ObjectSetInteger(0, light_name, OBJPROP_BGCOLOR, ampel_color);
    ObjectSetInteger(0, light_name, OBJPROP_COLOR, ampel_color);
    ObjectSetInteger(0, light_name, OBJPROP_BORDER_TYPE, BORDER_FLAT);
    ObjectSetInteger(0, light_name, OBJPROP_BACK, false);
    ObjectSetInteger(0, light_name, OBJPROP_SELECTABLE, false);
    ObjectSetInteger(0, light_name, OBJPROP_HIDDEN, true);

    // 3) Hauptlabel (z.B. "GRUEN: HANDELN")
    string label_name = pfx + "LABEL";
    if (ObjectFind(0, label_name) < 0)
        ObjectCreate(0, label_name, OBJ_LABEL, 0, 0, 0);

    ObjectSetInteger(0, label_name, OBJPROP_CORNER, CORNER_LEFT_UPPER);
    ObjectSetInteger(0, label_name, OBJPROP_ANCHOR, ANCHOR_LEFT_UPPER);
    ObjectSetInteger(0, label_name, OBJPROP_XDISTANCE, x_start + light_size + 12);
    ObjectSetInteger(0, label_name, OBJPROP_YDISTANCE, y_top + 4);
    ObjectSetString(0, label_name, OBJPROP_FONT, "Consolas");
    ObjectSetInteger(0, label_name, OBJPROP_FONTSIZE, font_size);
    ObjectSetInteger(0, label_name, OBJPROP_COLOR, ampel_color);
    ObjectSetString(0, label_name, OBJPROP_TEXT, ampel_label);
    ObjectSetInteger(0, label_name, OBJPROP_SELECTABLE, false);
    ObjectSetInteger(0, label_name, OBJPROP_HIDDEN, true);

    // 4) Detail-Zeile (z.B. "Dir=Kein | Prob=37% ...")
    string detail_name = pfx + "DETAIL";
    if (ObjectFind(0, detail_name) < 0)
        ObjectCreate(0, detail_name, OBJ_LABEL, 0, 0, 0);

    ObjectSetInteger(0, detail_name, OBJPROP_CORNER, CORNER_LEFT_UPPER);
    ObjectSetInteger(0, detail_name, OBJPROP_ANCHOR, ANCHOR_LEFT_UPPER);
    ObjectSetInteger(0, detail_name, OBJPROP_XDISTANCE, x_start + light_size + 12);
    ObjectSetInteger(0, detail_name, OBJPROP_YDISTANCE, y_top + font_size + 14);
    ObjectSetString(0, detail_name, OBJPROP_FONT, "Consolas");
    ObjectSetInteger(0, detail_name, OBJPROP_FONTSIZE, font_size - 2);
    ObjectSetInteger(0, detail_name, OBJPROP_COLOR, clrSilver);
    ObjectSetString(0, detail_name, OBJPROP_TEXT, ampel_detail);
    ObjectSetInteger(0, detail_name, OBJPROP_SELECTABLE, false);
    ObjectSetInteger(0, detail_name, OBJPROP_HIDDEN, true);
}

void DrawDashboardLabel(const string text)
{
    string text_prefix = "PYML_DASH_TEXT_";
    string bg_name = "PYML_DASH_BG";
    int font_size = MathMax(8, InpDashboardFontSize);
    int x = 14;
    int y = 20;
    int pad = 14;

    // Grobe Textfläche abschätzen (Monospace-Font: Consolas)
    string lines[];
    int line_count = StringSplit(text, '\n', lines);
    if (line_count <= 0)
        line_count = 1;

    int max_chars = 0;
    for (int i = 0; i < line_count; i++)
    {
        int len = StringLen(lines[i]);
        if (len > max_chars)
            max_chars = len;
    }

    // Grosszuegig berechnen damit alles passt
    int char_width = (int)(font_size * 0.72);
    int line_height_est = font_size + 10;
    int text_width = max_chars * char_width + 2 * pad;
    int text_height = line_count * line_height_est + 2 * pad;

    // Hintergrund auf ~60% Chartbreite ausdehnen (ca. 20 Quadrate)
    int chart_w = (int)ChartGetInteger(0, CHART_WIDTH_IN_PIXELS);
    int chart_h = (int)ChartGetInteger(0, CHART_HEIGHT_IN_PIXELS);
    int width = MathMax(text_width, (int)(chart_w * 0.45));
    int height = MathMax(text_height, (int)(chart_h * 0.40));
    width = MathMax(width, 300);
    height = MathMax(height, 250);

    if (InpDashboardTextBackground)
    {
        if (ObjectFind(0, bg_name) < 0)
            ObjectCreate(0, bg_name, OBJ_RECTANGLE_LABEL, 0, 0, 0);

        uint alpha_color = ColorToARGB(InpDashboardBgColor, (uchar)MathMax(0, MathMin(255, InpDashboardBgAlpha)));
        ObjectSetInteger(0, bg_name, OBJPROP_XDISTANCE, x - pad);
        ObjectSetInteger(0, bg_name, OBJPROP_YDISTANCE, y - pad);
        ObjectSetInteger(0, bg_name, OBJPROP_XSIZE, width);
        ObjectSetInteger(0, bg_name, OBJPROP_YSIZE, height);
        ObjectSetInteger(0, bg_name, OBJPROP_CORNER, CORNER_LEFT_UPPER);
        ObjectSetInteger(0, bg_name, OBJPROP_COLOR, clrDimGray);
        ObjectSetInteger(0, bg_name, OBJPROP_BGCOLOR, alpha_color);
        ObjectSetInteger(0, bg_name, OBJPROP_BORDER_TYPE, BORDER_FLAT);
        ObjectSetInteger(0, bg_name, OBJPROP_BACK, false);
        ObjectSetInteger(0, bg_name, OBJPROP_SELECTABLE, false);
        ObjectSetInteger(0, bg_name, OBJPROP_HIDDEN, true);
    }
    else
    {
        if (ObjectFind(0, bg_name) >= 0)
            ObjectDelete(0, bg_name);
    }

    // Alte Zeilen löschen und neu aufbauen (stabil bei wechselnder Zeilenanzahl)
    DeleteObjPrefix(text_prefix);

    int line_height = font_size + 12;
    for (int i = 0; i < line_count; i++)
    {
        string name = text_prefix + IntegerToString(i);
        if (ObjectFind(0, name) < 0)
            ObjectCreate(0, name, OBJ_LABEL, 0, 0, 0);

        ObjectSetInteger(0, name, OBJPROP_XDISTANCE, x);
        ObjectSetInteger(0, name, OBJPROP_YDISTANCE, y + i * line_height);
        ObjectSetInteger(0, name, OBJPROP_CORNER, CORNER_LEFT_UPPER);
        ObjectSetInteger(0, name, OBJPROP_ANCHOR, ANCHOR_LEFT_UPPER);
        ObjectSetInteger(0, name, OBJPROP_SELECTABLE, false);
        ObjectSetInteger(0, name, OBJPROP_BACK, false);
        ObjectSetInteger(0, name, OBJPROP_HIDDEN, true);
        ObjectSetString(0, name, OBJPROP_FONT, "Consolas");
        ObjectSetInteger(0, name, OBJPROP_FONTSIZE, font_size);
        ObjectSetInteger(0, name, OBJPROP_COLOR, InpDashboardTextColor);
        ObjectSetString(0, name, OBJPROP_TEXT, lines[i]);
    }
}

// Kurze State-Bezeichnungen fuer Kompaktmodus
string ShortState(const string state)
{
    if (StringCompare(state, "LIVE_SIGNAL") == 0)     return "SIGNAL";
    if (StringCompare(state, "LIVE_NO_SIGNAL") == 0)  return "IDLE";
    if (StringCompare(state, "MISSING") == 0)          return "MISS";
    if (StringCompare(state, "STALE") == 0)             return "STALE";
    if (StringCompare(state, "NO_TS") == 0)             return "NO_TS";
    if (StringCompare(state, "WAITING_FOR_CSV") == 0)   return "WAIT";
    return state;
}

void DrawDashboard()
{
    string st1 = SnapshotState(g_snap1);
    string st2 = SnapshotState(g_snap2);
    string overall = OverallState(st1, st2);
    string stale_src = InpUseSignalTimeframeForStale ? "SignalTF" : "ChartTF";
    string sep = "-------------------------";
    string dashboard_text;

    if (InpDashboardCompactMode)
    {
        // Kompakt: kurze Zeilen, max ~35 Zeichen pro Zeile
        string header = StringFormat("Live Dashboard | %s", overall);
        string stale_info = StringFormat(
            "STALE>%dmin (%s)",
            EffectiveStaleMinutes(), stale_src);

        // Symbol 1 – Zeile 1: Status, Zeile 2: Details
        string s1a = StringFormat("%s | %s | %s",
            g_snap1.symbol, ShortState(st1),
            FreshnessText(g_snap1));
        string s1b = StringFormat("  %s | P=%.2f | R=%d",
            g_snap1.richtung, g_snap1.prob,
            (int)g_snap1.rows);

        // Symbol 2 – Zeile 1: Status, Zeile 2: Details
        string s2a = StringFormat("%s | %s | %s",
            g_snap2.symbol, ShortState(st2),
            FreshnessText(g_snap2));
        string s2b = StringFormat("  %s | P=%.2f | R=%d",
            g_snap2.richtung, g_snap2.prob,
            (int)g_snap2.rows);

        dashboard_text = header + "\n" + stale_info + "\n" +
            sep + "\n" +
            s1a + "\n" + s1b + "\n" +
            s2a + "\n" + s2b;

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
        // Ausfuehrlich: alle Details auf kurze Zeilen verteilt
        string header = "=== Live Signal Dashboard ===";
        string status_line = StringFormat(
            "STATUS=%s | STALE>%dmin (%s)",
            overall, EffectiveStaleMinutes(), stale_src);

        // Symbol 1 – drei Zeilen
        string s1a = StringFormat("%s | %s | %s",
            g_snap1.symbol, st1,
            FreshnessText(g_snap1));
        string s1b = StringFormat("  Dir=%s | Prob=%.3f",
            g_snap1.richtung, g_snap1.prob);
        string s1c = StringFormat("  Regime=%s | Mode=%s | R=%d",
            g_snap1.regime_name, g_snap1.modus,
            (int)g_snap1.rows);

        // Symbol 2 – drei Zeilen
        string s2a = StringFormat("%s | %s | %s",
            g_snap2.symbol, st2,
            FreshnessText(g_snap2));
        string s2b = StringFormat("  Dir=%s | Prob=%.3f",
            g_snap2.richtung, g_snap2.prob);
        string s2c = StringFormat("  Regime=%s | Mode=%s | R=%d",
            g_snap2.regime_name, g_snap2.modus,
            (int)g_snap2.rows);

        dashboard_text = header + "\n" + status_line + "\n" +
            sep + "\n" +
            s1a + "\n" + s1b + "\n" + s1c + "\n" +
            s2a + "\n" + s2b + "\n" + s2c;

        if (InpDebugHistoryInfo)
        {
            string hist = StringFormat(
                "History: %s=%d | %s=%d (max=%d)",
                InpSymbol1, g_hist_count_1,
                InpSymbol2, g_hist_count_2,
                InpMaxTradesOnChart);
            dashboard_text += "\n" + sep + "\n" + hist;
        }

        string hint = "CSV: " + ExpectedFolderText();
        dashboard_text += "\n" + hint;
    }

    if (InpUseLargeDashboardText)
    {
        Comment("");
        DrawDashboardLabel(dashboard_text);
    }
    else
    {
        DeleteObjPrefix("PYML_DASH_");
        Comment(dashboard_text);
    }
}

void RefreshAll()
{
    string file_1 = BuildFileName(InpSymbol1);
    string file_2 = BuildFileName(InpSymbol2);

    bool ok_1 = ReadLatestSnapshot(InpSymbol1, g_snap1);
    bool ok_2 = ReadLatestSnapshot(InpSymbol2, g_snap2);

    LogFileState(InpSymbol1, file_1, ok_1, g_missing_1, g_missing_log_ts_1);
    LogFileState(InpSymbol2, file_2, ok_2, g_missing_2, g_missing_log_ts_2);

    MaybeAlert(g_snap1, g_last_alert_ts_1);
    MaybeAlert(g_snap2, g_last_alert_ts_2);

    g_hist_count_1 = CountRecentEntries(InpSymbol1, InpMaxTradesOnChart);
    g_hist_count_2 = CountRecentEntries(InpSymbol2, InpMaxTradesOnChart);

    DrawDashboard();

    // Chart-Zeichnungen aktualisieren (nur fuer passendes Symbol)
    DrawTradeOnChart(g_snap1);
    DrawTradeOnChart(g_snap2);
    DrawEntryHistory(g_snap1);
    DrawEntryHistory(g_snap2);
    DrawEmaGuides();
    DrawAmpel();
}

int OnInit()
{
    g_refresh_seconds = (InpRefreshSeconds < 1) ? 1 : InpRefreshSeconds;
    EventSetTimer(g_refresh_seconds);

    g_snap1.symbol = InpSymbol1;
    g_snap2.symbol = InpSymbol2;
    g_snap1.valid = false;
    g_snap2.valid = false;

    SetupTechnicalOverlay();

    Print("[Dashboard] Initialisiert. Warte auf CSV-Signale...");
    RefreshAll();
    return (INIT_SUCCEEDED);
}

void OnDeinit(const int reason)
{
    EventKillTimer();
    Comment("");
    ReleaseTechnicalOverlay();

    // Alle Chart-Objekte entfernen
    DeleteObjPrefix("PYML_");
    ChartRedraw(0);
}

void OnTick()
{
    // Keine Tick-Logik nötig. Refresh erfolgt über Timer.
}

void OnTimer()
{
    RefreshAll();
}
