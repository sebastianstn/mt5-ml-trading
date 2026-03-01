#property strict
#property version "1.00"
#property description "MT5 Dashboard fuer Python Live-Signale (USDCAD/USDJPY)"
#property description "Liest CSV aus Common/Files und zeigt Status + Alerts."

input string InpSymbol1 = "USDCAD";
input string InpSymbol2 = "USDJPY";
input string InpFileSuffix = "_live_trades.csv";
input bool InpUseCommonFiles = true;
input int InpRefreshSeconds = 5;
input bool InpEnableAlerts = true;
input int InpStaleMinutes = 20;
input int InpMissingFileLogEverySec = 300;

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

    int age_min = (int)((TimeCurrent() - snap.ts) / 60);
    string tag = (age_min <= InpStaleMinutes) ? "OK" : "STALE";
    return StringFormat("%s (%d min)", tag, age_min);
}

string SnapshotState(const SignalSnapshot &snap)
{
    if (!snap.valid)
        return "MISSING";

    if (snap.ts <= 0)
        return "NO_TS";

    int age_min = (int)((TimeCurrent() - snap.ts) / 60);
    if (age_min > InpStaleMinutes)
        return "STALE";

    if (StringCompare(snap.richtung, "Kein") == 0)
        return "LIVE_NO_SIGNAL";

    return "LIVE_SIGNAL";
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

void DrawDashboard()
{
    string st1 = SnapshotState(g_snap1);
    string st2 = SnapshotState(g_snap2);
    string overall = OverallState(st1, st2);

    string line1 = StringFormat(
        "=== Live Signal Dashboard (Python -> MT5) === | STATUS=%s",
        overall);

    string s1 = StringFormat(
        "%s | State=%s | %s | Dir=%s | Prob=%.3f | Regime=%s | Mode=%s | Rows=%d",
        g_snap1.symbol,
        st1,
        FreshnessText(g_snap1),
        g_snap1.richtung,
        g_snap1.prob,
        g_snap1.regime_name,
        g_snap1.modus,
        (int)g_snap1.rows);

    string s2 = StringFormat(
        "%s | State=%s | %s | Dir=%s | Prob=%.3f | Regime=%s | Mode=%s | Rows=%d",
        g_snap2.symbol,
        st2,
        FreshnessText(g_snap2),
        g_snap2.richtung,
        g_snap2.prob,
        g_snap2.regime_name,
        g_snap2.modus,
        (int)g_snap2.rows);

    string hint = "CSV-Quelle: " + ExpectedFolderText() + "<SYMBOL>" + InpFileSuffix;
    string legend = "Legend: MISSING=no file, STALE=old timestamp, LIVE_SIGNAL=frisches Signal";
    Comment(line1 + "\n" + s1 + "\n" + s2 + "\n" + hint + "\n" + legend);
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

    DrawDashboard();
}

int OnInit()
{
    g_refresh_seconds = (InpRefreshSeconds < 1) ? 1 : InpRefreshSeconds;
    EventSetTimer(g_refresh_seconds);

    g_snap1.symbol = InpSymbol1;
    g_snap2.symbol = InpSymbol2;
    g_snap1.valid = false;
    g_snap2.valid = false;

    Print("[Dashboard] Initialisiert. Warte auf CSV-Signale...");
    RefreshAll();
    return (INIT_SUCCEEDED);
}

void OnDeinit(const int reason)
{
    EventKillTimer();
    Comment("");
}

void OnTick()
{
    // Keine Tick-Logik nötig. Refresh erfolgt über Timer.
}

void OnTimer()
{
    RefreshAll();
}
