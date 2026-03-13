# Nomenklator – Abkürzungen im Projekt

**Projekt:** MT5 ML-Trading-System  
**Stand:** 2026-03-02  
**Gilt für:** Linux-Server + Windows-Laptop (MT5)

**Interaktive Suche:** `reports/nomenklator_abkuerzungen_suche.html`

> Dieser Nomenklator sammelt die wichtigsten Abkürzungen aus Code, Reports und Roadmap.
> Fokus: fachlich relevante Kürzel + häufige Projekt-Konstanten.

## 📖 Inhaltsverzeichnis

- [Trading & Markt](#trading--markt)
- [Zeitrahmen & Marktstruktur](#zeitrahmen--marktstruktur)
- [Indikatoren & Features](#indikatoren--features)
- [Machine Learning & Statistik](#machine-learning--statistik)
- [Daten, Dateien & Formate](#daten-dateien--formate)
- [Infrastruktur, Betrieb & Protokolle](#infrastruktur-betrieb--protokolle)
- [Symbole (Forex)](#symbole-forex)
- [Krypto-/Sentiment-Bezug](#krypto-sentiment-bezug)
- [Häufige Projekt-Konstanten im Code](#häufige-projekt-konstanten-im-code)
- [Hinweis zur Vollständigkeit](#hinweis-zur-vollständigkeit)

---

## Trading & Markt

| Abkürzung | Bedeutung | Kurz erklärt |
| --- | --- | --- |
| MT5 | MetaTrader 5 | Trading-Plattform und Python-API (Windows-Host). |
| FX | Foreign Exchange | Devisenmarkt (Forex). |
| OHLC | Open High Low Close | Kerzenpreise: Eröffnung, Hoch, Tief, Schluss. |
| OHLCV | Open High Low Close Volume | OHLC plus Volumen. |
| TP | Take Profit | Gewinnziel einer Position. |
| SL | Stop Loss | Verlustbegrenzung einer Position. |
| RRR | Risk-Reward Ratio | Verhältnis Risiko zu Chance, z. B. 1:1. |
| PF | Profit Factor | Summe Gewinne / Summe Verluste. |
| DD | Drawdown | Rückgang vom Equity-Hoch zum Tief. |
| MaxDD | Maximaler Drawdown | Größter historischer Drawdown in %. |
| WR | Win Rate | Trefferquote gewonnener Trades in %. |
| BT | Backtest | Simulation auf historischen Daten. |
| LIVE | Live-/Paper-Betrieb | Laufender Betrieb statt Offline-Training. |
| PAPER | Paper-Trading | Simulation ohne Echtgeld. |
| PAPER_ONLY | Nur Paper erlaubt | Eskalation zu Echtgeld gesperrt. |
| GO / NO-GO | Entscheidungsstatus | KPI-basiert freigegeben oder nicht. |
| UNKLAR | Bewertung unklar | Zu wenig Signale/Daten für robuste Entscheidung. |
| SPREAD | Bid-Ask-Differenz | Handelskosten zwischen Kauf/Verkauf. |
| SWAP | Overnight-Kosten | Haltekosten über Nacht. |
| LOT | Positionsgröße | Handelsvolumen in Lot-Einheiten. |
| EA | Expert Advisor | Automatisiertes Script im MT5-Umfeld. |
| GTC | Good Till Cancelled | Order gilt bis manuell gelöscht/ausgeführt. |
| IOC | Immediate Or Cancel | Sofort ausführbarer Teil wird ausgeführt, Rest verworfen. |

## Zeitrahmen & Marktstruktur

| Abkürzung | Bedeutung | Kurz erklärt |
| --- | --- | --- |
| M15 | 15-Minuten-Chart | Intraday-Zeitrahmen mit hoher Signalzahl. |
| M30 | 30-Minuten-Chart | Intraday-Zeitrahmen zwischen M15 und H1. |
| M60 | 60-Minuten-Chart | Im Projekt als eigener Flow (MT5 intern H1). |
| H1 | 1-Stunden-Chart | Standard-Zeitrahmen des Basissystems. |
| H4 | 4-Stunden-Chart | Höherer TF für robustere, seltenere Signale. |
| D1 | Tages-Chart | Ein Tagesbalken als übergeordneter Kontext. |
| TF | Timeframe | Allgemeiner Begriff für Zeitrahmen. |
| MTF | Multi-Timeframe | Features aus mehreren Zeitrahmen kombiniert. |
| ADX | Average Directional Index | Trendstärke-Indikator. |
| ATR | Average True Range | Volatilitätsmaß, oft für dynamischen SL. |
| REGIME | Marktphase | Klassifikation z. B. Trend/Seitwärts/hohe Vola. |

## Indikatoren & Features

| Abkürzung | Bedeutung | Kurz erklärt |
| --- | --- | --- |
| SMA | Simple Moving Average | Einfacher gleitender Durchschnitt. |
| EMA | Exponential Moving Average | Exponentiell gewichteter Durchschnitt. |
| RSI | Relative Strength Index | Momentum-Indikator (Überkauft/Überverkauft). |
| MACD | Moving Average Convergence Divergence | Trend-/Momentum-Kombinationsindikator. |
| BB | Bollinger Bands | Volatilitätsbänder um gleitenden Mittelwert. |
| OBV | On-Balance Volume | Volumenfluss-Proxy. |
| ROC | Rate of Change | Preisänderung über definierten Zeitraum. |
| VWAP | Volume Weighted Average Price | Volumengewichteter Durchschnittspreis. |
| SMA20 / SMA50 / SMA200 | SMA über 20/50/200 Perioden | Häufige Trendfilter im Projekt. |
| ATR_14 | ATR mit 14 Perioden | Standard-Setting für Volatilität. |

## Machine Learning & Statistik

| Abkürzung | Bedeutung | Kurz erklärt |
| --- | --- | --- |
| ML | Machine Learning | Modellbasiertes Lernen aus Daten. |
| XGB | XGBoost | Gradient-Boosting-Modell (Baumbasiert). |
| LGBM | LightGBM | Schnelles Boosting-Modell (Baumbasiert). |
| SHAP | SHapley Additive exPlanations | Erklärbarkeit von Feature-Beiträgen. |
| F1 | F1-Score | Harm. Mittel von Precision und Recall. |
| F1-Macro | Makro-F1 | F1 gleichgewichtet über alle Klassen. |
| PSI | Population Stability Index | Drift-Metrik zwischen Verteilungen. |
| SVM | Support Vector Machine | Klassischer ML-Algorithmus (hier eher Referenz). |
| CI | Continuous Integration | Automatische Prüfungen bei Code-Änderungen. |
| RFE | Recursive Feature Elimination | Iterative Feature-Auswahl (Roadmap offen). |

## Daten, Dateien & Formate

| Abkürzung | Bedeutung | Kurz erklärt |
| --- | --- | --- |
| CSV | Comma-Separated Values | Standardformat für Kerzen/Features/Trades. |
| PKL | Pickle-Containerdatei | Modellartefakt; im Projekt via `joblib` geschrieben. |
| JSON | JavaScript Object Notation | Strukturierte Konfig-/Historiendaten. |
| PNG | Portable Network Graphics | Plot-/Chart-Bilder. |
| UTF-8 | Unicode Transformation Format | Textkodierung für Dateien/Logs. |
| KB / MB / GB / TB | Kilobyte/Megabyte/Gigabyte/Terabyte | Größenangaben für Dateien/Speicher. |

## Infrastruktur, Betrieb & Protokolle

| Abkürzung | Bedeutung | Kurz erklärt |
| --- | --- | --- |
| API | Application Programming Interface | Programmschnittstelle zu externen Diensten. |
| SMTP | Simple Mail Transfer Protocol | E-Mail-Versandprotokoll. |
| SSH | Secure Shell | Sichere Remote-Verbindung zum Linux-Server. |
| SCP | Secure Copy | Dateiübertragung über SSH. |
| SFTP | SSH File Transfer Protocol | Dateitransfer über SSH-Protokoll. |
| HTTP | Hypertext Transfer Protocol | Web-Protokoll für API-Aufrufe. |
| SLA | Service Level Agreement | Verfügbarkeits-/Qualitätszusage von Diensten. |
| VPS | Virtual Private Server | Dauerbetrieb-Option für 24/7-Prozesse. |
| UTC | Coordinated Universal Time | Zeitzonenreferenz im Datenhandling. |
| STALE | Veraltete Daten | Dashboard-/Datenfrische-Warnstatus. |
| HEARTBEAT | Lebenszeichen-Log | Regelmäßiger Betriebsstatus eines Prozesses. |

## Symbole (Forex)

| Kürzel | Bedeutung |
| --- | --- |
| EURUSD | Euro / US-Dollar |
| GBPUSD | Britisches Pfund / US-Dollar |
| USDJPY | US-Dollar / Japanischer Yen |
| AUDUSD | Australischer Dollar / US-Dollar |
| USDCAD | US-Dollar / Kanadischer Dollar |
| USDCHF | US-Dollar / Schweizer Franken |
| NZDUSD | Neuseeland-Dollar / US-Dollar |

## Krypto-/Sentiment-Bezug

| Abkürzung | Bedeutung | Kurz erklärt |
| --- | --- | --- |
| BTC | Bitcoin | Externe Kontextvariable in Features. |
| BTCUSDT | Bitcoin vs. Tether | Binance-Marktbezug für Flows. |
| OI | Open Interest | Offene Kontraktanzahl (Derivate-Kontext). |
| USDT | Tether USD | Stablecoin-Quote in Krypto-Paaren. |

## Häufige Projekt-Konstanten im Code

| Kürzel | Bedeutung |
| --- | --- |
| BASE_DIR | Projekt-Basisverzeichnis |
| DATA_DIR | Datenverzeichnis |
| MODEL_DIR | Modellverzeichnis |
| PLOTS_DIR | Plot-/Grafikverzeichnis |
| LOG_DIR / LOGS_DIR | Log-Verzeichnisse |
| REPORTS_DIR | Report-Verzeichnis |
| BACKTEST_DIR | Backtest-Verzeichnis |
| KPI_HISTORY_PATH | Historie der Wochen-KPIs |
| KERN_SYMBOLE | Operative Symbole (`USDCAD`, `USDJPY`) |
| AKTIVE_SYMBOLE | Aktiv überwachte Symbolmenge |
| TIMEFRAME_CONFIG | Zuordnung je Zeitrahmen (z. B. Bars pro Stunde) |
| AUSSCHLUSS_SPALTEN | Nicht als Features verwendete Spalten |
| KLASSEN_NAMEN | Mapping der Zielklassen (Short/Neutral/Long) |
| PAPER_GATE_WOCHEN | Wochenanzahl für 3-Monats-Gate (12) |
| KILL_SWITCH_DD_DEFAULT | Standard-Limit für Kill-Switch-Drawdown |
| HEARTBEAT_LOG_DEFAULT | Standardintervall Heartbeat-Logging |

---

## Hinweis zur Vollständigkeit

Der Nomenklator ist **praktisch-vollständig** für den aktuellen Projektbetrieb (Roadmap Phase 7).
Wenn du möchtest, erweitere ich ihn als nächsten Schritt automatisch um **alle** im Code gefundenen Kürzel (auch seltene Konstanten) und sortiere sie alphabetisch in einer zweiten Referenzliste.

