# Doc Sync Checklist (Phase 7)

Ziel: Verhindern, dass zentrale Projektdokumente beim Tagesbetrieb auseinanderlaufen.

---

## Wann ausführen?

- Nach Änderungen an Status/Phase, Betriebs-Policy oder Roadmap.
- Vor größeren Commits/Merges.
- Spätestens 1x pro Woche zusammen mit dem KPI-Report.

---

## Pflichtdateien (Synchron halten)

- `README.md`
- `Roadmap.md`
- `CLAUDE.md`
- `.github/copilot-instructions.md`
- `reports/paper_trading_90d_plan.md`

---

## Mindestinhalt (Quick Check)

1. **Phase-Status:** überall konsistent (`Phase 7` aktiv).
2. **Operative Paare:** `USDCAD` + `USDJPY` als aktive Policy sichtbar.
3. **Roadmap-Verweis:** überall `Roadmap.md` (kein `ROADMAP.md`).
4. **Roadmap-Policy:** Zeile „Aktive operative Paare (Paper)“ vorhanden.

---

## Automatischer Guard

Nutze dafür das Skript:

- `reports/doc_drift_guard.py`

Wenn der Guard fehlschlägt, zuerst die gemeldeten Dateien korrigieren und danach erneut prüfen.

---

## Optionaler Team-Standard

- Dokumenten-Updates als eigener Commit-Abschnitt („docs: sync phase/policy“).
- Im Wochenreport kurz notieren, ob Doc-Guard grün war.
