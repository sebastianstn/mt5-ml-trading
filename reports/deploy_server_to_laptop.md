# Deploy Runbook – Server → Windows Laptop (Paper-Betrieb)

**Stand:** 2026-03-01  
**Ziel:** Reproduzierbarer Transfer von Trading-Artefakten vom Linux-Server auf den Windows-Laptop.

---

## Architektur (wichtig)

- **Linux-Server:** Source of Truth (Code, Modelle, Git, Reports)
- **Windows-Laptop:** Runtime (MT5, `live_trader.py`, Sync, Dashboard)

> In deinem Setup liegt `.git` nur auf dem Server. Deshalb: **Git-Hooks nur auf dem Server**.

---

## 1) Vor dem Deploy (Server)

1. Virtuelle Umgebung aktivieren.
2. Optional: Tests/Guard laufen lassen.
3. Prüfen, welche Dateien übertragen werden sollen.

Typische Artefakte:

- `live/live_trader.py`
- `models/lgbm_usdcad_v1.pkl`
- `models/lgbm_usdjpy_v1.pkl`
- optional weitere Modelle / `requirements-laptop.txt`

---

## 2) Standard-Deploy (empfohlen)

Nutze das bestehende Skript auf dem Server:

- `deploy_to_laptop.sh`

Vorteile:

- erstellt Zielordner auf Windows
- überträgt Dateien robust per SFTP
- vermeidet Quoting-Probleme bei Windows-Pfaden mit Leerzeichen

---

## 3) Nach dem Deploy (Windows-Laptop)

1. MT5 offen und verbunden.
2. Trader-Prozesse neu starten (USDCAD, USDJPY).
3. Logs prüfen:
   - `logs/USDCAD_live_trades.csv`
   - `logs/USDJPY_live_trades.csv`
4. Sync prüfen (`MT5_Sync_Live_Logs`).
5. Dashboard prüfen: kein permanentes `MISSING` / `STALE`.

---

## 4) Smoke-Checks

### Server

- Deploy-Skript lief ohne Fehlercode.
- Erwartete Dateien wurden übertragen.

### Laptop

- `live_trader.py` startet ohne Import-Fehler.
- Heartbeat-Einträge erscheinen regelmäßig (auch bei `Kein` Signal).
- CSVs werden in MT5 `Common\Files` gespiegelt.

---

## 5) Häufige Fehler + Fix

### Fehler: `scp ... Exit Code 1`

Häufige Ursachen:

1. Platzhalter nicht ersetzt (`<LAPTOP_USER>`, `<LAPTOP_IP_ODER_NAME>`).
2. Pfad mit Leerzeichen nicht korrekt gequotet.
3. OpenSSH auf Windows nicht erreichbar.

Fix:

- bevorzugt `deploy_to_laptop.sh` verwenden (SFTP-basiert)
- alternativ Host/User/Pfad explizit prüfen

### Fehler: Dateien landen auf Laptop, aber Dashboard bleibt `MISSING`

- Quelle/Target im Sync prüfen
- Task-Status kontrollieren (`LastTaskResult`)
- `InpUseCommonFiles=true` im EA sicherstellen

---

## 6) Change-Disziplin (Phase 7)

- Änderungen zuerst auf dem Server committen.
- Deploy danach bewusst ausführen (kein „silent drift").
- Incident bei Ausfall im Wochenlog dokumentieren.

---

## 7) Kurz-Checkliste pro Deploy

- [ ] Server-Code final (inkl. Guard)
- [ ] Deploy auf Laptop ausgeführt
- [ ] Trader USDCAD + USDJPY neu gestartet
- [ ] Sync/Task grün
- [ ] Dashboard CONNECTED/PARTIAL ohne Dauerfehler
- [ ] Daily Ops + Incident-Log aktualisiert
