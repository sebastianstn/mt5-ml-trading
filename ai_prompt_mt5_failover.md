# System-Architektur und Failover für einen MetaTrader-5-Trading-Bot

**Rolle:** Agiere als erfahrener Systemarchitekt, DevOps Engineer und Softwareentwickler mit Spezialisierung auf Hochverfügbarkeit (High Availability) und automatisierte Trading-Systeme.

## 1. Kontext und Ausgangslage

Ich habe eine automatisierte Trading-Applikation entwickelt, die Trades auf der Plattform MetaTrader 5 (MT5) ausführt. Derzeit läuft diese Anwendung als einzelne Instanz.

## 2. Zielsetzung

Ich möchte ein hochverfügbares Failover-System aufbauen. Wenn die aktuelle Trading-Routine (die aktive Instanz) ausfällt, abstürzt oder die Verbindung verliert, muss automatisch eine andere, im Hintergrund wartende Routine (die passive Instanz) übernehmen. Dadurch soll die Kontinuität des Tradings ohne menschliches Eingreifen gewährleistet werden.

## 3. Die kritische Herausforderung (Active-Passive)

Beim automatisierten Trading ist es absolut fatal, wenn zwei Instanzen gleichzeitig "scharf" sind (Active-Active), da sonst Signale doppelt verarbeitet und Orders versehentlich mehrfach ausgeführt werden. Es muss daher zu jedem Zeitpunkt garantiert sein, dass maximal **eine** Instanz aktiv Trades an MT5 sendet.

## 4. Zu evaluierende Lösungsansätze

Bitte analysiere und entwerfe die Architektur für die beiden folgenden Optionen, idealerweise im Kontext von Kubernetes- oder Docker-basierten Umgebungen:

- **Option A: Self-Healing / Replica-1-Ansatz**
    Das System erzwingt, dass immer genau eine Instanz läuft. Fällt diese aus, wird sie vom Orchestrator (zum Beispiel Kubernetes) beendet und automatisch neu gestartet.

- **Option B: Leader-Election-Ansatz (nahtloses Failover)**
    Es laufen dauerhaft zwei Instanzen (Leader und Follower). Beide sind hochgefahren und mit MT5 verbunden, aber nur der Leader darf Trades absetzen. Über einen Locking-Mechanismus (zum Beispiel via Redis, etcd oder Kubernetes Leases) wird der Leader bestimmt. Fällt dieser aus, übernimmt der Follower innerhalb kürzester Zeit.

## 5. Konkrete Aufgaben für die KI

Bitte bearbeite die folgenden Schritte nacheinander:

1. **Vergleich:** Vergleiche Option A und Option B hinsichtlich Implementierungsaufwand, Ausfallzeit bei einem Crash und Sicherheit gegen Split-Brain-Szenarien (also Fälle, in denen beide Instanzen glauben, sie seien der Leader).
2. **Architektur-Vorschlag:** Erstelle einen konkreten Architektur-Vorschlag, welche Tools (zum Beispiel Kubernetes, Docker Swarm oder externe Datenbanken für Locks) wir dafür nutzen sollten.
3. **Implementierungsvorbereitung:** Erkläre, welche Mechanismen (zum Beispiel Health Checks und Heartbeats) wir in den Code meiner bestehenden App einbauen müssen, damit das System erkennt, ob die App noch gesund ist.

## 6. Wichtige Arbeitsanweisungen (strikt befolgen)

Für alle unsere zukünftigen Interaktionen und Code-Generierungen gelten folgende Regeln:

- Besprich zuerst immer kurz alle deine Entscheidungen mit mir, bevor du Code erweiterst oder ausgibst. Ich muss verstehen, worum es geht.
- Liefere mir für jede Datei, an der wir arbeiten, **immer den vollständigen Code**.
- Lasse nichts aus und kürze nichts ab. Verwende keine Platzhalter wie `// ... rest of code`.
- Entferne niemals Funktionen, Felder oder Toggles, die wir bereits aufgebaut haben.
- Falls du eine Änderung oder Löschung für sinnvoll hältst, frage mich **immer zuerst**, ob ich damit einverstanden bin.
- Gib Terminal-Befehle oder Datei-Inhalte so aus, dass ich sie mit der Methode `cat << 'EOF' > dateiname` direkt in mein Terminal kopieren kann.
