"""Tests für die MT5-Common-Files-Synchronisierung im Live-Trader."""

from pathlib import Path

from live import live_trader


RESOLVE_MT5_COMMON_FILES_DIR = getattr(live_trader, "_resolve_mt5_common_files_dir")
MIRROR_CSV_TO_MT5_COMMON = getattr(live_trader, "_mirror_csv_to_mt5_common")


def test_resolve_mt5_common_files_dir_prefers_override(monkeypatch) -> None:
    """Expliziter Override soll Vorrang vor APPDATA haben."""
    override_dir = Path("/tmp/custom_mt5_common")
    monkeypatch.setenv(live_trader.MT5_COMMON_FILES_ENV, str(override_dir))
    monkeypatch.setenv("APPDATA", "/tmp/appdata_should_not_win")

    resolved = RESOLVE_MT5_COMMON_FILES_DIR()

    assert resolved == override_dir


def test_resolve_mt5_common_files_dir_uses_appdata(monkeypatch) -> None:
    """Ohne Override soll APPDATA auf den Standardordner abgebildet werden."""
    monkeypatch.delenv(live_trader.MT5_COMMON_FILES_ENV, raising=False)
    monkeypatch.setenv("APPDATA", "/tmp/roaming")

    resolved = RESOLVE_MT5_COMMON_FILES_DIR()

    assert resolved == Path("/tmp/roaming/MetaQuotes/Terminal/Common/Files")


def test_mirror_csv_to_mt5_common_copies_latest_file(
    tmp_path: Path, monkeypatch
) -> None:
    """Die aktive CSV soll vollständig nach MT5 Common/Files gespiegelt werden."""
    local_csv = tmp_path / "logs" / "paper_test128" / "USDCAD_signals.csv"
    local_csv.parent.mkdir(parents=True, exist_ok=True)
    local_csv.write_text("time,symbol\n2026-03-10 19:35:09,USDCAD\n", encoding="utf-8")

    mt5_common_dir = tmp_path / "mt5_common"
    monkeypatch.setenv(live_trader.MT5_COMMON_FILES_ENV, str(mt5_common_dir))
    monkeypatch.delenv("APPDATA", raising=False)

    ok = MIRROR_CSV_TO_MT5_COMMON(local_csv, "USDCAD_signals.csv")

    target_csv = mt5_common_dir / "USDCAD_signals.csv"
    assert ok is True
    assert target_csv.exists()
    assert target_csv.read_text(encoding="utf-8") == local_csv.read_text(
        encoding="utf-8"
    )
