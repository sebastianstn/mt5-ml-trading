"""
external_api.py – Externe Features (Fear & Greed Index, BTC Funding Rate)

Holt Marktsentiment-Daten von externen APIs mit Fallback-Werten bei Fehler.
Läuft auf: Windows 11 Laptop
"""

import logging
import time
from typing import Optional

import pandas as pd
import requests

logger = logging.getLogger(__name__)


def fear_greed_holen() -> dict:
    """
    Holt den aktuellen Fear & Greed Index von alternative.me.

    Returns:
        Dict mit 'fear_greed_value' (0–100) und 'fear_greed_class' (0–3).
        Fallback-Werte 50 und 1 bei API-Fehler.
    """
    fallback = {"fear_greed_value": 50.0, "fear_greed_class": 1.0}
    try:
        resp = requests.get(
            "https://api.alternative.me/fng/?limit=1&format=json",
            timeout=10,
        )
        resp.raise_for_status()
        daten = resp.json().get("data", [{}])[0]
        wert = float(daten.get("value", 50))
        # Klassifizierung: 0=Extreme Fear, 1=Fear, 2=Greed, 3=Extreme Greed
        if wert < 25:
            klasse = 0.0
        elif wert < 50:
            klasse = 1.0
        elif wert < 75:
            klasse = 2.0
        else:
            klasse = 3.0
        logger.info(f"Fear & Greed: {wert:.0f} (Klasse {klasse:.0f})")
        return {"fear_greed_value": wert, "fear_greed_class": klasse}
    except (requests.RequestException, KeyError, ValueError) as e:
        logger.warning(f"Fear & Greed API Fehler: {e} – Fallback 50/1")
        return fallback


def btc_funding_holen() -> float:
    """
    Holt die aktuelle BTC Funding Rate von Binance Futures.

    Returns:
        Funding Rate als Float (z.B. 0.0001 = 0.01%).
        Fallback-Wert 0.0 bei API-Fehler.
    """
    try:
        resp = requests.get(
            "https://fapi.binance.com/fapi/v1/premiumIndex",
            params={"symbol": "BTCUSDT"},
            timeout=10,
        )
        resp.raise_for_status()
        rate = float(resp.json().get("lastFundingRate", 0.0))
        logger.info(f"BTC Funding Rate: {rate:.6f}")
        return rate
    except (requests.RequestException, KeyError, ValueError) as e:
        logger.warning(f"BTC Funding Rate API Fehler: {e} – Fallback 0.0")
        return 0.0


def externe_features_holen(
    cache: Optional[dict] = None, max_age_seconds: int = 240
) -> dict:
    """
    Holt externe Features optional mit Cache, um API-Rauschen zu reduzieren.

    Args:
        cache: Optionales Cache-Dict, das zwischen Loop-Durchläufen weitergegeben wird.
        max_age_seconds: Maximales Alter der Cache-Werte in Sekunden.

    Returns:
        Dict mit fear_greed_value, fear_greed_class, btc_funding_rate.
    """
    now_ts = time.time()
    if cache is not None:
        cached_ts = float(cache.get("fetched_at", 0.0))
        if cached_ts > 0 and (now_ts - cached_ts) <= max_age_seconds:
            return {
                "fear_greed_value": float(cache.get("fear_greed_value", 50.0)),
                "fear_greed_class": float(cache.get("fear_greed_class", 1.0)),
                "btc_funding_rate": float(cache.get("btc_funding_rate", 0.0)),
            }

    fg = fear_greed_holen()
    btc_rate = btc_funding_holen()
    values = {
        "fear_greed_value": float(fg["fear_greed_value"]),
        "fear_greed_class": float(fg["fear_greed_class"]),
        "btc_funding_rate": float(btc_rate),
    }
    if cache is not None:
        cache.update(values)
        cache["fetched_at"] = now_ts
    return values


def externe_features_einfuegen(
    df: pd.DataFrame, external_features: Optional[dict] = None
) -> pd.DataFrame:
    """
    Fügt Fear & Greed und BTC Funding Rate als Features ein.

    Args:
        df: Feature-DataFrame (bereits mit technischen Indikatoren)
        external_features: Optional bereits geholte externe Features (für Loop-Caching)

    Returns:
        DataFrame mit 3 zusätzlichen Spalten.
    """
    if external_features is None:
        external_features = externe_features_holen()

    df["fear_greed_value"] = float(external_features.get("fear_greed_value", 50.0))
    df["fear_greed_class"] = float(external_features.get("fear_greed_class", 1.0))
    df["btc_funding_rate"] = float(external_features.get("btc_funding_rate", 0.0))
    return df
