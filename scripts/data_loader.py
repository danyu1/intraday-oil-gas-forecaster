from __future__ import annotations
import pandas as pd
from pathlib import Path
from typing import Dict, Optional
from .constants import SPREAD_FILES

REQUIRED_COLS = ["timestamp", "open", "high", "low", "close", "volume"]


def _coerce_schema(df: pd.DataFrame) -> pd.DataFrame:
    """Ensure required columns exist and dtypes are reasonable."""
    df = df.copy()
    if "timestamp" not in df.columns:
        raise ValueError("DataFrame must contain a 'timestamp' column")
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True, errors="coerce")
    # Minimal numeric coercion
    for c in ["open", "high", "low", "close", "volume"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
        else:
            raise ValueError(f"Missing required column: {c}")
    df = df.dropna(subset=["timestamp", "open", "high", "low", "close"]).sort_values("timestamp")
    return df


def load_series(symbol: str, path_override: Optional[str] = None) -> pd.DataFrame:
    """
    Load time series for either a spread (virtual symbol) or a standard asset.

    - For spreads, reads the Parquet produced by iq_spreads_ingest.py
      and renames columns to OHLCV schema expected by training scripts.
    - For non-spreads, tries to read a Parquet/CSV at path_override.
    """
    # Spread path
    if symbol in SPREAD_FILES and path_override is None:
        p = SPREAD_FILES[symbol]
        if not p.exists():
            raise FileNotFoundError(f"Expected Parquet not found for {symbol}: {p}")
        df = pd.read_parquet(p)
        # Map spread_* → OHLCV
        colmap = {
            "spread_open": "open",
            "spread_high": "high",
            "spread_low": "low",
            "spread_close": "close",
            "spread_vol": "volume",
        }
        # If ingest also saved leg closes, keep them as features if desired later
        df = df.rename(columns=colmap)
        df = _coerce_schema(df)
        return df

    # Non-spread or explicit override
    if path_override is None:
        raise ValueError("Non-spread symbols require path_override to a Parquet/CSV file.")

    p = Path(path_override)
    if not p.exists():
        raise FileNotFoundError(f"Data file not found: {p}")

    if p.suffix.lower() in (".parquet", ".pq"):
        df = pd.read_parquet(p)
    else:
        df = pd.read_csv(p)

    df = _coerce_schema(df)
    return df


def add_returns_and_targets(df: pd.DataFrame, horizons=(1,3,6)) -> pd.DataFrame:
    df = df.copy()
    df["ret_5m"] = df["close"].pct_change()
    for h in horizons:
        df[f"fwd_ret_{h}"] = df["close"].pct_change(periods=h).shift(-h)
    return df.dropna().reset_index(drop=True)