# scripts/make_features.py
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
FEAT = ROOT / "data" / "features"
FEAT.mkdir(parents=True, exist_ok=True)

SYMS = ["CL", "NG"]
HORIZONS = [5, 15, 30]
WINS = [15, 30, 60, 120, 240]

def load_csv(sym: str) -> pd.DataFrame:
    p = RAW / f"{sym}.csv"
    if not p.exists():
        raise FileNotFoundError(f"Missing {p}. Run proxy_data.py first.")
    df = pd.read_csv(p)
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True, errors="coerce")
    for c in ["open","high","low","close","volume"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.dropna(subset=["timestamp","close"]).sort_values("timestamp").reset_index(drop=True)
    return df

def rsi(s: pd.Series, n=14):
    d = s.diff()
    up = d.clip(lower=0).ewm(alpha=1/n, adjust=False).mean()
    dn = (-d.clip(upper=0)).ewm(alpha=1/n, adjust=False).mean()
    rs = up / dn.replace(0, np.nan)
    return 100 - (100/(1+rs))

def stoch_k(h, l, c, n=14):
    ll = l.rolling(n).min()
    hh = h.rolling(n).max()
    return 100 * (c - ll) / (hh - ll)

def cci(h, l, c, n=20):
    tp = (h + l + c)/3
    ma = tp.rolling(n).mean()
    md = (tp - ma).abs().rolling(n).mean()
    return (tp - ma) / (0.015 * md)

def atr(h, l, c, n=14):
    pc = c.shift(1)
    tr = pd.concat([(h-l).abs(), (h-pc).abs(), (l-pc).abs()], axis=1).max(axis=1)
    return tr.rolling(n).mean()

def add_features(df: pd.DataFrame):
    out = df.copy()
    out["ret_1m"] = np.log(out["close"]).diff()
    out["rsi14"] = rsi(out["close"], 14)
    out["stoch14"] = stoch_k(out["high"], out["low"], out["close"], 14)
    out["cci20"] = cci(out["high"], out["low"], out["close"], 20)
    out["atr14"] = atr(out["high"], out["low"], out["close"], 14)
    for w in WINS:
        out[f"rv_{w}"] = out["ret_1m"].rolling(w).std()
        out[f"z_close_{w}"] = (out["close"] - out["close"].rolling(w).mean()) / out["close"].rolling(w).std()
    ts = out["timestamp"].dt.tz_convert("America/New_York")
    out["minute_of_day"] = ts.dt.hour * 60 + ts.dt.minute
    out["dow"] = ts.dt.weekday
    return out

def add_labels(df: pd.DataFrame):
    out = df.copy()
    px = out["close"]
    for h in HORIZONS:
        out[f"tgt_ret_{h}m"] = np.log(px.shift(-h) / px)
    return out

def main():
    for sym in SYMS:
        df = load_csv(sym)
        df = add_features(df)
        df = add_labels(df)
        df = df.dropna().reset_index(drop=True)
        outp = FEAT / f"{sym}.parquet"
        df.to_parquet(outp, index=False)
        print(f"[ok] features -> {outp} rows={len(df)}")

if __name__ == "__main__":
    main()
