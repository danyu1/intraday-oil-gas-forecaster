# scripts/ensemble_diagnostics.py
from pathlib import Path
import numpy as np, pandas as pd, re

ROOT = Path(__file__).resolve().parents[1]
OUT  = ROOT / "outputs"

SYMBOL = "CL"
HORIZONS = [5, 15, 30]

def sharpe(x):
    x = np.asarray(x); x = x[np.isfinite(x)]
    if x.size < 2: return np.nan
    return float(np.nanmean(x) / (np.nanstd(x, ddof=1) + 1e-9))

def _read_oof(path: Path, name: str) -> pd.DataFrame:
    df = pd.read_parquet(path)
    if "ds" not in df or "pred" not in df or "y" not in df:
        raise ValueError(f"{path} must have columns ['ds','y','pred']")
    df = df.copy()
    df["ds"] = pd.to_datetime(df["ds"], utc=True, errors="coerce")
    df = df.dropna(subset=["ds","y","pred"])
    return df.rename(columns={"pred": name})

def merged_oof(symbol: str, h: int):
    frames = []
    # xgb first (prefer its y if needed)
    p = OUT / f"oof_xgb_{symbol}_{h}m.parquet"
    if p.exists(): frames.append(_read_oof(p, "XGB"))
    for fp in OUT.glob(f"oof_dl_*_{symbol}_{h}m.parquet"):
        m = re.match(rf"oof_dl_(.+)_{symbol}_{h}m\.parquet", fp.name)
        name = (m.group(1).upper() if m else "DL")
        frames.append(_read_oof(fp, name))
    if not frames: return None
    df = frames[0]
    for f in frames[1:]:
        df = pd.merge(df, f.drop(columns=["y"]), on="ds", how="inner")
    df = df.sort_values("ds").reset_index(drop=True)
    return df

def best_two_weight(p1, p2):
    ws = np.linspace(0, 1, 101)
    best = (0.5, -1e9)
    for w in ws:
        s = sharpe(w*p1 + (1-w)*p2)
        if s > best[1]: best = (w, s)
    return best

def main():
    for h in HORIZONS:
        df = merged_oof(SYMBOL, h)
        if df is None or len(df) < 50:
            print(f"[{SYMBOL} {h}m] no/too few OOF rows"); continue
        cols = [c for c in df.columns if c not in ("ds","y")]
        y = df["y"].to_numpy()
        print(f"\n=== {SYMBOL} {h}m | rows={len(df)} | models={cols} ===")
        for c in cols:
            print(f"  {c:>4} OOF Sharpe: {sharpe(df[c]): .3f}")
        if len(cols) >= 2:
            corr = np.corrcoef(df[cols].to_numpy().T)
            print("  Corr matrix:")
            print(pd.DataFrame(corr, index=cols, columns=cols).round(3))
            if len(cols) == 2:
                w, s = best_two_weight(df[cols[0]].to_numpy(), df[cols[1]].to_numpy())
                print(f"  Sharpe-max weights → {cols[0]}={w:.2f}, {cols[1]}={1-w:.2f} | Sharpe≈{s:.3f}")

if __name__ == "__main__":
    main()
