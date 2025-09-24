# scripts/fit_ensemble_sharpe.py
from pathlib import Path
import numpy as np, pandas as pd, re

ROOT = Path(__file__).resolve().parents[1]
OUT  = ROOT / "outputs"; OUT.mkdir(parents=True, exist_ok=True)

SYMBOL = "CL"
HORIZONS = [5, 15, 30]
SAMPLES_KGE3 = 4000

def sharpe(a):
    a = np.asarray(a); a = a[np.isfinite(a)]
    if a.size < 2: return -1e9
    return float(np.nanmean(a) / (np.nanstd(a, ddof=1) + 1e-9))

def _read_oof(path: Path, name: str):
    df = pd.read_parquet(path)
    for c in ("ds","y","pred"):
        if c not in df: raise ValueError(f"{path} missing {c}")
    df = df.copy()
    df["ds"] = pd.to_datetime(df["ds"], utc=True, errors="coerce")
    df = df.dropna(subset=["ds","y","pred"])
    return df.rename(columns={"pred": name})[["ds","y",name]]

def load_merged(symbol, h):
    frames = []
    p = OUT / f"oof_xgb_{symbol}_{h}m.parquet"
    if p.exists(): frames.append(_read_oof(p, "XGB"))
    for fp in OUT.glob(f"oof_dl_*_{symbol}_{h}m.parquet"):
        m = re.match(rf"oof_dl_(.+)_{symbol}_{h}m\.parquet", fp.name)
        name = (m.group(1).upper() if m else "DL")
        frames.append(_read_oof(fp, name))
    if not frames: return None, None, None
    df = frames[0]
    for f in frames[1:]:
        df = pd.merge(df, f, on=["ds","y"], how="inner")
    df = df.sort_values("ds").reset_index(drop=True)
    cols = [c for c in df.columns if c not in ("ds","y")]
    P = df[cols].to_numpy(dtype=float); y = df["y"].to_numpy(dtype=float)
    mask = np.isfinite(y)
    for j in range(P.shape[1]): mask &= np.isfinite(P[:,j])
    return cols, P[mask], y[mask], df.loc[mask, "ds"].to_numpy()

def rand_simplex(n, m):
    x = np.random.gamma(1.0, 1.0, size=(m, n))
    x /= x.sum(axis=1, keepdims=True)
    return x

def main():
    np.random.seed(42)
    rows = []
    for h in HORIZONS:
        names, P, y, ds = load_merged(SYMBOL, h)
        if names is None or len(y) < 10:
            print(f"[warn] no aligned OOF for {SYMBOL} {h}m"); continue
        k = P.shape[1]
        best_w, best_s = None, -1e9
        if k == 1:
            best_w, best_s = np.array([1.0]), sharpe(P[:,0])
        elif k == 2:
            for w in np.linspace(0,1,201):
                s = sharpe(w*P[:,0] + (1-w)*P[:,1])
                if s > best_s: best_w, best_s = np.array([w,1-w]), s
        else:
            for w in rand_simplex(k, SAMPLES_KGE3):
                s = sharpe(P @ w)
                if s > best_s: best_w, best_s = w, s
        yhat = P @ best_w
        pd.DataFrame({"ds": ds, "y": y, "yhat": yhat}).to_parquet(OUT / f"ens_{SYMBOL}_{h}m.parquet", index=False)
        print(f"[ok] {SYMBOL} {h}m | models={names} | w={np.round(best_w,3)} | Sharpe≈{best_s:.3f} | rows={len(y)}")
        rows.append({"h": h, "models": names, "weights": np.round(best_w,3).tolist(), "sharpe_proxy": round(best_s,3)})
    if rows:
        pd.DataFrame(rows).to_parquet(OUT / f"ensemble_summary_{SYMBOL}.parquet", index=False)
    else:
        print("[warn] no ensemble summaries created.")

if __name__ == "__main__":
    main()
