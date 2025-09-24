# scripts/fit_ensemble.py
from pathlib import Path
import numpy as np
import pandas as pd
import re

ROOT = Path(__file__).resolve().parents[1]
OUT  = ROOT / "outputs"; OUT.mkdir(parents=True, exist_ok=True)

SYMBOL = "CL"
HORIZONS = [5, 15, 30]
ALPHA = 0.1
NONNEG = True

def sharpe(a):
    a = np.asarray(a); a = a[np.isfinite(a)]
    if a.size < 2: return float("nan")
    return float(np.nanmean(a) / (np.nanstd(a, ddof=1) + 1e-9))

def ridge(P, y, alpha, nonneg):
    A = P.T @ P + alpha * np.eye(P.shape[1]); b = P.T @ y
    try:
        w = np.linalg.solve(A, b)
    except np.linalg.LinAlgError:
        w = np.linalg.pinv(A) @ b
    if nonneg:
        w = np.clip(w, 0, None)
        if w.sum() == 0: w = np.ones_like(w)
    return w / (w.sum() + 1e-12)

def _read_oof(path: Path, model_name: str):
    """Load an OOF parquet and return:
       - df_pred: DataFrame with columns ['ds', <model_name>] (ds tz-aware UTC)
       - df_y:    DataFrame with columns ['ds','y'] if present, else None
    """
    df = pd.read_parquet(path)
    if "ds" not in df.columns or "pred" not in df.columns:
        raise ValueError(f"{path} missing required columns 'ds' and/or 'pred'")
    ds = pd.to_datetime(df["ds"], utc=True, errors="coerce")
    df_pred = pd.DataFrame({"ds": ds, model_name: pd.to_numeric(df["pred"], errors="coerce")})
    df_y = None
    if "y" in df.columns:
        df_y = pd.DataFrame({"ds": ds, "y": pd.to_numeric(df["y"], errors="coerce")})
    return df_pred.dropna(subset=["ds"]), (df_y.dropna(subset=["ds"]) if df_y is not None else None)

def load_oof(symbol: str, h: int):
    # Collect model frames
    model_frames = []
    y_sources = []

    # XGB first (prefer its y)
    p = OUT / f"oof_xgb_{symbol}_{h}m.parquet"
    if p.exists():
        df_pred, df_y = _read_oof(p, "XGB")
        model_frames.append(("XGB", df_pred))
        if df_y is not None and not df_y.empty:
            y_sources.append(df_y)

    # DL models (any oof_dl_*)
    for fp in OUT.glob(f"oof_dl_*_{symbol}_{h}m.parquet"):
        # model name from filename
        m = re.match(rf"oof_dl_(.+)_{symbol}_{h}m\.parquet", fp.name)
        name = (m.group(1).upper() if m else "DL")
        df_pred, df_y = _read_oof(fp, name)
        model_frames.append((name, df_pred))
        if df_y is not None and not df_y.empty:
            y_sources.append(df_y)

    if not model_frames:
        return [], None, None, None

    # Merge all predictions on ds (inner)
    merged = model_frames[0][1]
    ordered_names = [model_frames[0][0]]
    for name, frm in model_frames[1:]:
        merged = pd.merge(merged, frm, on="ds", how="inner")
        ordered_names.append(name)

    merged = merged.sort_values("ds").reset_index(drop=True)

    # Attach y from the first available y_source (prefer XGB which we added first)
    if not y_sources:
        raise ValueError("No 'y' column found in any OOF file; re-run train_xgb.py to generate OOF with y.")
    y_merged = y_sources[0]
    merged = pd.merge(merged, y_merged, on="ds", how="inner")

    # Build P, y, ds
    P = merged[ordered_names].to_numpy(dtype=float)
    y = merged["y"].to_numpy(dtype=float)
    ds = merged["ds"].to_numpy()

    # Final NaN mask
    mask = np.isfinite(y)
    for j in range(P.shape[1]):
        mask &= np.isfinite(P[:, j])
    return ordered_names, P[mask], y[mask], ds[mask]

def main():
    rows = []
    for h in HORIZONS:
        try:
            names, P, y, ds = load_oof(SYMBOL, h)
        except Exception as e:
            print(f"[warn] {SYMBOL} {h}m | {e}")
            continue

        if P is None or len(y) < 10:
            print(f"[warn] no aligned OOF for {SYMBOL} {h}m"); continue

        w = ridge(P, y, ALPHA, NONNEG)
        yhat = P @ w
        sr = sharpe(yhat)

        ens_path = OUT / f"ens_{SYMBOL}_{h}m.parquet"
        pd.DataFrame({"ds": ds, "y": y, "yhat": yhat}).to_parquet(ens_path, index=False)

        print(f"[ok] {SYMBOL} {h}m | models={names} | w={np.round(w,3)} | Sharpe≈{sr:.3f} | rows={len(y)}")
        rows.append({"h": h, "models": names, "weights": np.round(w,3).tolist(), "sharpe_proxy": round(sr,3)})

    if rows:
        pd.DataFrame(rows).to_parquet(OUT / f"ensemble_summary_{SYMBOL}.parquet", index=False)
    else:
        print("[warn] no ensemble summaries created.")

if __name__ == "__main__":
    main()
