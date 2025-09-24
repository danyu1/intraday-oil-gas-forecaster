# scripts/proxy_data.py
from pathlib import Path
import pandas as pd
import yfinance as yf

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
RAW.mkdir(parents=True, exist_ok=True)

def grab(ticker: str, out_path: Path, symbol_alias: str):
    df = yf.download(ticker, period="60d", interval="5m", auto_adjust=False, progress=False)
    if df.empty:
        raise SystemExit(f"Yahoo returned no data for {ticker}. Try during market hours, or switch to 5m/60d.")
    df = df.reset_index().rename(columns={
        "Datetime":"timestamp","Open":"open","High":"high","Low":"low","Close":"close","Volume":"volume"
    })
    df["symbol"] = symbol_alias
    df = df[["timestamp","symbol","open","high","low","close","volume"]]
    # Ensure UTC timestamps
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    df.to_csv(out_path, index=False)
    print(f"[ok] wrote {out_path} rows={len(df)}")

def main():
    grab("USO", RAW / "CL.csv", "CL")   # WTI proxy
    grab("UNG", RAW / "NG.csv", "NG")   # NatGas proxy

if __name__ == "__main__":
    main()
