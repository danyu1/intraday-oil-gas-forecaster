# scripts/iq_spreads_ingest.py
from __future__ import annotations
import os, sys, socket
from datetime import datetime, timedelta, timezone
from typing import List, Tuple, Optional, Dict
import pandas as pd
from dotenv import load_dotenv

load_dotenv()

HOST = os.getenv("IQFEED_HOST", "127.0.0.1")
LOOKUP_PORT = int(os.getenv("IQFEED_LOOKUP_PORT", "9100"))
BAR_SEC = int(os.getenv("IQ_BAR_SECONDS", "60"))          # source bars (1m)
RESAMPLE_SEC = int(os.getenv("RESAMPLE_SECONDS", "300"))  # output bars (5m)
HISTORY_DAYS = int(os.getenv("HISTORY_DAYS", "3650"))
DATA_DIR = os.getenv("DATA_DIR", "./data/iqfeed")
OUT_DIR = os.getenv("OUT_DIR", "./data/processed")

os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(OUT_DIR, exist_ok=True)
RAW_DIR = os.path.join(DATA_DIR, "raw")
os.makedirs(RAW_DIR, exist_ok=True)

def now_utc():
    return datetime.now(timezone.utc)

def _port_reachable() -> bool:
    s = socket.socket(); s.settimeout(2.0)
    try:
        s.connect((HOST, LOOKUP_PORT)); s.close(); return True
    except Exception as e:
        print(f"[iq] cannot reach {HOST}:{LOOKUP_PORT}: {e}", flush=True); return False

def _normalize_hist_symbol(sym: str) -> str:
    # Map chain-style roots to history-style futures roots
    if sym.startswith("QCL"): return "@CL" + sym[3:]  # QCLX25 -> @CLX25
    if sym.startswith("QNG"): return "@NG" + sym[3:]  # QNGX25 -> @NGX25
    return sym

def _recv_until_end(sock: socket.socket) -> str:
    chunks=[]
    while True:
        c = sock.recv(65535)
        if not c: break
        t = c.decode("utf-8","ignore")
        chunks.append(t)
        if "!ENDMSG!" in t: break
    return "".join(chunks)

# ---------- CFU: futures chain (raw) ----------
def chain_futures_raw(root: str) -> List[str]:
    if not _port_reachable(): return []
    cmd = f"CFU,{root}\n"
    s = socket.socket(); s.settimeout(5.0); s.connect((HOST, LOOKUP_PORT))
    # (Optional) set protocol – harmless if unsupported
    try: s.sendall(b"S,SET PROTOCOL,6.2\n")
    except Exception: pass
    s.sendall(cmd.encode())
    txt = _recv_until_end(s); s.close()
    syms=[]
    for line in txt.splitlines():
        if not line or line.startswith("!"): continue
        sym = line.split(",")[0].strip()
        if sym.upper().startswith(root.upper()): syms.append(sym)
    return syms

# ---------- HIT: historical interval bars (raw) ----------
def hit_minute_bars(symbol: str, begin_dt: datetime, end_dt: datetime, interval_sec: int = 60) -> pd.DataFrame:
    """
    HIT returns rows as:
      Timestamp, Open, Low, High, Close, TotalVolume, OpenInterest
    We use full session (leave market time filters blank) and include partial bar (1).
    """
    if not _port_reachable(): return pd.DataFrame()
    b = begin_dt.strftime("%Y%m%d %H%M%S")
    e = end_dt.strftime("%Y%m%d %H%M%S")
    # Format: HIT,<SYM>,<INTERVAL>,<BEGIN>,<END>,,,,<INCLUDE_PARTIAL>
    cmd = f"HIT,{symbol},{interval_sec},{b},{e},,,,1\n"

    s = socket.socket(); s.settimeout(10.0); s.connect((HOST, LOOKUP_PORT))
    # (Optional) ensure protocol
    try: s.sendall(b"S,SET PROTOCOL,6.2\n")
    except Exception: pass
    s.sendall(cmd.encode())
    txt = _recv_until_end(s); s.close()

    rows=[]
    for line in txt.splitlines():
        if not line or line.startswith("!"):  # !ENDMSG!
            continue
        p = line.split(",")
        if len(p) < 7:  # need at least Timestamp,O,L,H,C,Vol,OI
            continue
        ts, op, lo, hi, cl, vol = p[0], p[1], p[2], p[3], p[4], p[5]
        try:
            rows.append((
                pd.to_datetime(ts, utc=True),
                float(op), float(hi), float(lo), float(cl),
                int(float(vol))
            ))
        except Exception:
            pass

    if not rows: return pd.DataFrame()
    return pd.DataFrame(rows, columns=["timestamp","open","high","low","close","volume"])

def fetch_minute_history(symbol: str, start: datetime, end: datetime) -> pd.DataFrame:
    sym = _normalize_hist_symbol(symbol)
    df = hit_minute_bars(sym, start, end, interval_sec=BAR_SEC)
    if df is None or df.empty:
        # CSV fallback for offline testing
        path = os.path.join(RAW_DIR, f"{sym}_1m.csv")
        if os.path.exists(path):
            df = pd.read_csv(path)
            df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True, errors="coerce")
        else:
            raise RuntimeError(f"No data for {sym}. Check entitlements and symbol. (Tried HIT and no CSV at {path})")
    return df

def resample_to(df: pd.DataFrame, seconds: int) -> pd.DataFrame:
    if df is None or df.empty: return df
    df = df.sort_values("timestamp").set_index("timestamp")
    rule = f"{seconds}s"
    agg = {"open":"first","high":"max","low":"min","close":"last","volume":"sum"}
    out = df.resample(rule, label="right", closed="right").agg(agg).dropna(subset=["open","high","low","close"])
    out.reset_index(inplace=True)
    return out

def minute_to_5min(df: pd.DataFrame) -> pd.DataFrame:
    return resample_to(df, RESAMPLE_SEC)

def get_chain_symbols(root: str) -> List[str]:
    syms = chain_futures_raw(root)
    if syms: return syms
    # fallback: manually curated file
    chain_csv = os.path.join(DATA_DIR, f"{root}_chain.csv")
    if os.path.exists(chain_csv):
        ch = pd.read_csv(chain_csv)["symbol"].dropna().astype(str).tolist()
        if ch: return ch
    return []

def pick_front_two_by_volume(root: str) -> Tuple[str,str]:
    candidates = get_chain_symbols(root)
    if not candidates:
        map_path = os.path.join(DATA_DIR, f"{root}_front2.csv")
        if os.path.exists(map_path):
            sym = pd.read_csv(map_path)["symbol"].dropna().astype(str).tolist()
            if len(sym) >= 2: return sym[0], sym[1]
        raise RuntimeError(f"No chain for {root}. Create {map_path} with two symbols to override.")
    end = now_utc(); start = end - timedelta(days=2)
    vols: Dict[str,int] = {}
    for sym in candidates[:8]:
        try:
            df = fetch_minute_history(sym, start, end)
            if not df.empty:
                vols[sym] = int(df["volume"].tail(600).sum())  # ~10 hours of 1m bars
        except Exception:
            continue
    if len(vols) < 2:
        return candidates[0], candidates[1]
    ordered = sorted(vols.items(), key=lambda kv: kv[1], reverse=True)
    front, second = ordered[0][0], ordered[1][0]
    pd.DataFrame({"symbol":[front,second]}).to_csv(os.path.join(DATA_DIR, f"{root}_front2.csv"), index=False)
    return front, second

def build_spread(root: str, label: str):
    print(f"[{root}] Selecting front two contracts...")
    front, second = pick_front_two_by_volume(root)
    print(f"[{root}] Front={front}  Second={second}")

    end = now_utc(); start = end - timedelta(days=HISTORY_DAYS)

    fh = _normalize_hist_symbol(front)
    sh = _normalize_hist_symbol(second)

    print(f"[{root}] Fetching history {fh} ...", flush=True)
    f_df = fetch_minute_history(front, start, end)
    print(f"[{root}] Fetching history {sh} ...", flush=True)
    s_df = fetch_minute_history(second, start, end)

    if f_df.empty or s_df.empty:
        raise RuntimeError(f"{root}: empty history (check entitlements).")

    print(f"[{root}] Resampling to {RESAMPLE_SEC}s ...", flush=True)
    f_5 = minute_to_5min(f_df)
    s_5 = minute_to_5min(s_df)

    merged = pd.merge_asof(
        f_5.sort_values("timestamp"),
        s_5.sort_values("timestamp"),
        on="timestamp",
        direction="nearest",
        tolerance=pd.Timedelta(seconds=RESAMPLE_SEC//2),
        suffixes=(f"_{front}", f"_{second}")
    )

    merged["spread_close"] = merged[f"close_{front}"] - merged[f"close_{second}"]
    merged["spread_open"]  = merged[f"open_{front}"]  - merged[f"open_{second}"]
    merged["spread_high"]  = merged[f"high_{front}"]  - merged[f"high_{second}"]
    merged["spread_low"]   = merged[f"low_{front}"]   - merged[f"low_{second}"]
    vf = merged.get(f"volume_{front}"); vs = merged.get(f"volume_{second}")
    merged["spread_vol"] = (vf.fillna(0) if vf is not None else 0) + (vs.fillna(0) if vs is not None else 0)

    out_cols = ["timestamp","spread_open","spread_high","spread_low","spread_close","spread_vol",
                f"close_{front}", f"close_{second}"]
    merged = merged.dropna(subset=["spread_close"])[out_cols]

    out_path = os.path.join(OUT_DIR, f"{label}_SPREAD_{RESAMPLE_SEC}s.parquet")
    merged.to_parquet(out_path, index=False)
    print(f"[{root}] Wrote {out_path} rows={len(merged)}", flush=True)

def main():
    build_spread("QCL", "CL1-CL2")
    build_spread("QNG", "NG1-NG2")

if __name__ == "__main__":
    try:
        main()
    except Exception:
        print("[FATAL]"); import traceback; traceback.print_exc(); sys.exit(1)
