import pandas as pd, yfinance as yf

def grab(ticker, out_path, sym_alias):
    df = yf.download(ticker, period="7d", interval="1m", auto_adjust=False, progress=False).reset_index()
    df.rename(columns={"Datetime":"timestamp","Open":"open","High":"high","Low":"low","Close":"close","Volume":"volume"}, inplace=True)
    df["symbol"] = sym_alias
    df = df[["timestamp","symbol","open","high","low","close","volume"]]
    df.to_csv(out_path, index=False)

grab("USO", r"data/raw/CL.csv", "CL")
grab("UNG", r"data/raw/NG.csv", "NG")
print("Wrote data/raw/CL.csv and data/raw/NG.csv")