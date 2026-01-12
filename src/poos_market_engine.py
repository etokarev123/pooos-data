import os
import io
import json
import numpy as np
import pandas as pd
import boto3
from botocore.config import Config
from dotenv import load_dotenv

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

load_dotenv()

# ---------------- R2 ----------------
R2_ENDPOINT = os.environ["R2_ENDPOINT"]
R2_KEY = os.environ["R2_ACCESS_KEY_ID"]
R2_SECRET = os.environ["R2_SECRET_ACCESS_KEY"]
R2_BUCKET = os.environ["R2_BUCKET"]

DATA_PREFIX = os.getenv("DATA_PREFIX", "yahoo/1d/")   # daily bars already built
RESULT_PREFIX = os.getenv("RESULT_PREFIX", "results/poos_market_engine_v1")

# --------------- Universe / Market proxy ---------------
# If SPY exists in your daily data, we’ll use it; otherwise we build equal-weight index from universe.
MARKET_TICKER = os.getenv("MARKET_TICKER", "SPY")

# --------------- Indicators ---------------
SMA50 = int(os.getenv("SMA50", "50"))
SMA200 = int(os.getenv("SMA200", "200"))

THR_BREADTH50_ON = float(os.getenv("THR_BREADTH50_ON", "0.60"))    # 60% above SMA50
THR_BREADTH200_ON = float(os.getenv("THR_BREADTH200_ON", "0.50"))  # 50% above SMA200
THR_THRUST_10D = float(os.getenv("THR_THRUST_10D", "0.05"))        # +5pp in 10d

VOL_WINDOW = int(os.getenv("VOL_WINDOW", "20"))                    # realized vol window
VOL_MAX = float(os.getenv("VOL_MAX", "0.03"))                      # daily sigma threshold ~3% (tune)

# Scoring thresholds
SCORE_RISK_ON = int(os.getenv("SCORE_RISK_ON", "5"))
SCORE_RISK_OFF = int(os.getenv("SCORE_RISK_OFF", "2"))

# Performance controls
MAX_TICKERS = int(os.getenv("MAX_TICKERS", "0"))  # 0 = all
TAIL_DAYS = int(os.getenv("TAIL_DAYS", "0"))      # 0 = all; for OOS speed set e.g. 600

# ---------------- R2 client ----------------
s3 = boto3.client(
    "s3",
    endpoint_url=R2_ENDPOINT,
    aws_access_key_id=R2_KEY,
    aws_secret_access_key=R2_SECRET,
    region_name="auto",
    config=Config(signature_version="s3v4", retries={"max_attempts": 10, "mode": "standard"}),
)

def list_tickers(prefix: str):
    out = []
    token = None
    while True:
        kwargs = dict(Bucket=R2_BUCKET, Prefix=prefix, MaxKeys=1000)
        if token:
            kwargs["ContinuationToken"] = token
        resp = s3.list_objects_v2(**kwargs)
        for obj in resp.get("Contents", []):
            key = obj["Key"]
            if key.endswith(".parquet"):
                out.append(key.split("/")[-1].replace(".parquet", ""))
        if resp.get("IsTruncated"):
            token = resp.get("NextContinuationToken")
        else:
            break
    return sorted(set(out))

def read_parquet(key: str) -> pd.DataFrame:
    obj = s3.get_object(Bucket=R2_BUCKET, Key=key)
    return pd.read_parquet(io.BytesIO(obj["Body"].read()))

def put_bytes(key: str, data: bytes, content_type="application/octet-stream"):
    s3.put_object(Bucket=R2_BUCKET, Key=key, Body=data, ContentType=content_type)

def put_text(key: str, text: str, content_type="text/plain; charset=utf-8"):
    put_bytes(key, text.encode("utf-8"), content_type=content_type)

def sma(s: pd.Series, n: int) -> pd.Series:
    return s.rolling(n, min_periods=n).mean()

def build_panel(tickers: list[str]) -> pd.DataFrame:
    """
    Output: panel with columns [date, ticker, close, volume, ret_1d, above50, above200, adv]
    where adv = 1 if ret_1d>0 else -1 if ret_1d<0 else 0
    """
    frames = []
    for i, t in enumerate(tickers, 1):
        try:
            df = read_parquet(f"{DATA_PREFIX}{t}.parquet")
            if df is None or df.empty:
                continue

            need = {"date", "close", "volume"}
            if not need.issubset(set(df.columns)):
                continue

            d = df[["date", "close", "volume"]].copy()
            d["date"] = pd.to_datetime(d["date"], utc=True)
            d = d.sort_values("date")

            if TAIL_DAYS > 0 and len(d) > TAIL_DAYS:
                d = d.iloc[-TAIL_DAYS:].copy()

            d["ret_1d"] = d["close"].pct_change(1)
            d["sma50"] = sma(d["close"], SMA50)
            d["sma200"] = sma(d["close"], SMA200)

            d["above50"] = (d["close"] > d["sma50"]).astype(int)
            d["above200"] = (d["close"] > d["sma200"]).astype(int)

            d["adv"] = np.where(d["ret_1d"] > 0, 1, np.where(d["ret_1d"] < 0, -1, 0))

            d["ticker"] = t
            frames.append(d[["date","ticker","close","volume","ret_1d","above50","above200","adv"]])

            if i % 50 == 0:
                print(f"Loaded {i}/{len(tickers)}")
        except Exception as e:
            print("Error loading", t, ":", e)

    if not frames:
        return pd.DataFrame()

    panel = pd.concat(frames, ignore_index=True)
    panel = panel.dropna(subset=["date", "close", "volume"])
    return panel

def build_market_proxy(panel: pd.DataFrame, market_ticker: str):
    """
    Returns market series df with columns:
    [date, m_close, m_ret, m_sma50, m_sma200, m_trend_up, m_vol20]
    Uses SPY if present, else equal-weight index across tickers.
    """
    panel = panel.copy()
    dates = pd.to_datetime(sorted(panel["date"].unique()), utc=True)

    if market_ticker in set(panel["ticker"].unique()):
        m = panel[panel["ticker"] == market_ticker].copy()
        m = m.sort_values("date")
        m = m[["date","close"]].rename(columns={"close":"m_close"})
        m["m_ret"] = m["m_close"].pct_change(1)
    else:
        # Equal-weight proxy: average of daily returns across tickers
        r = panel.pivot_table(index="date", columns="ticker", values="ret_1d")
        r = r.reindex(dates)
        ew_ret = r.mean(axis=1, skipna=True)
        m = pd.DataFrame({"date": ew_ret.index, "m_ret": ew_ret.values})
        # convert returns to pseudo-price
        m["m_close"] = (1.0 + m["m_ret"].fillna(0.0)).cumprod()

    m["m_sma50"] = sma(m["m_close"], SMA50)
    m["m_sma200"] = sma(m["m_close"], SMA200)
    m["m_trend_up"] = ((m["m_close"] > m["m_sma50"]) & (m["m_close"] > m["m_sma200"])).astype(int)

    m["m_vol20"] = m["m_ret"].rolling(VOL_WINDOW, min_periods=VOL_WINDOW).std(ddof=0)
    m["m_vol_ok"] = (m["m_vol20"] <= VOL_MAX).astype(int)

    return m

def main():
    tickers = list_tickers(DATA_PREFIX)
    if not tickers:
        raise SystemExit("No daily data in R2 under yahoo/1d/. Build daily first.")
    if MAX_TICKERS > 0:
        tickers = tickers[:MAX_TICKERS]

    print("Tickers:", len(tickers))

    panel = build_panel(tickers)
    if panel.empty:
        raise SystemExit("Panel empty — not enough data or schema mismatch.")

    # --- Breadth + A/D + Up/Down volume ---
    # Breadth50/200 = mean(aboveX) across tickers per day
    g = panel.groupby("date")

    breadth50 = g["above50"].mean()
    breadth200 = g["above200"].mean()
    ad = g["adv"].sum()  # advance - decline (net)
    ad_line = ad.cumsum()

    # Up/Down volume
    def up_down_vol(df):
        up = df.loc[df["ret_1d"] > 0, "volume"].sum()
        dn = df.loc[df["ret_1d"] < 0, "volume"].sum()
        return pd.Series({"up_vol": up, "down_vol": dn})

    ud = g.apply(up_down_vol)
    ud["ud_ratio"] = np.where(ud["down_vol"] > 0, ud["up_vol"] / ud["down_vol"], np.nan)
    ud["ud_ratio_smooth"] = ud["ud_ratio"].rolling(5, min_periods=5).mean()

    # Breadth thrust: change in breadth50 over 10 days
    breadth50_thrust_10d = breadth50 - breadth50.shift(10)

    # Market proxy series
    m = build_market_proxy(panel, MARKET_TICKER).set_index("date")

    # Combine
    df = pd.DataFrame({
        "breadth50": breadth50,
        "breadth200": breadth200,
        "breadth50_thrust_10d": breadth50_thrust_10d,
        "ad_net": ad,
        "ad_line": ad_line,
        "up_vol": ud["up_vol"],
        "down_vol": ud["down_vol"],
        "ud_ratio": ud["ud_ratio"],
        "ud_ratio_smooth": ud["ud_ratio_smooth"],
        "m_close": m["m_close"],
        "m_ret": m["m_ret"],
        "m_trend_up": m["m_trend_up"],
        "m_vol20": m["m_vol20"],
        "m_vol_ok": m["m_vol_ok"],
    }).dropna(subset=["breadth50","breadth200","m_trend_up","m_vol_ok"], how="any")

    # --- Build POOS-style market score ---
    df["sig_breadth50"] = (df["breadth50"] >= THR_BREADTH50_ON).astype(int)
    df["sig_breadth200"] = (df["breadth200"] >= THR_BREADTH200_ON).astype(int)
    df["sig_thrust"] = (df["breadth50_thrust_10d"] >= THR_THRUST_10D).astype(int)
    df["sig_ad"] = (df["ad_net"] > 0).astype(int)
    df["sig_ud"] = (df["ud_ratio_smooth"] > 1.0).astype(int)
    df["sig_trend"] = df["m_trend_up"].astype(int)
    df["sig_vol"] = df["m_vol_ok"].astype(int)

    df["market_score"] = (
        df["sig_breadth50"] +
        df["sig_breadth200"] +
        df["sig_thrust"] +
        df["sig_ad"] +
        df["sig_ud"] +
        df["sig_trend"] +
        df["sig_vol"]
    )

    def classify(score):
        if score >= SCORE_RISK_ON:
            return "risk_on"
        if score <= SCORE_RISK_OFF:
            return "risk_off"
        return "neutral"

    df["market_state"] = df["market_score"].apply(classify)

    # Save CSV
    out_state = df.reset_index()[["date","market_state","market_score"]]
    out_components = df.reset_index()

    put_text(f"{RESULT_PREFIX}/market_state.csv", out_state.to_csv(index=False))
    put_text(f"{RESULT_PREFIX}/components.csv", out_components.to_csv(index=False))

    stats = {
        "mode": "POOS Market Engine v1 (breadth + AD + up/down vol + trend + vol)",
        "tickers_used": int(panel["ticker"].nunique()),
        "market_ticker_preference": MARKET_TICKER,
        "used_spy": bool(MARKET_TICKER in set(panel["ticker"].unique())),
        "thresholds": {
            "breadth50_on": THR_BREADTH50_ON,
            "breadth200_on": THR_BREADTH200_ON,
            "thrust_10d": THR_THRUST_10D,
            "vol_window": VOL_WINDOW,
            "vol_max": VOL_MAX,
            "score_risk_on": SCORE_RISK_ON,
            "score_risk_off": SCORE_RISK_OFF,
        },
        "state_counts": df["market_state"].value_counts().to_dict(),
        "date_start": str(out_state["date"].min()),
        "date_end": str(out_state["date"].max()),
    }
    put_text(f"{RESULT_PREFIX}/stats.json", json.dumps(stats, indent=2))

    # Plot breadth
    plt.figure()
    plt.plot(df.index, df["breadth50"], label="%>SMA50")
    plt.plot(df.index, df["breadth200"], label="%>SMA200")
    plt.legend()
    plt.title("POOS Market Engine: Breadth")
    plt.xticks(rotation=30, ha="right")
    plt.tight_layout()
    bbuf = io.BytesIO()
    plt.savefig(bbuf, format="png", dpi=160)
    plt.close()
    put_bytes(f"{RESULT_PREFIX}/breadth.png", bbuf.getvalue(), content_type="image/png")

    # Plot state as color bands (simple)
    plt.figure()
    y = df["market_score"].values
    plt.plot(df.index, y)
    plt.title("POOS Market Engine: Market Score")
    plt.xticks(rotation=30, ha="right")
    plt.tight_layout()
    sbuf = io.BytesIO()
    plt.savefig(sbuf, format="png", dpi=160)
    plt.close()
    put_bytes(f"{RESULT_PREFIX}/score.png", sbuf.getvalue(), content_type="image/png")

    print("Saved to R2:", RESULT_PREFIX)
    print("Stats:", stats)

if __name__ == "__main__":
    main()
