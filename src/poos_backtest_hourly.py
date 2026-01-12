import os
import io
import json
import math
import pandas as pd
import numpy as np
import boto3
from botocore.config import Config
from dotenv import load_dotenv
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

load_dotenv()

# ---------------- ENV (R2) ----------------
R2_ENDPOINT = os.environ["R2_ENDPOINT"]
R2_KEY = os.environ["R2_ACCESS_KEY_ID"]
R2_SECRET = os.environ["R2_SECRET_ACCESS_KEY"]
R2_BUCKET = os.environ["R2_BUCKET"]

# -------------- ENV (Strategy) ------------
# "Rocket"/impulse on hourly bars:
LOOKBACK_BARS = int(os.getenv("LOOKBACK_BARS", "35"))          # ~5 trading days if ~7 bars/day
IMPULSE_MIN_RET = float(os.getenv("IMPULSE_MIN_RET", "0.30"))  # 30%
VOL_MA_BARS = int(os.getenv("VOL_MA_BARS", "140"))             # ~20 trading days * 7 bars
VOL_MULT = float(os.getenv("VOL_MULT", "2.0"))

# Trend filter
USE_TREND_FILTER = os.getenv("USE_TREND_FILTER", "1") == "1"

# Entry/exit
ATR_PERIOD = int(os.getenv("ATR_PERIOD", "14"))
STOP_ATR = float(os.getenv("STOP_ATR", "1.0"))
TP_ATR = float(os.getenv("TP_ATR", "1.5"))
MOVE_BE_PCT = float(os.getenv("MOVE_BE_PCT", "0.01"))          # +1% -> BE
MAX_HOLD_BARS = int(os.getenv("MAX_HOLD_BARS", "300"))         # ~40 trading days
COOLOFF_BARS = int(os.getenv("COOLOFF_BARS", "140"))           # ~20 days

# Portfolio approximation (trade-by-trade compounding)
RISK_PER_TRADE = float(os.getenv("RISK_PER_TRADE", "0.01"))    # 1% equity per trade

# Optional: only backtest last N days of hourly bars (for speed). 0 = all.
TAIL_BARS = int(os.getenv("TAIL_BARS", "0"))

RESULT_PREFIX = os.getenv("RESULT_PREFIX", "results/poos_hourly")

# ---------------- R2 client ----------------
s3 = boto3.client(
    "s3",
    endpoint_url=R2_ENDPOINT,
    aws_access_key_id=R2_KEY,
    aws_secret_access_key=R2_SECRET,
    region_name="auto",
    config=Config(signature_version="s3v4", retries={"max_attempts": 10, "mode": "standard"}),
)

def list_parquet_tickers(prefix="yahoo/1h/"):
    tickers = []
    token = None
    while True:
        kwargs = dict(Bucket=R2_BUCKET, Prefix=prefix, MaxKeys=1000)
        if token:
            kwargs["ContinuationToken"] = token
        resp = s3.list_objects_v2(**kwargs)
        for obj in resp.get("Contents", []):
            key = obj["Key"]
            if key.endswith(".parquet"):
                t = key.split("/")[-1].replace(".parquet", "")
                tickers.append(t)
        if resp.get("IsTruncated"):
            token = resp.get("NextContinuationToken")
        else:
            break
    return sorted(set(tickers))

def read_parquet_from_r2(key: str) -> pd.DataFrame:
    obj = s3.get_object(Bucket=R2_BUCKET, Key=key)
    body = obj["Body"].read()
    return pd.read_parquet(io.BytesIO(body))

def put_bytes(key: str, data: bytes, content_type="application/octet-stream"):
    s3.put_object(Bucket=R2_BUCKET, Key=key, Body=data, ContentType=content_type)

def put_text(key: str, text: str, content_type="text/plain; charset=utf-8"):
    put_bytes(key, text.encode("utf-8"), content_type=content_type)

def ema(series: pd.Series, span: int) -> pd.Series:
    return series.ewm(span=span, adjust=False).mean()

def atr(df: pd.DataFrame, period: int) -> pd.Series:
    prev_close = df["close"].shift(1)
    tr = pd.concat([
        (df["high"] - df["low"]).abs(),
        (df["high"] - prev_close).abs(),
        (df["low"] - prev_close).abs(),
    ], axis=1).max(axis=1)
    return tr.rolling(period, min_periods=period).mean()

def backtest_one(df: pd.DataFrame, ticker: str):
    # Expect columns: date, open, high, low, close, volume
    df = df.sort_values("date").dropna(subset=["open","high","low","close","volume"]).copy()
    if TAIL_BARS > 0 and len(df) > TAIL_BARS:
        df = df.iloc[-TAIL_BARS:].copy()

    # indicators
    df["ema10"]  = ema(df["close"], 10)
    df["ema20"]  = ema(df["close"], 20)
    df["ema50"]  = ema(df["close"], 50)
    df["ema100"] = ema(df["close"], 100)
    df["ema200"] = ema(df["close"], 200)
    df["atr"]    = atr(df, ATR_PERIOD)
    df["vol_ma"] = df["volume"].rolling(VOL_MA_BARS, min_periods=VOL_MA_BARS).mean()

    df["imp_ret"] = df["close"] / df["close"].shift(LOOKBACK_BARS) - 1.0
    df["imp_vol_ok"] = df["volume"] > (VOL_MULT * df["vol_ma"])

    if USE_TREND_FILTER:
        df["trend_ok"] = (
            (df["close"] > df["ema10"]) &
            (df["close"] > df["ema20"]) &
            (df["close"] > df["ema50"]) &
            (df["close"] > df["ema100"]) &
            (df["close"] > df["ema200"])
        )
    else:
        df["trend_ok"] = True

    df["is_impulse"] = (df["imp_ret"] >= IMPULSE_MIN_RET) & df["imp_vol_ok"] & df["trend_ok"]

    in_pos = False
    watch = False
    cooldown_until = -1

    entry = None
    stop = None
    tp = None
    initial_stop = None
    be_moved = False
    hold = 0
    entry_time = None

    trades = []

    for i in range(len(df)):
        row = df.iloc[i]
        t = row["date"]

        if i <= cooldown_until:
            continue

        if in_pos:
            hold += 1
            lo, hi = float(row["low"]), float(row["high"])

            # move to BE on +1%
            if (not be_moved) and hi >= entry * (1.0 + MOVE_BE_PCT):
                stop = entry
                be_moved = True

            exit_reason = None
            exit_px = None

            # conservative: stop first if both happen same bar
            if lo <= stop:
                exit_reason = "STOP"
                exit_px = stop
            elif hi >= tp:
                exit_reason = "TP"
                exit_px = tp

            if exit_reason is not None:
                risk_per_share = entry - initial_stop
                r_mult = (exit_px - entry) / risk_per_share if risk_per_share and risk_per_share > 0 else 0.0
                trades.append({
                    "ticker": ticker,
                    "entry_time": entry_time,
                    "exit_time": t,
                    "entry": float(entry),
                    "exit": float(exit_px),
                    "stop_init": float(initial_stop),
                    "tp": float(tp),
                    "exit_reason": exit_reason,
                    "r_mult": float(r_mult),
                    "be_moved": bool(be_moved),
                    "hold_bars": int(hold),
                })

                # reset
                in_pos = False
                watch = False
                cooldown_until = i + COOLOFF_BARS
                entry = stop = tp = initial_stop = None
                be_moved = False
                hold = 0
                entry_time = None
                continue

            if hold >= MAX_HOLD_BARS:
                exit_px = float(row["close"])
                risk_per_share = entry - initial_stop
                r_mult = (exit_px - entry) / risk_per_share if risk_per_share and risk_per_share > 0 else 0.0
                trades.append({
                    "ticker": ticker,
                    "entry_time": entry_time,
                    "exit_time": t,
                    "entry": float(entry),
                    "exit": float(exit_px),
                    "stop_init": float(initial_stop),
                    "tp": float(tp),
                    "exit_reason": "TIME",
                    "r_mult": float(r_mult),
                    "be_moved": bool(be_moved),
                    "hold_bars": int(hold),
                })

                in_pos = False
                watch = False
                cooldown_until = i + COOLOFF_BARS
                entry = stop = tp = initial_stop = None
                be_moved = False
                hold = 0
                entry_time = None
                continue

        # not in position
        if not in_pos:
            if not watch:
                if bool(row["is_impulse"]):
                    watch = True
            else:
                # if trend breaks, stop watching
                if not bool(row["trend_ok"]):
                    watch = False
                    continue

                ema20 = row["ema20"]
                a = row["atr"]
                if pd.isna(ema20) or pd.isna(a):
                    continue

                # first touch criterion (hourly): low <= ema20
                if float(row["low"]) <= float(ema20):
                    entry = float(ema20)
                    entry_time = t
                    initial_stop = entry - STOP_ATR * float(a)
                    stop = initial_stop
                    tp = entry + TP_ATR * float(a)
                    be_moved = False
                    hold = 0
                    in_pos = True
                    watch = False

    return pd.DataFrame(trades)

def main():
    tickers = list_parquet_tickers("yahoo/1h/")
    print("Tickers in R2:", len(tickers))
    if not tickers:
        raise SystemExit("No parquet files found in yahoo/1h/")

    all_trades = []
    for i, t in enumerate(tickers, 1):
        key = f"yahoo/1h/{t}.parquet"
        try:
            df = read_parquet_from_r2(key)
            if df is None or df.empty:
                continue
            # normalize date type
            if "date" in df.columns:
                df["date"] = pd.to_datetime(df["date"], utc=True)
            tr = backtest_one(df, t)
            if not tr.empty:
                all_trades.append(tr)
            if i % 25 == 0:
                print(f"Processed {i}/{len(tickers)}")
        except Exception as e:
            print("Error reading/backtesting", t, ":", e)

    trades = pd.concat(all_trades, ignore_index=True) if all_trades else pd.DataFrame()
    print("Trades:", len(trades))

    # Portfolio approximation: compound trade results in exit-time order
    equity = 1.0
    curve = []
    if not trades.empty:
        trades["entry_time"] = pd.to_datetime(trades["entry_time"], utc=True)
        trades["exit_time"] = pd.to_datetime(trades["exit_time"], utc=True)
        trades = trades.sort_values("exit_time")

        for _, r in trades.iterrows():
            equity *= (1.0 + RISK_PER_TRADE * float(r["r_mult"]))
            curve.append((r["exit_time"], equity))

    equity_df = pd.DataFrame(curve, columns=["time", "equity"])
    stats = {
        "tickers_tested": int(len(tickers)),
        "trades": int(len(trades)) if not trades.empty else 0,
        "win_rate": float((trades["r_mult"] > 0).mean()) if not trades.empty else 0.0,
        "avg_R": float(trades["r_mult"].mean()) if not trades.empty else 0.0,
        "median_R": float(trades["r_mult"].median()) if not trades.empty else 0.0,
        "final_equity_trade_compound": float(equity),
        "risk_per_trade": float(RISK_PER_TRADE),
        "lookback_bars": int(LOOKBACK_BARS),
        "impulse_min_ret": float(IMPULSE_MIN_RET),
        "vol_mult": float(VOL_MULT),
        "atr_period": int(ATR_PERIOD),
        "stop_atr": float(STOP_ATR),
        "tp_atr": float(TP_ATR),
        "move_be_pct": float(MOVE_BE_PCT),
    }

    # Save artifacts to R2
    # trades.csv
    if not trades.empty:
        put_text(f"{RESULT_PREFIX}/trades.csv", trades.to_csv(index=False))
    else:
        put_text(f"{RESULT_PREFIX}/trades.csv", "ticker,entry_time,exit_time,entry,exit,stop_init,tp,exit_reason,r_mult,be_moved,hold_bars\n")

    # equity.csv
    if not equity_df.empty:
        put_text(f"{RESULT_PREFIX}/equity.csv", equity_df.to_csv(index=False))
    else:
        put_text(f"{RESULT_PREFIX}/equity.csv", "time,equity\n")

    # stats.json
    put_text(f"{RESULT_PREFIX}/stats.json", json.dumps(stats, indent=2))

    # equity.png
    plt.figure()
    if not equity_df.empty:
        plt.plot(equity_df["time"], equity_df["equity"])
        plt.xticks(rotation=30, ha="right")
        plt.title("POOS Hourly (trade-compounded) Equity")
        plt.tight_layout()
    else:
        plt.text(0.5, 0.5, "No trades", ha="center", va="center")
        plt.axis("off")

    buf = io.BytesIO()
    plt.savefig(buf, format="png", dpi=160)
    plt.close()
    put_bytes(f"{RESULT_PREFIX}/equity.png", buf.getvalue(), content_type="image/png")

    print("Saved to R2:", RESULT_PREFIX)
    print("Stats:", stats)

if __name__ == "__main__":
    main()
