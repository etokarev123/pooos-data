import os
import io
import json
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

# -------------- ENV (Intraday POOS-style) ------------
# Intraday impulse: smaller and faster than daily POOS
LOOKBACK_BARS = int(os.getenv("LOOKBACK_BARS", "20"))           # ~3 trading days
IMPULSE_MIN_RET = float(os.getenv("IMPULSE_MIN_RET", "0.08"))   # 8% impulse
VOL_MA_BARS = int(os.getenv("VOL_MA_BARS", "60"))               # ~8-9 trading days
VOL_MULT = float(os.getenv("VOL_MULT", "1.3"))                  # 1.3x volume

# Trend filter (lighter): close above EMA20 and EMA50, EMA20 > EMA50
USE_TREND_FILTER = os.getenv("USE_TREND_FILTER", "1") == "1"

# Entry: pullback touch EMA20 (default). You can switch to EMA10 by env ENTRY_EMA=10
ENTRY_EMA = int(os.getenv("ENTRY_EMA", "20"))

# Exits
ATR_PERIOD = int(os.getenv("ATR_PERIOD", "14"))
STOP_ATR = float(os.getenv("STOP_ATR", "1.0"))
TP_ATR = float(os.getenv("TP_ATR", "1.5"))
MOVE_BE_PCT = float(os.getenv("MOVE_BE_PCT", "0.01"))           # +1% => move stop to BE

# Trade management
MAX_HOLD_BARS = int(os.getenv("MAX_HOLD_BARS", "120"))           # ~2-3 trading weeks
COOLOFF_BARS = int(os.getenv("COOLOFF_BARS", "40"))              # ~1 trading week

# Equity compounding (still simplified)
RISK_PER_TRADE = float(os.getenv("RISK_PER_TRADE", "0.01"))

# Speed: optionally use only last N bars
TAIL_BARS = int(os.getenv("TAIL_BARS", "0"))

# Store results here (new folder so you don't overwrite the previous "No trades")
RESULT_PREFIX = os.getenv("RESULT_PREFIX", "results/poos_hourly_v2")

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
                tickers.append(key.split("/")[-1].replace(".parquet", ""))
        if resp.get("IsTruncated"):
            token = resp.get("NextContinuationToken")
        else:
            break
    return sorted(set(tickers))

def read_parquet_from_r2(key: str) -> pd.DataFrame:
    obj = s3.get_object(Bucket=R2_BUCKET, Key=key)
    return pd.read_parquet(io.BytesIO(obj["Body"].read()))

def put_bytes(key: str, data: bytes, content_type="application/octet-stream"):
    s3.put_object(Bucket=R2_BUCKET, Key=key, Body=data, ContentType=content_type)

def put_text(key: str, text: str, content_type="text/plain; charset=utf-8"):
    put_bytes(key, text.encode("utf-8"), content_type=content_type)

def ema(series: pd.Series, span: int) -> pd.Series:
    return series.ewm(span=span, adjust=False).mean()

def atr(df: pd.DataFrame, period: int) -> pd.Series:
    prev_close = df["close"].shift(1)
    tr = pd.concat(
        [
            (df["high"] - df["low"]).abs(),
            (df["high"] - prev_close).abs(),
            (df["low"] - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)
    return tr.rolling(period, min_periods=period).mean()

def backtest_one(df: pd.DataFrame, ticker: str) -> pd.DataFrame:
    df = df.sort_values("date").dropna(subset=["open", "high", "low", "close", "volume"]).copy()
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"], utc=True)

    if TAIL_BARS > 0 and len(df) > TAIL_BARS:
        df = df.iloc[-TAIL_BARS:].copy()

    # Indicators
    df["ema10"] = ema(df["close"], 10)
    df["ema20"] = ema(df["close"], 20)
    df["ema50"] = ema(df["close"], 50)
    df["atr"] = atr(df, ATR_PERIOD)
    df["vol_ma"] = df["volume"].rolling(VOL_MA_BARS, min_periods=VOL_MA_BARS).mean()

    # Impulse
    df["imp_ret"] = df["close"] / df["close"].shift(LOOKBACK_BARS) - 1.0
    df["imp_vol_ok"] = df["volume"] > (VOL_MULT * df["vol_ma"])

    if USE_TREND_FILTER:
        df["trend_ok"] = (df["close"] > df["ema20"]) & (df["close"] > df["ema50"]) & (df["ema20"] > df["ema50"])
    else:
        df["trend_ok"] = True

    df["is_impulse"] = (df["imp_ret"] >= IMPULSE_MIN_RET) & df["imp_vol_ok"] & df["trend_ok"]

    # Entry EMA choice
    ema_col = "ema20" if ENTRY_EMA == 20 else "ema10"

    in_pos = False
    watch = False
    cooldown_until = -1

    entry = stop = tp = initial_stop = None
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

            # Move to break-even
            if (not be_moved) and hi >= entry * (1.0 + MOVE_BE_PCT):
                stop = entry
                be_moved = True

            exit_reason = None
            exit_px = None

            # conservative: stop first if both touched
            if lo <= stop:
                exit_reason = "STOP"
                exit_px = stop
            elif hi >= tp:
                exit_reason = "TP"
                exit_px = tp

            if exit_reason:
                risk_per_share = entry - initial_stop
                r_mult = (exit_px - entry) / risk_per_share if (risk_per_share and risk_per_share > 0) else 0.0
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
                    "entry_ema": ema_col,
                })

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
                r_mult = (exit_px - entry) / risk_per_share if (risk_per_share and risk_per_share > 0) else 0.0
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
                    "entry_ema": ema_col,
                })

                in_pos = False
                watch = False
                cooldown_until = i + COOLOFF_BARS
                entry = stop = tp = initial_stop = None
                be_moved = False
                hold = 0
                entry_time = None
                continue

        # Not in position
        if not in_pos:
            if not watch:
                if bool(row["is_impulse"]):
                    watch = True
            else:
                # If trend breaks badly, drop watch
                if USE_TREND_FILTER and (not bool(row["trend_ok"])):
                    watch = False
                    continue

                a = row["atr"]
                e = row[ema_col]
                if pd.isna(a) or pd.isna(e):
                    continue

                # Pullback "touch" entry: low <= entry_ema
                if float(row["low"]) <= float(e):
                    entry = float(e)
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
            tr = backtest_one(df, t)
            if not tr.empty:
                all_trades.append(tr)
            if i % 25 == 0:
                print(f"Processed {i}/{len(tickers)}")
        except Exception as e:
            print("Error reading/backtesting", t, ":", e)

    trades = pd.concat(all_trades, ignore_index=True) if all_trades else pd.DataFrame()
    print("Trades:", len(trades))

    # Equity (simple trade-by-trade compound)
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
        "mode": "POOS-style intraday (1h)",
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
        "vol_ma_bars": int(VOL_MA_BARS),
        "atr_period": int(ATR_PERIOD),
        "stop_atr": float(STOP_ATR),
        "tp_atr": float(TP_ATR),
        "move_be_pct": float(MOVE_BE_PCT),
        "entry_ema": int(ENTRY_EMA),
        "trend_filter": bool(USE_TREND_FILTER),
    }

    # Save outputs to R2
    if not trades.empty:
        put_text(f"{RESULT_PREFIX}/trades.csv", trades.to_csv(index=False))
    else:
        put_text(f"{RESULT_PREFIX}/trades.csv", "ticker,entry_time,exit_time,entry,exit,stop_init,tp,exit_reason,r_mult,be_moved,hold_bars,entry_ema\n")

    if not equity_df.empty:
        put_text(f"{RESULT_PREFIX}/equity.csv", equity_df.to_csv(index=False))
    else:
        put_text(f"{RESULT_PREFIX}/equity.csv", "time,equity\n")

    put_text(f"{RESULT_PREFIX}/stats.json", json.dumps(stats, indent=2))

    plt.figure()
    if not equity_df.empty:
        plt.plot(equity_df["time"], equity_df["equity"])
        plt.xticks(rotation=30, ha="right")
        plt.title("POOS-style Intraday (1h) Equity (trade-compounded)")
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
