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

# ---------------- R2 ----------------
R2_ENDPOINT = os.environ["R2_ENDPOINT"]
R2_KEY = os.environ["R2_ACCESS_KEY_ID"]
R2_SECRET = os.environ["R2_SECRET_ACCESS_KEY"]
R2_BUCKET = os.environ["R2_BUCKET"]

DATA_PREFIX = os.getenv("DATA_PREFIX", "yahoo/1d/")

# ---------------- POOS (DAILY) ----------------
# These are defaults; we can tune to match your PDF after we see first run.
IMP_LOOKBACK_DAYS = int(os.getenv("IMP_LOOKBACK_DAYS", "20"))        # impulse lookback
IMP_MIN_RET = float(os.getenv("IMP_MIN_RET", "0.30"))                # +30% in 20 days
VOL_MA_DAYS = int(os.getenv("VOL_MA_DAYS", "20"))                    # volume baseline
VOL_MULT = float(os.getenv("VOL_MULT", "2.0"))                       # 2x volume

USE_TREND_FILTER = os.getenv("USE_TREND_FILTER", "1") == "1"
ENTRY_EMA = int(os.getenv("ENTRY_EMA", "20"))                        # pullback to EMA20 by default

ATR_PERIOD = int(os.getenv("ATR_PERIOD", "14"))
STOP_ATR = float(os.getenv("STOP_ATR", "1.0"))
TP_ATR = float(os.getenv("TP_ATR", "1.5"))
MOVE_BE_PCT = float(os.getenv("MOVE_BE_PCT", "0.01"))                # +1% => BE

MAX_HOLD_DAYS = int(os.getenv("MAX_HOLD_DAYS", "15"))                # POOS typically short swing
COOLOFF_DAYS = int(os.getenv("COOLOFF_DAYS", "10"))

TAIL_DAYS = int(os.getenv("TAIL_DAYS", "0"))                         # 0 = all (for OOS, set 252 or 504)

# -------------- Portfolio --------------
MAX_POSITIONS = int(os.getenv("MAX_POSITIONS", "20"))                # daily POOS usually fewer
CAPITAL_UTIL  = float(os.getenv("CAPITAL_UTIL", "0.75"))
POS_FRACTION  = float(os.getenv("POS_FRACTION", "")) if os.getenv("POS_FRACTION") else (CAPITAL_UTIL / MAX_POSITIONS)

SLIPPAGE_BPS   = float(os.getenv("SLIPPAGE_BPS", "5"))
COMMISSION_BPS = float(os.getenv("COMMISSION_BPS", "1"))

RESULT_PREFIX = os.getenv("RESULT_PREFIX", "results/poos_daily_v1")

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

def ema(s: pd.Series, span: int) -> pd.Series:
    return s.ewm(span=span, adjust=False).mean()

def atr(df: pd.DataFrame, period: int) -> pd.Series:
    prev_close = df["close"].shift(1)
    tr = pd.concat([
        (df["high"] - df["low"]).abs(),
        (df["high"] - prev_close).abs(),
        (df["low"] - prev_close).abs(),
    ], axis=1).max(axis=1)
    return tr.rolling(period, min_periods=period).mean()

def apply_costs(entry_px: float, exit_px: float) -> tuple[float, float]:
    slip = SLIPPAGE_BPS / 10000.0
    comm = COMMISSION_BPS / 10000.0
    entry_adj = entry_px * (1.0 + slip + comm)
    exit_adj  = exit_px  * (1.0 - slip - comm)
    return entry_adj, exit_adj

def generate_trades(df: pd.DataFrame, ticker: str) -> pd.DataFrame:
    df = df.sort_values("date").dropna(subset=["open","high","low","close","volume"]).copy()
    df["date"] = pd.to_datetime(df["date"], utc=True)

    if TAIL_DAYS > 0 and len(df) > TAIL_DAYS:
        df = df.iloc[-TAIL_DAYS:].copy()

    # Indicators
    df["ema10"]  = ema(df["close"], 10)
    df["ema20"]  = ema(df["close"], 20)
    df["ema50"]  = ema(df["close"], 50)
    df["ema200"] = ema(df["close"], 200)
    df["atr"]    = atr(df, ATR_PERIOD)
    df["vol_ma"] = df["volume"].rolling(VOL_MA_DAYS, min_periods=VOL_MA_DAYS).mean()

    df["imp_ret"] = df["close"] / df["close"].shift(IMP_LOOKBACK_DAYS) - 1.0
    df["vol_ok"] = df["volume"] > (VOL_MULT * df["vol_ma"])

    if USE_TREND_FILTER:
        df["trend_ok"] = (df["close"] > df["ema20"]) & (df["close"] > df["ema50"]) & (df["close"] > df["ema200"])
    else:
        df["trend_ok"] = True

    df["is_impulse"] = (df["imp_ret"] >= IMP_MIN_RET) & df["vol_ok"] & df["trend_ok"]

    ema_col = "ema20" if ENTRY_EMA == 20 else "ema10"

    in_pos = False
    watch = False
    cooldown_until = -1

    entry = stop = tp = initial_stop = None
    be_moved = False
    hold = 0
    entry_time = None

    # score for tie-breaking when many signals
    watch_score = None

    trades = []

    for i in range(len(df)):
        row = df.iloc[i]
        t = row["date"]

        if i <= cooldown_until:
            continue

        if in_pos:
            hold += 1
            lo, hi = float(row["low"]), float(row["high"])

            if (not be_moved) and hi >= entry * (1.0 + MOVE_BE_PCT):
                stop = entry
                be_moved = True

            exit_reason = None
            exit_px = None

            # conservative: stop first if both hit same day
            if lo <= stop:
                exit_reason = "STOP"
                exit_px = stop
            elif hi >= tp:
                exit_reason = "TP"
                exit_px = tp

            if exit_reason is not None:
                trades.append({
                    "ticker": ticker,
                    "entry_time": entry_time,
                    "exit_time": t,
                    "entry": float(entry),
                    "exit": float(exit_px),
                    "stop_init": float(initial_stop),
                    "tp": float(tp),
                    "exit_reason": exit_reason,
                    "hold_days": int(hold),
                    "be_moved": bool(be_moved),
                    "score": float(watch_score) if watch_score is not None else np.nan,
                })
                in_pos = False
                watch = False
                cooldown_until = i + COOLOFF_DAYS
                entry = stop = tp = initial_stop = None
                be_moved = False
                hold = 0
                entry_time = None
                watch_score = None
                continue

            if hold >= MAX_HOLD_DAYS:
                exit_px = float(row["close"])
                trades.append({
                    "ticker": ticker,
                    "entry_time": entry_time,
                    "exit_time": t,
                    "entry": float(entry),
                    "exit": float(exit_px),
                    "stop_init": float(initial_stop),
                    "tp": float(tp),
                    "exit_reason": "TIME",
                    "hold_days": int(hold),
                    "be_moved": bool(be_moved),
                    "score": float(watch_score) if watch_score is not None else np.nan,
                })
                in_pos = False
                watch = False
                cooldown_until = i + COOLOFF_DAYS
                entry = stop = tp = initial_stop = None
                be_moved = False
                hold = 0
                entry_time = None
                watch_score = None
                continue

        if not in_pos:
            if not watch:
                if bool(row["is_impulse"]):
                    # score: impulse size * volume multiple
                    vol_ratio = float(row["volume"] / row["vol_ma"]) if row["vol_ma"] and not pd.isna(row["vol_ma"]) else 0.0
                    watch_score = float(row["imp_ret"]) * vol_ratio
                    watch = True
            else:
                # If trend breaks, stop watching
                if USE_TREND_FILTER and (not bool(row["trend_ok"])):
                    watch = False
                    watch_score = None
                    continue

                a = row["atr"]
                e = row[ema_col]
                if pd.isna(a) or pd.isna(e):
                    continue

                # pullback touch
                if float(row["low"]) <= float(e):
                    entry = float(e)
                    entry_time = t
                    initial_stop = entry - STOP_ATR * float(a)
                    stop = initial_stop
                    tp = entry + TP_ATR * float(a)
                    in_pos = True
                    watch = False
                    be_moved = False
                    hold = 0

    return pd.DataFrame(trades)

def simulate_portfolio(trades: pd.DataFrame):
    if trades.empty:
        return pd.DataFrame(columns=["time","equity"]), pd.DataFrame()

    trades = trades.copy()
    trades["entry_time"] = pd.to_datetime(trades["entry_time"], utc=True)
    trades["exit_time"]  = pd.to_datetime(trades["exit_time"], utc=True)
    trades["score"] = pd.to_numeric(trades["score"], errors="coerce").fillna(0.0)

    trades = trades.sort_values(["entry_time","score"], ascending=[True, False]).reset_index(drop=True)
    trades["trade_id"] = np.arange(len(trades))

    equity = 1.0
    curve = []
    open_pos = {}

    entry_times = trades["entry_time"].sort_values().unique()

    def close_exits_up_to(t):
        nonlocal equity, open_pos, curve
        to_close = [tid for tid, pos in open_pos.items() if pos["exit_time"] <= t]
        to_close = sorted(to_close, key=lambda tid: open_pos[tid]["exit_time"])
        for tid in to_close:
            pos = open_pos.pop(tid)
            equity *= (1.0 + pos["pos_frac"] * pos["ret"])
            curve.append((pos["exit_time"], equity, "CLOSE", pos["ticker"], tid))

    for et in entry_times:
        close_exits_up_to(et)

        free_slots = MAX_POSITIONS - len(open_pos)
        if free_slots <= 0:
            continue

        used_frac = sum(p["pos_frac"] for p in open_pos.values())
        free_frac = max(0.0, CAPITAL_UTIL - used_frac)
        if free_frac <= 0:
            continue

        batch = trades[trades["entry_time"] == et].copy()
        if batch.empty:
            continue

        max_by_frac = int(free_frac / POS_FRACTION) if POS_FRACTION > 0 else 0
        can_open = max(0, min(free_slots, max_by_frac))
        if can_open <= 0:
            continue

        batch = batch.sort_values("score", ascending=False).head(can_open)

        for _, r in batch.iterrows():
            entry_px = float(r["entry"])
            exit_px  = float(r["exit"])
            entry_adj, exit_adj = apply_costs(entry_px, exit_px)
            ret = (exit_adj / entry_adj) - 1.0

            open_pos[int(r["trade_id"])] = {
                "ticker": r["ticker"],
                "entry_time": r["entry_time"],
                "exit_time": r["exit_time"],
                "ret": ret,
                "pos_frac": POS_FRACTION,
            }
            curve.append((r["entry_time"], equity, "OPEN", r["ticker"], int(r["trade_id"])))

    close_exits_up_to(pd.Timestamp.max.tz_localize("UTC"))

    events = pd.DataFrame(curve, columns=["time","equity","event","ticker","trade_id"])
    equity_points = events[events["event"]=="CLOSE"][["time","equity"]].reset_index(drop=True)
    return equity_points, events

def main():
    tickers = list_tickers(DATA_PREFIX)
    print("Daily tickers in R2:", len(tickers))
    if not tickers:
        raise SystemExit("No daily parquet found. Run make_daily_from_hourly.py first.")

    all_trades = []
    for i, t in enumerate(tickers, 1):
        try:
            df = read_parquet(f"{DATA_PREFIX}{t}.parquet")
            if df is None or df.empty:
                continue
            tr = generate_trades(df, t)
            if not tr.empty:
                all_trades.append(tr)
            if i % 50 == 0:
                print(f"Processed {i}/{len(tickers)}")
        except Exception as e:
            print("Error:", t, e)

    trades = pd.concat(all_trades, ignore_index=True) if all_trades else pd.DataFrame()
    print("Trades:", len(trades))

    equity_df, events_df = simulate_portfolio(trades)

    stats = {
        "mode": "POOS DAILY (from 1h->RTH daily bars)",
        "tickers_tested": int(len(tickers)),
        "trades": int(len(trades)) if not trades.empty else 0,
        "final_equity": float(equity_df["equity"].iloc[-1]) if not equity_df.empty else 1.0,
        "max_positions": int(MAX_POSITIONS),
        "capital_util": float(CAPITAL_UTIL),
        "pos_fraction": float(POS_FRACTION),
        "slippage_bps_per_side": float(SLIPPAGE_BPS),
        "commission_bps_per_side": float(COMMISSION_BPS),
        "imp_lookback_days": int(IMP_LOOKBACK_DAYS),
        "imp_min_ret": float(IMP_MIN_RET),
        "vol_ma_days": int(VOL_MA_DAYS),
        "vol_mult": float(VOL_MULT),
        "entry_ema": int(ENTRY_EMA),
        "atr_period": int(ATR_PERIOD),
        "stop_atr": float(STOP_ATR),
        "tp_atr": float(TP_ATR),
        "move_be_pct": float(MOVE_BE_PCT),
        "max_hold_days": int(MAX_HOLD_DAYS),
        "cooloff_days": int(COOLOFF_DAYS),
        "tail_days": int(TAIL_DAYS),
    }

    put_text(f"{RESULT_PREFIX}/stats.json", json.dumps(stats, indent=2))
    put_text(f"{RESULT_PREFIX}/trades.csv", trades.to_csv(index=False) if not trades.empty else "ticker,entry_time,exit_time,entry,exit,stop_init,tp,exit_reason,hold_days,be_moved,score\n")
    put_text(f"{RESULT_PREFIX}/equity.csv", equity_df.to_csv(index=False) if not equity_df.empty else "time,equity\n")
    put_text(f"{RESULT_PREFIX}/events.csv", events_df.to_csv(index=False) if not events_df.empty else "time,equity,event,ticker,trade_id\n")

    # plot
    plt.figure()
    if not equity_df.empty:
        plt.plot(pd.to_datetime(equity_df["time"]), equity_df["equity"])
        plt.xticks(rotation=30, ha="right")
        plt.title("POOS Daily Portfolio Equity (RTH daily, costs)")
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
