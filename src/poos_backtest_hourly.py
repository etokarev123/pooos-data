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

# -------------- ENV (POOS-style intraday) ------------
LOOKBACK_BARS   = int(os.getenv("LOOKBACK_BARS", "20"))
IMPULSE_MIN_RET = float(os.getenv("IMPULSE_MIN_RET", "0.08"))
VOL_MA_BARS     = int(os.getenv("VOL_MA_BARS", "60"))
VOL_MULT        = float(os.getenv("VOL_MULT", "1.3"))

USE_TREND_FILTER = os.getenv("USE_TREND_FILTER", "1") == "1"
ENTRY_EMA        = int(os.getenv("ENTRY_EMA", "20"))  # 20 or 10

ATR_PERIOD  = int(os.getenv("ATR_PERIOD", "14"))
STOP_ATR    = float(os.getenv("STOP_ATR", "1.0"))
TP_ATR      = float(os.getenv("TP_ATR", "1.5"))
MOVE_BE_PCT = float(os.getenv("MOVE_BE_PCT", "0.01"))

MAX_HOLD_BARS = int(os.getenv("MAX_HOLD_BARS", "120"))
COOLOFF_BARS  = int(os.getenv("COOLOFF_BARS", "40"))

TAIL_BARS = int(os.getenv("TAIL_BARS", "0"))

# -------- Portfolio constraints ----------
MAX_POSITIONS = int(os.getenv("MAX_POSITIONS", "75"))
CAPITAL_UTIL  = float(os.getenv("CAPITAL_UTIL", "0.75"))   # use 75% of equity at most
POS_FRACTION  = float(os.getenv("POS_FRACTION", "")) if os.getenv("POS_FRACTION") else (CAPITAL_UTIL / MAX_POSITIONS)

# -------- Costs (bps) ----------
# applied on entry and exit:
SLIPPAGE_BPS   = float(os.getenv("SLIPPAGE_BPS", "5"))     # 5 bps = 0.05% per side
COMMISSION_BPS = float(os.getenv("COMMISSION_BPS", "1"))   # 1 bps = 0.01% per side

RESULT_PREFIX = os.getenv("RESULT_PREFIX", "results/poos_hourly_v3_portfolio")

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
    df["date"] = pd.to_datetime(df["date"], utc=True)

    if TAIL_BARS > 0 and len(df) > TAIL_BARS:
        df = df.iloc[-TAIL_BARS:].copy()

    df["ema10"] = ema(df["close"], 10)
    df["ema20"] = ema(df["close"], 20)
    df["ema50"] = ema(df["close"], 50)
    df["atr"] = atr(df, ATR_PERIOD)
    df["vol_ma"] = df["volume"].rolling(VOL_MA_BARS, min_periods=VOL_MA_BARS).mean()

    df["imp_ret"] = df["close"] / df["close"].shift(LOOKBACK_BARS) - 1.0
    df["vol_ratio"] = df["volume"] / df["vol_ma"]
    df["imp_vol_ok"] = df["volume"] > (VOL_MULT * df["vol_ma"])

    if USE_TREND_FILTER:
        df["trend_ok"] = (df["close"] > df["ema20"]) & (df["close"] > df["ema50"]) & (df["ema20"] > df["ema50"])
    else:
        df["trend_ok"] = True

    df["is_impulse"] = (df["imp_ret"] >= IMPULSE_MIN_RET) & df["imp_vol_ok"] & df["trend_ok"]

    ema_col = "ema20" if ENTRY_EMA == 20 else "ema10"

    in_pos = False
    watch = False
    cooldown_until = -1

    entry = stop = tp = initial_stop = None
    be_moved = False
    hold = 0
    entry_time = None

    # score captured at impulse moment; reused for the eventual entry
    watch_score = None
    watch_imp_ret = None
    watch_vol_ratio = None

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

            if lo <= stop:
                exit_reason = "STOP"
                exit_px = stop
            elif hi >= tp:
                exit_reason = "TP"
                exit_px = tp

            if exit_reason:
                trades.append({
                    "ticker": ticker,
                    "entry_time": entry_time,
                    "exit_time": t,
                    "entry": float(entry),
                    "exit": float(exit_px),
                    "stop_init": float(initial_stop),
                    "tp": float(tp),
                    "exit_reason": exit_reason,
                    "be_moved": bool(be_moved),
                    "hold_bars": int(hold),
                    "entry_ema": ema_col,
                    "score": float(watch_score) if watch_score is not None else np.nan,
                    "imp_ret": float(watch_imp_ret) if watch_imp_ret is not None else np.nan,
                    "vol_ratio": float(watch_vol_ratio) if watch_vol_ratio is not None else np.nan,
                })

                in_pos = False
                watch = False
                cooldown_until = i + COOLOFF_BARS
                entry = stop = tp = initial_stop = None
                be_moved = False
                hold = 0
                entry_time = None
                watch_score = watch_imp_ret = watch_vol_ratio = None
                continue

            if hold >= MAX_HOLD_BARS:
                exit_reason = "TIME"
                exit_px = float(row["close"])
                trades.append({
                    "ticker": ticker,
                    "entry_time": entry_time,
                    "exit_time": t,
                    "entry": float(entry),
                    "exit": float(exit_px),
                    "stop_init": float(initial_stop),
                    "tp": float(tp),
                    "exit_reason": exit_reason,
                    "be_moved": bool(be_moved),
                    "hold_bars": int(hold),
                    "entry_ema": ema_col,
                    "score": float(watch_score) if watch_score is not None else np.nan,
                    "imp_ret": float(watch_imp_ret) if watch_imp_ret is not None else np.nan,
                    "vol_ratio": float(watch_vol_ratio) if watch_vol_ratio is not None else np.nan,
                })

                in_pos = False
                watch = False
                cooldown_until = i + COOLOFF_BARS
                entry = stop = tp = initial_stop = None
                be_moved = False
                hold = 0
                entry_time = None
                watch_score = watch_imp_ret = watch_vol_ratio = None
                continue

        if not in_pos:
            if not watch:
                if bool(row["is_impulse"]):
                    # score: impulse strength * volume ratio
                    imp_ret = float(row["imp_ret"])
                    vol_ratio = float(row["vol_ratio"]) if not pd.isna(row["vol_ratio"]) else 0.0
                    watch_score = imp_ret * vol_ratio
                    watch_imp_ret = imp_ret
                    watch_vol_ratio = vol_ratio
                    watch = True
            else:
                if USE_TREND_FILTER and (not bool(row["trend_ok"])):
                    watch = False
                    watch_score = watch_imp_ret = watch_vol_ratio = None
                    continue

                a = row["atr"]
                e = row[ema_col]
                if pd.isna(a) or pd.isna(e):
                    continue

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

def apply_costs(entry_px: float, exit_px: float) -> tuple[float, float]:
    # For long positions: entry worsens (higher), exit worsens (lower)
    slip = SLIPPAGE_BPS / 10000.0
    comm = COMMISSION_BPS / 10000.0
    entry_adj = entry_px * (1.0 + slip + comm)
    exit_adj  = exit_px  * (1.0 - slip - comm)
    return entry_adj, exit_adj

def simulate_portfolio(trades: pd.DataFrame):
    if trades.empty:
        return pd.DataFrame(columns=["time", "equity"]), pd.DataFrame()

    trades = trades.copy()
    trades["entry_time"] = pd.to_datetime(trades["entry_time"], utc=True)
    trades["exit_time"]  = pd.to_datetime(trades["exit_time"], utc=True)

    # For ranking when too many entries at same time
    trades["score"] = pd.to_numeric(trades["score"], errors="coerce").fillna(0.0)

    # Build entry groups
    trades = trades.sort_values(["entry_time", "score"], ascending=[True, False]).reset_index(drop=True)
    trades["trade_id"] = np.arange(len(trades))

    equity = 1.0
    curve = []
    open_positions = {}  # trade_id -> dict(entry_time, exit_time, ret, pos_frac)

    # We'll process events in time order: entry batches, then exits that occur before next entries
    entry_times = trades["entry_time"].sort_values().unique()

    def close_exits_up_to(t):
        nonlocal equity, open_positions, curve
        to_close = [tid for tid, pos in open_positions.items() if pos["exit_time"] <= t]
        # close in chronological order
        to_close = sorted(to_close, key=lambda tid: open_positions[tid]["exit_time"])
        for tid in to_close:
            pos = open_positions.pop(tid)
            # apply return to equity on close
            equity *= (1.0 + pos["pos_frac"] * pos["ret"])
            curve.append((pos["exit_time"], equity, "CLOSE", pos["ticker"], tid))

    for et in entry_times:
        close_exits_up_to(et)

        # free slots
        free_slots = MAX_POSITIONS - len(open_positions)
        if free_slots <= 0:
            continue

        # capital utilization constraint (approx): we limit sum(pos_frac) <= CAPITAL_UTIL
        used_frac = sum(pos["pos_frac"] for pos in open_positions.values())
        free_frac = max(0.0, CAPITAL_UTIL - used_frac)
        if free_frac <= 0:
            continue

        batch = trades[trades["entry_time"] == et].copy()
        if batch.empty:
            continue

        # Determine how many we can open: by slots and by free_frac
        # Each new position uses POS_FRACTION of equity (approx)
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

            open_positions[int(r["trade_id"])] = {
                "ticker": r["ticker"],
                "entry_time": r["entry_time"],
                "exit_time": r["exit_time"],
                "ret": ret,
                "pos_frac": POS_FRACTION,
                "score": float(r["score"]),
            }
            curve.append((r["entry_time"], equity, "OPEN", r["ticker"], int(r["trade_id"])))

    # close remaining
    close_exits_up_to(pd.Timestamp.max.tz_localize("UTC"))

    curve_df = pd.DataFrame(curve, columns=["time", "equity", "event", "ticker", "trade_id"])
    equity_points = curve_df[curve_df["event"] == "CLOSE"][["time", "equity"]].reset_index(drop=True)
    return equity_points, curve_df

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
    print("Trades (raw signals):", len(trades))

    equity_df, events_df = simulate_portfolio(trades)

    # Stats
    stats = {
        "mode": "POOS-style intraday (1h) PORTFOLIO",
        "tickers_tested": int(len(tickers)),
        "raw_trades": int(len(trades)) if not trades.empty else 0,
        "max_positions": int(MAX_POSITIONS),
        "capital_util": float(CAPITAL_UTIL),
        "pos_fraction": float(POS_FRACTION),
        "slippage_bps_per_side": float(SLIPPAGE_BPS),
        "commission_bps_per_side": float(COMMISSION_BPS),
        "final_equity": float(equity_df["equity"].iloc[-1]) if not equity_df.empty else 1.0,
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

    # Save outputs
    put_text(f"{RESULT_PREFIX}/stats.json", json.dumps(stats, indent=2))

    if not trades.empty:
        put_text(f"{RESULT_PREFIX}/raw_trades.csv", trades.to_csv(index=False))
    else:
        put_text(f"{RESULT_PREFIX}/raw_trades.csv", "ticker,entry_time,exit_time,entry,exit,stop_init,tp,exit_reason,be_moved,hold_bars,entry_ema,score,imp_ret,vol_ratio\n")

    if not equity_df.empty:
        put_text(f"{RESULT_PREFIX}/equity.csv", equity_df.to_csv(index=False))
    else:
        put_text(f"{RESULT_PREFIX}/equity.csv", "time,equity\n")

    if not events_df.empty:
        put_text(f"{RESULT_PREFIX}/events.csv", events_df.to_csv(index=False))
    else:
        put_text(f"{RESULT_PREFIX}/events.csv", "time,equity,event,ticker,trade_id\n")

    # Plot equity
    plt.figure()
    if not equity_df.empty:
        plt.plot(pd.to_datetime(equity_df["time"]), equity_df["equity"])
        plt.xticks(rotation=30, ha="right")
        plt.title("POOS-style Intraday Portfolio Equity (max 75 pos, costs)")
        plt.tight_layout()
    else:
        plt.text(0.5, 0.5, "No closes / No trades", ha="center", va="center")
        plt.axis("off")

    buf = io.BytesIO()
    plt.savefig(buf, format="png", dpi=160)
    plt.close()
    put_bytes(f"{RESULT_PREFIX}/equity.png", buf.getvalue(), content_type="image/png")

    print("Saved to R2:", RESULT_PREFIX)
    print("Stats:", stats)

if __name__ == "__main__":
    main()
