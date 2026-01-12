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

DATA_PREFIX = os.getenv("DATA_PREFIX", "yahoo/1d/")
MARKET_STATE_KEY = os.getenv("MARKET_STATE_KEY", "results/poos_market_engine_v1/market_state.csv")
RESULT_PREFIX = os.getenv("RESULT_PREFIX", "results/poos_daily_pooslike_v4_trailing")

# ---------------- Costs ----------------
SLIPPAGE_BPS   = float(os.getenv("SLIPPAGE_BPS", "5"))
COMMISSION_BPS = float(os.getenv("COMMISSION_BPS", "1"))

# ---------------- Portfolio ----------------
MAX_POSITIONS = int(os.getenv("MAX_POSITIONS", "75"))
CAPITAL_UTIL  = float(os.getenv("CAPITAL_UTIL", "1.50"))
POS_FRACTION  = float(os.getenv("POS_FRACTION", "0.03")) if os.getenv("POS_FRACTION") else (CAPITAL_UTIL / MAX_POSITIONS)

# ---------------- Market gating ----------------
ALLOW_ENTRY_STATES = [s.strip() for s in os.getenv("ALLOW_ENTRY_STATES", "risk_on,neutral").split(",") if s.strip()]
FORCE_EXIT_STATES  = [s.strip() for s in os.getenv("FORCE_EXIT_STATES", "risk_off").split(",") if s.strip()]

# ---------------- POOS-like setup ----------------
# Impulse: 1-2 day burst + volume
IMPULSE_RET1_MIN = float(os.getenv("IMPULSE_RET1_MIN", "0.04"))
IMPULSE_RET2_MIN = float(os.getenv("IMPULSE_RET2_MIN", "0.07"))
VOL_MA_DAYS      = int(os.getenv("VOL_MA_DAYS", "20"))
VOL_MULT_IMP     = float(os.getenv("VOL_MULT_IMP", "1.2"))

# Memory after impulse
IMPULSE_MEMORY_DAYS = int(os.getenv("IMPULSE_MEMORY_DAYS", "21"))

# Consolidation / tightness
CONS_LOOKBACK = int(os.getenv("CONS_LOOKBACK", "5"))
CONS_MAX_RANGE_PCT = float(os.getenv("CONS_MAX_RANGE_PCT", "0.12"))

# Trend filter
USE_TREND_FILTER = os.getenv("USE_TREND_FILTER", "1") == "1"

# Entry & risk
ENTRY_EMA = int(os.getenv("ENTRY_EMA", "20"))   # currently EMA20
ATR_PERIOD = int(os.getenv("ATR_PERIOD", "14"))

# We keep initial risk stop wide-ish, winners will be trailed
STOP_ATR = float(os.getenv("STOP_ATR", "1.2"))

# Hard TP is optional (we will mostly rely on trailing). If you want "no TP", set VERY HIGH.
TP_ATR = float(os.getenv("TP_ATR", "8.0"))

MOVE_BE_PCT = float(os.getenv("MOVE_BE_PCT", "0.01"))

# Trailing config
TRAIL_MODE = os.getenv("TRAIL_MODE", "ema_atr")  # "ema_atr" or "close_atr"
TRAIL_ATR_MULT = float(os.getenv("TRAIL_ATR_MULT", "1.8"))  # how tight the trail is after BE

# Partial take-profit (optional)
ENABLE_PARTIAL_TP = os.getenv("ENABLE_PARTIAL_TP", "1") == "1"
PARTIAL_TP_R = float(os.getenv("PARTIAL_TP_R", "2.0"))        # take some at 2R
PARTIAL_FRACTION = float(os.getenv("PARTIAL_FRACTION", "0.50"))  # take 50% off

MAX_HOLD_DAYS = int(os.getenv("MAX_HOLD_DAYS", "30"))
COOLOFF_DAYS  = int(os.getenv("COOLOFF_DAYS", "10"))

TAIL_DAYS = int(os.getenv("TAIL_DAYS", "504"))

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

def read_text(key: str) -> str:
    obj = s3.get_object(Bucket=R2_BUCKET, Key=key)
    return obj["Body"].read().decode("utf-8")

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

def load_market_state() -> pd.DataFrame:
    ms = pd.read_csv(io.StringIO(read_text(MARKET_STATE_KEY)))
    ms["date"] = pd.to_datetime(ms["date"], utc=True)
    ms["d"] = ms["date"].dt.normalize()
    return ms[["date","d","market_state","market_score"]].sort_values("date")

def compute_consolidation(df: pd.DataFrame) -> pd.Series:
    roll_hi = df["high"].rolling(CONS_LOOKBACK, min_periods=CONS_LOOKBACK).max()
    roll_lo = df["low"].rolling(CONS_LOOKBACK, min_periods=CONS_LOOKBACK).min()
    rng = (roll_hi - roll_lo) / df["close"]
    return rng

def generate_trades(df: pd.DataFrame, ticker: str, ms_map: dict):
    df = df.sort_values("date").dropna(subset=["open","high","low","close","volume"]).copy()
    df["date"] = pd.to_datetime(df["date"], utc=True)

    if TAIL_DAYS > 0 and len(df) > TAIL_DAYS:
        df = df.iloc[-TAIL_DAYS:].copy()

    df["ema20"]  = ema(df["close"], 20)
    df["ema50"]  = ema(df["close"], 50)
    df["ema200"] = ema(df["close"], 200)
    df["atr"]    = atr(df, ATR_PERIOD)
    df["vol_ma"] = df["volume"].rolling(VOL_MA_DAYS, min_periods=VOL_MA_DAYS).mean()

    df["ret1"] = df["close"].pct_change(1)
    df["ret2"] = df["close"].pct_change(2)

    df["is_impulse"] = (
        ((df["ret1"] >= IMPULSE_RET1_MIN) | (df["ret2"] >= IMPULSE_RET2_MIN))
        & (df["volume"] >= (VOL_MULT_IMP * df["vol_ma"]))
    ).astype(int)

    df["impulse_recent"] = (
        df["is_impulse"].rolling(IMPULSE_MEMORY_DAYS, min_periods=1).max().fillna(0).astype(int)
    )

    df["cons_range_pct"] = compute_consolidation(df)
    df["is_tight"] = (df["cons_range_pct"] <= CONS_MAX_RANGE_PCT).astype(int)

    if USE_TREND_FILTER:
        df["trend_ok"] = (df["close"] > df["ema20"]) & (df["close"] > df["ema50"]) & (df["close"] > df["ema200"])
    else:
        df["trend_ok"] = True

    ema_col = "ema20"  # currently only EMA20 entry

    watch = False
    cooldown_until = -1

    # Position state for per-ticker simulation
    in_pos = False
    entry = None
    entry_time = None
    entry_mstate = None

    initial_stop = None
    stop = None
    tp = None

    be_moved = False
    hold = 0
    watch_score = None

    # partial TP state
    partial_done = False
    partial_px = None
    r_unit = None  # entry - initial_stop

    trades = []

    diag = {
        "ticker": ticker,
        "rows": int(len(df)),
        "impulses": int(df["is_impulse"].sum()),
        "impulse_recent_days": int(df["impulse_recent"].sum()),
        "tight_days": int(df["is_tight"].sum()),
    }

    for i in range(len(df)):
        row = df.iloc[i]
        t = row["date"]
        d = t.normalize()

        if i <= cooldown_until:
            continue

        mstate = ms_map.get(d, "neutral")

        # ---------------- manage open position ----------------
        if in_pos:
            hold += 1
            lo, hi, cl = float(row["low"]), float(row["high"]), float(row["close"])

            # move to BE once price moves +MOVE_BE_PCT in our favor
            if (not be_moved) and hi >= entry * (1.0 + MOVE_BE_PCT):
                stop = max(stop, entry)
                be_moved = True

            # after BE, trail stop (POOS-like: let winners run)
            if be_moved:
                a = row["atr"]
                e = row["ema20"]
                if not pd.isna(a) and not pd.isna(e):
                    if TRAIL_MODE == "ema_atr":
                        trail = float(e) - TRAIL_ATR_MULT * float(a)
                    else:
                        trail = float(cl) - TRAIL_ATR_MULT * float(a)
                    stop = max(stop, trail)

            # Partial TP (take some off at PARTIAL_TP_R * R)
            if ENABLE_PARTIAL_TP and (not partial_done) and (r_unit is not None) and (r_unit > 0):
                target_partial = entry + PARTIAL_TP_R * r_unit
                if hi >= target_partial:
                    partial_done = True
                    partial_px = target_partial
                    # record partial trade as separate trade with size_mult = PARTIAL_FRACTION
                    trades.append({
                        "ticker": ticker,
                        "entry_time": entry_time,
                        "exit_time": t,
                        "entry": float(entry),
                        "exit": float(partial_px),
                        "exit_reason": f"PARTIAL_{PARTIAL_TP_R}R",
                        "hold_days": int(hold),
                        "be_moved": bool(be_moved),
                        "score": float(watch_score) if watch_score is not None else np.nan,
                        "entry_market_state": entry_mstate,
                        "size_mult": float(PARTIAL_FRACTION),
                    })

            # Hard TP (usually very far if you want trailing)
            if tp is not None and hi >= tp:
                exit_px = tp
                # remaining size
                rem = 1.0 - (PARTIAL_FRACTION if partial_done else 0.0)
                if rem > 0:
                    trades.append({
                        "ticker": ticker,
                        "entry_time": entry_time,
                        "exit_time": t,
                        "entry": float(entry),
                        "exit": float(exit_px),
                        "exit_reason": "TP",
                        "hold_days": int(hold),
                        "be_moved": bool(be_moved),
                        "score": float(watch_score) if watch_score is not None else np.nan,
                        "entry_market_state": entry_mstate,
                        "size_mult": float(rem),
                    })
                in_pos = False
                cooldown_until = i + COOLOFF_DAYS
                continue

            # Stop hit (includes trailed stop)
            if lo <= stop:
                exit_px = stop
                rem = 1.0 - (PARTIAL_FRACTION if partial_done else 0.0)
                if rem > 0:
                    trades.append({
                        "ticker": ticker,
                        "entry_time": entry_time,
                        "exit_time": t,
                        "entry": float(entry),
                        "exit": float(exit_px),
                        "exit_reason": "STOP",
                        "hold_days": int(hold),
                        "be_moved": bool(be_moved),
                        "score": float(watch_score) if watch_score is not None else np.nan,
                        "entry_market_state": entry_mstate,
                        "size_mult": float(rem),
                    })
                in_pos = False
                cooldown_until = i + COOLOFF_DAYS
                continue

            # Time exit
            if hold >= MAX_HOLD_DAYS:
                exit_px = float(row["close"])
                rem = 1.0 - (PARTIAL_FRACTION if partial_done else 0.0)
                if rem > 0:
                    trades.append({
                        "ticker": ticker,
                        "entry_time": entry_time,
                        "exit_time": t,
                        "entry": float(entry),
                        "exit": float(exit_px),
                        "exit_reason": "TIME",
                        "hold_days": int(hold),
                        "be_moved": bool(be_moved),
                        "score": float(watch_score) if watch_score is not None else np.nan,
                        "entry_market_state": entry_mstate,
                        "size_mult": float(rem),
                    })
                in_pos = False
                cooldown_until = i + COOLOFF_DAYS
                continue

        # ---------------- find new entry ----------------
        if not in_pos:
            if not watch:
                if (mstate in ALLOW_ENTRY_STATES) and bool(row["impulse_recent"]) and bool(row["trend_ok"]):
                    vol_ratio = float(row["volume"] / row["vol_ma"]) if row["vol_ma"] and not pd.isna(row["vol_ma"]) else 0.0
                    burst = max(float(row["ret1"]) if not pd.isna(row["ret1"]) else 0.0,
                                float(row["ret2"]) if not pd.isna(row["ret2"]) else 0.0)
                    watch_score = float(burst) * float(vol_ratio)
                    watch = True
            else:
                if mstate not in ALLOW_ENTRY_STATES:
                    watch = False
                    watch_score = None
                    continue

                if USE_TREND_FILTER and (not bool(row["trend_ok"])):
                    watch = False
                    watch_score = None
                    continue

                if not bool(row["is_tight"]):
                    continue

                a = row["atr"]
                e = row[ema_col]
                if pd.isna(a) or pd.isna(e):
                    continue

                if float(row["low"]) <= float(e):
                    entry = float(e)
                    entry_time = t
                    entry_mstate = mstate

                    initial_stop = entry - STOP_ATR * float(a)
                    stop = initial_stop
                    tp = entry + TP_ATR * float(a) if TP_ATR and TP_ATR > 0 else None

                    be_moved = False
                    hold = 0
                    partial_done = False
                    partial_px = None
                    r_unit = max(0.0, entry - initial_stop)

                    in_pos = True
                    watch = False

    return pd.DataFrame(trades), diag

# ---------- Portfolio simulator with DAILY mark-to-market ----------

def simulate_portfolio_daily(trades: pd.DataFrame, market_state: pd.DataFrame):
    """
    Self-financing-ish simulation with cash + positions.
    Each opened trade uses notional = POS_FRACTION * size_mult * equity_at_open.
    Equity is marked daily using daily closes.
    Forced exit on risk_off at that day's close.
    """

    if trades.empty:
        return pd.DataFrame(columns=["date","equity"]), pd.DataFrame()

    trades = trades.copy()
    trades["entry_time"] = pd.to_datetime(trades["entry_time"], utc=True)
    trades["exit_time"]  = pd.to_datetime(trades["exit_time"], utc=True)
    if "size_mult" not in trades.columns:
        trades["size_mult"] = 1.0
    trades["size_mult"] = pd.to_numeric(trades["size_mult"], errors="coerce").fillna(1.0)

    # Market state map by day
    ms = market_state.copy()
    ms["d"] = pd.to_datetime(ms["d"], utc=True)
    ms_map = dict(zip(ms["d"], ms["market_state"]))

    # Universe tickers
    tickers = sorted(trades["ticker"].unique().tolist())

    # Determine calendar range
    start = min(trades["entry_time"].min().normalize(), trades["exit_time"].min().normalize())
    end = max(trades["exit_time"].max().normalize(), trades["entry_time"].max().normalize())

    # Load daily closes for tickers used (only in window)
    closes = []
    for tkr in tickers:
        try:
            df = read_parquet(f"{DATA_PREFIX}{tkr}.parquet")
            df["date"] = pd.to_datetime(df["date"], utc=True)
            df["d"] = df["date"].dt.normalize()
            df = df[(df["d"] >= start) & (df["d"] <= end)][["d","close"]].dropna()
            df["ticker"] = tkr
            closes.append(df)
        except Exception as e:
            print("close load err", tkr, e)

    close_df = pd.concat(closes, ignore_index=True) if closes else pd.DataFrame(columns=["d","close","ticker"])
    close_df = close_df.rename(columns={"d":"date"})
    close_pivot = close_df.pivot(index="date", columns="ticker", values="close").sort_index()

    # Build calendar from available close dates (intersection/union)
    cal = close_pivot.index
    if len(cal) == 0:
        return pd.DataFrame(columns=["date","equity"]), pd.DataFrame()

    # Prepare trade lists by day
    trades["d_entry"] = trades["entry_time"].dt.normalize()
    trades["d_exit"]  = trades["exit_time"].dt.normalize()

    # Sort for capacity preference (score desc)
    trades["score"] = pd.to_numeric(trades.get("score", 0.0), errors="coerce").fillna(0.0)
    trades = trades.sort_values(["entry_time","score"], ascending=[True, False]).reset_index(drop=True)
    trades["trade_id"] = np.arange(len(trades))

    entries_by_day = {d: df for d, df in trades.groupby("d_entry")}
    exits_by_day = {d: df for d, df in trades.groupby("d_exit")}

    # Position book
    # pos: trade_id -> dict
    positions = {}

    cash = 1.0
    equity_curve = []
    events = []

    def compute_equity(date):
        eq = cash
        if positions:
            prices = close_pivot.loc[date]
            for tid, p in positions.items():
                px = prices.get(p["ticker"], np.nan)
                if pd.isna(px):
                    continue
                # mark-to-market using raw close; costs are applied on entry/exit only
                eq += p["notional"] * (float(px) / p["entry_px"])
        return float(eq)

    def used_util(current_equity):
        # approx utilization = sum(notional)/equity
        notional_sum = sum(p["notional"] for p in positions.values())
        return (notional_sum / current_equity) if current_equity > 0 else 0.0

    for d in cal:
        # 1) mark-to-market equity at start of day close (we will use close-to-close)
        eq = compute_equity(d)

        # 2) forced exit on risk_off (at this day's close)
        if ms_map.get(d, "neutral") in FORCE_EXIT_STATES and positions:
            prices = close_pivot.loc[d]
            to_close = list(positions.keys())
            for tid in to_close:
                p = positions.pop(tid)
                px = prices.get(p["ticker"], np.nan)
                if pd.isna(px):
                    px = p["entry_px"]
                # apply costs on exit
                entry_adj, exit_adj = apply_costs(p["entry_px"], float(px))
                ret = (exit_adj / entry_adj) - 1.0
                cash += p["notional"] * (1.0 + ret)
                events.append((pd.Timestamp(d).tz_localize("UTC"), "FORCE_EXIT", p["ticker"], int(tid), "MARKET_RISK_OFF"))
            eq = compute_equity(d)

        # 3) process scheduled exits (if not already force-exited)
        if d in exits_by_day and positions:
            prices = close_pivot.loc[d]
            for _, r in exits_by_day[d].iterrows():
                tid = int(r["trade_id"])
                if tid not in positions:
                    continue
                p = positions.pop(tid)
                px = prices.get(p["ticker"], np.nan)
                if pd.isna(px):
                    px = float(r["exit"])
                entry_adj, exit_adj = apply_costs(p["entry_px"], float(px))
                ret = (exit_adj / entry_adj) - 1.0
                cash += p["notional"] * (1.0 + ret)
                events.append((pd.Timestamp(d).tz_localize("UTC"), "CLOSE", p["ticker"], tid, str(r.get("exit_reason",""))))
            eq = compute_equity(d)

        # 4) open new trades for this day (if market allows entries that day)
        if d in entries_by_day:
            if ms_map.get(d, "neutral") in ALLOW_ENTRY_STATES:
                batch = entries_by_day[d].copy()

                # capacity: max_positions + util cap
                current_equity = compute_equity(d)
                free_slots = MAX_POSITIONS - len(positions)
                if free_slots > 0 and current_equity > 0:
                    current_util = used_util(current_equity)
                    free_util = max(0.0, CAPITAL_UTIL - current_util)
                    # open trades by score until capacity
                    batch = batch.sort_values("score", ascending=False)

                    prices = close_pivot.loc[d]

                    for _, r in batch.iterrows():
                        if free_slots <= 0:
                            break
                        current_equity = compute_equity(d)
                        current_util = used_util(current_equity)
                        free_util = max(0.0, CAPITAL_UTIL - current_util)
                        if free_util <= 0:
                            break

                        size_mult = float(r.get("size_mult", 1.0))
                        pos_frac = POS_FRACTION * size_mult
                        # util consumed approx pos_frac (since notional = pos_frac * equity)
                        if pos_frac <= 0:
                            continue
                        if pos_frac > free_util:
                            continue

                        # need price
                        px = prices.get(r["ticker"], np.nan)
                        if pd.isna(px):
                            continue

                        # apply costs on entry
                        entry_adj, _ = apply_costs(float(px), float(px))
                        notional = pos_frac * current_equity
                        cash -= notional  # invest notional (can go negative -> leverage)

                        tid = int(r["trade_id"])
                        positions[tid] = {
                            "ticker": r["ticker"],
                            "entry_px": float(entry_adj),
                            "notional": float(notional),
                        }
                        events.append((pd.Timestamp(d).tz_localize("UTC"), "OPEN", r["ticker"], tid, ""))
                        free_slots -= 1

        # 5) record daily equity
        eq_end = compute_equity(d)
        equity_curve.append((pd.Timestamp(d).tz_localize("UTC"), eq_end))

    equity_df = pd.DataFrame(equity_curve, columns=["date","equity"])
    events_df = pd.DataFrame(events, columns=["time","event","ticker","trade_id","reason"])
    return equity_df, events_df

def main():
    ms = load_market_state()
    ms_map = dict(zip(ms["d"], ms["market_state"]))

    tickers = list_tickers(DATA_PREFIX)
    print("Tickers:", len(tickers))

    all_trades = []
    diags = []

    for i, tkr in enumerate(tickers, 1):
        try:
            df = read_parquet(f"{DATA_PREFIX}{tkr}.parquet")
            if df is None or df.empty:
                continue
            tr, diag = generate_trades(df, tkr, ms_map)
            diags.append(diag)
            if not tr.empty:
                all_trades.append(tr)
            if i % 50 == 0:
                print(f"Processed {i}/{len(tickers)}")
        except Exception as e:
            print("Error:", tkr, e)

    trades = pd.concat(all_trades, ignore_index=True) if all_trades else pd.DataFrame()
    diag_df = pd.DataFrame(diags)

    equity_df, events_df = simulate_portfolio_daily(trades, ms)

    env_debug = {
        "IMPULSE_RET1_MIN": os.getenv("IMPULSE_RET1_MIN"),
        "IMPULSE_RET2_MIN": os.getenv("IMPULSE_RET2_MIN"),
        "VOL_MULT_IMP": os.getenv("VOL_MULT_IMP"),
        "IMPULSE_MEMORY_DAYS": os.getenv("IMPULSE_MEMORY_DAYS"),
        "CONS_LOOKBACK": os.getenv("CONS_LOOKBACK"),
        "CONS_MAX_RANGE_PCT": os.getenv("CONS_MAX_RANGE_PCT"),
        "ALLOW_ENTRY_STATES": os.getenv("ALLOW_ENTRY_STATES"),
        "FORCE_EXIT_STATES": os.getenv("FORCE_EXIT_STATES"),
        "MAX_POSITIONS": os.getenv("MAX_POSITIONS"),
        "CAPITAL_UTIL": os.getenv("CAPITAL_UTIL"),
        "POS_FRACTION": os.getenv("POS_FRACTION"),
        "TAIL_DAYS": os.getenv("TAIL_DAYS"),
        "TRAIL_MODE": os.getenv("TRAIL_MODE"),
        "TRAIL_ATR_MULT": os.getenv("TRAIL_ATR_MULT"),
        "ENABLE_PARTIAL_TP": os.getenv("ENABLE_PARTIAL_TP"),
        "PARTIAL_TP_R": os.getenv("PARTIAL_TP_R"),
        "PARTIAL_FRACTION": os.getenv("PARTIAL_FRACTION"),
        "MARKET_STATE_KEY": os.getenv("MARKET_STATE_KEY"),
    }

    stats = {
        "mode": "POOS DAILY (POOS-like) + MARKET GATING + TRAILING + DAILY MTM",
        "tickers_tested": int(len(tickers)),
        "trades": int(len(trades)) if not trades.empty else 0,
        "final_equity": float(equity_df["equity"].iloc[-1]) if not equity_df.empty else 1.0,
        "max_positions": int(MAX_POSITIONS),
        "capital_util": float(CAPITAL_UTIL),
        "pos_fraction": float(POS_FRACTION),
        "slippage_bps_per_side": float(SLIPPAGE_BPS),
        "commission_bps_per_side": float(COMMISSION_BPS),
        "allow_entry_states": ALLOW_ENTRY_STATES,
        "force_exit_states": FORCE_EXIT_STATES,
        "setup_params": {
            "impulse_ret1_min": float(IMPULSE_RET1_MIN),
            "impulse_ret2_min": float(IMPULSE_RET2_MIN),
            "vol_mult_imp": float(VOL_MULT_IMP),
            "impulse_memory_days": int(IMPULSE_MEMORY_DAYS),
            "cons_lookback": int(CONS_LOOKBACK),
            "cons_max_range_pct": float(CONS_MAX_RANGE_PCT),
            "use_trend_filter": bool(USE_TREND_FILTER),
            "entry_ema": int(ENTRY_EMA),
            "atr_period": int(ATR_PERIOD),
            "stop_atr": float(STOP_ATR),
            "tp_atr": float(TP_ATR),
            "move_be_pct": float(MOVE_BE_PCT),
            "trail_mode": str(TRAIL_MODE),
            "trail_atr_mult": float(TRAIL_ATR_MULT),
            "enable_partial_tp": bool(ENABLE_PARTIAL_TP),
            "partial_tp_r": float(PARTIAL_TP_R),
            "partial_fraction": float(PARTIAL_FRACTION),
            "max_hold_days": int(MAX_HOLD_DAYS),
            "cooloff_days": int(COOLOFF_DAYS),
            "tail_days": int(TAIL_DAYS),
        },
        "market_state_counts": ms["market_state"].value_counts().to_dict(),
        "diag_sum": {
            "impulses_total": int(diag_df["impulses"].sum()) if not diag_df.empty else 0,
            "impulse_recent_days_total": int(diag_df["impulse_recent_days"].sum()) if not diag_df.empty else 0,
            "tight_days_total": int(diag_df["tight_days"].sum()) if not diag_df.empty else 0,
        },
        "env_debug": env_debug,
    }

    put_text(f"{RESULT_PREFIX}/stats.json", json.dumps(stats, indent=2))
    put_text(f"{RESULT_PREFIX}/trades.csv", trades.to_csv(index=False) if not trades.empty else "ticker,entry_time,exit_time,entry,exit,exit_reason,hold_days,be_moved,score,entry_market_state,size_mult\n")
    put_text(f"{RESULT_PREFIX}/equity.csv", equity_df.to_csv(index=False) if not equity_df.empty else "date,equity\n")
    put_text(f"{RESULT_PREFIX}/events.csv", events_df.to_csv(index=False) if not events_df.empty else "time,event,ticker,trade_id,reason\n")
    put_text(f"{RESULT_PREFIX}/diag.csv", diag_df.to_csv(index=False) if not diag_df.empty else "ticker,rows,impulses,impulse_recent_days,tight_days\n")

    plt.figure()
    if not equity_df.empty:
        plt.plot(pd.to_datetime(equity_df["date"]), equity_df["equity"])
        plt.xticks(rotation=30, ha="right")
        plt.title("POOS Daily Equity (market-gated, trailing, daily MTM)")
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
