import os
import io
import json
import math
import numpy as np
import pandas as pd
import boto3
from botocore.config import Config
from dotenv import load_dotenv

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

load_dotenv()

# ===================== R2 =====================
R2_ENDPOINT = os.environ["R2_ENDPOINT"]
R2_KEY = os.environ["R2_ACCESS_KEY_ID"]
R2_SECRET = os.environ["R2_SECRET_ACCESS_KEY"]
R2_BUCKET = os.environ["R2_BUCKET"]

DATA_PREFIX = os.getenv("DATA_PREFIX", "yahoo/1d/")
MARKET_STATE_KEY = os.getenv("MARKET_STATE_KEY", "results/poos_market_engine_v1/market_state.csv")

# optional: ticker,sector (csv)
SECTOR_MAP_KEY = os.getenv("SECTOR_MAP_KEY", "results/poos_meta/sector_map.csv")

RESULT_PREFIX = os.getenv("RESULT_PREFIX", "results/poos_daily_v3_2_moderate")

# ===================== Costs =====================
SLIPPAGE_BPS   = float(os.getenv("SLIPPAGE_BPS", "5"))
COMMISSION_BPS = float(os.getenv("COMMISSION_BPS", "1"))

# ===================== Portfolio (base) =====================
BASE_MAX_POSITIONS = int(os.getenv("MAX_POSITIONS", "75"))
BASE_CAPITAL_UTIL  = float(os.getenv("CAPITAL_UTIL", "1.50"))
BASE_POS_FRACTION  = float(os.getenv("POS_FRACTION", "0.03"))

# ===================== Market gating =====================
ALLOW_ENTRY_STATES = [s.strip() for s in os.getenv("ALLOW_ENTRY_STATES", "risk_on,neutral").split(",") if s.strip()]
FORCE_EXIT_STATES  = [s.strip() for s in os.getenv("FORCE_EXIT_STATES", "risk_off").split(",") if s.strip()]

# ===================== POOS-like setup =====================
IMPULSE_RET1_MIN = float(os.getenv("IMPULSE_RET1_MIN", "0.04"))
IMPULSE_RET2_MIN = float(os.getenv("IMPULSE_RET2_MIN", "0.07"))

VOL_MA_DAYS      = int(os.getenv("VOL_MA_DAYS", "20"))
VOL_MULT_IMP     = float(os.getenv("VOL_MULT_IMP", "1.2"))
IMPULSE_MEMORY_DAYS = int(os.getenv("IMPULSE_MEMORY_DAYS", "21"))

CONS_LOOKBACK = int(os.getenv("CONS_LOOKBACK", "5"))
CONS_MAX_RANGE_PCT = float(os.getenv("CONS_MAX_RANGE_PCT", "0.12"))

USE_TREND_FILTER = os.getenv("USE_TREND_FILTER", "1") == "1"
ENTRY_EMA = int(os.getenv("ENTRY_EMA", "20"))
ATR_PERIOD = int(os.getenv("ATR_PERIOD", "14"))

STOP_ATR = float(os.getenv("STOP_ATR", "1.5"))
TP_ATR = float(os.getenv("TP_ATR", "8.0"))  # keep as far target; trailing will usually manage
MOVE_BE_PCT = float(os.getenv("MOVE_BE_PCT", "0.01"))

# trailing (moderate)
TRAIL_MODE = os.getenv("TRAIL_MODE", "ema_atr")  # ema_atr / close_atr
TRAIL_ATR_MULT = float(os.getenv("TRAIL_ATR_MULT", "1.8"))

# partial tp (moderate default ON via env)
ENABLE_PARTIAL_TP = os.getenv("ENABLE_PARTIAL_TP", "1") == "1"
PARTIAL_TP_R = float(os.getenv("PARTIAL_TP_R", "2.0"))
PARTIAL_FRACTION = float(os.getenv("PARTIAL_FRACTION", "0.50"))

MAX_HOLD_DAYS = int(os.getenv("MAX_HOLD_DAYS", "45"))
COOLOFF_DAYS  = int(os.getenv("COOLOFF_DAYS", "10"))

TAIL_DAYS = int(os.getenv("TAIL_DAYS", "504"))

# ===================== RS + Dynamic exposure =====================
RS_WIN1 = int(os.getenv("RS_WIN1", "63"))
RS_WIN2 = int(os.getenv("RS_WIN2", "126"))
RS_TOP_Q = float(os.getenv("RS_TOP_Q", "0.70"))  # 0.70 => top 30%

# optional sector filters (enabled only if sector_map loaded)
SECTOR_ENABLE = os.getenv("SECTOR_ENABLE", "1") == "1"
SECTOR_TOP_Q = float(os.getenv("SECTOR_TOP_Q", "0.50"))       # top 50% sectors by avg RS
IN_SECTOR_TOP_Q = float(os.getenv("IN_SECTOR_TOP_Q", "0.60")) # ticker must be top 40% in its sector

# dynamic exposure by market_score
DYNEXP_ENABLE = os.getenv("DYNEXP_ENABLE", "1") == "1"
SCORE_MIN_EXPO = float(os.getenv("SCORE_MIN_EXPO", "0.20"))
SCORE_MAX_EXPO = float(os.getenv("SCORE_MAX_EXPO", "1.00"))

# ===================== NEW: score sizing + entry throttles =====================
SCORE_SIZING_ENABLE = os.getenv("SCORE_SIZING_ENABLE", "1") == "1"
SCORE_SIZING_TOP_Q = float(os.getenv("SCORE_SIZING_TOP_Q", "0.75"))  # top 25%
SCORE_SIZING_MULT  = float(os.getenv("SCORE_SIZING_MULT",  "1.25"))  # mild boost

MAX_OPENS_PER_DAY = int(os.getenv("MAX_OPENS_PER_DAY", "25"))  # keeps portfolio focused
ONE_OPEN_PER_TICKER_PER_DAY = os.getenv("ONE_OPEN_PER_TICKER_PER_DAY", "1") == "1"

# ===================== R2 client =====================
s3 = boto3.client(
    "s3",
    endpoint_url=R2_ENDPOINT,
    aws_access_key_id=R2_KEY,
    aws_secret_access_key=R2_SECRET,
    region_name="auto",
    config=Config(signature_version="s3v4", retries={"max_attempts": 10, "mode": "standard"}),
)

def as_utc(ts: pd.Timestamp) -> pd.Timestamp:
    if not isinstance(ts, pd.Timestamp):
        ts = pd.Timestamp(ts)
    if ts.tzinfo is None:
        return ts.tz_localize("UTC")
    return ts.tz_convert("UTC")

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
    if "market_score" not in ms.columns:
        ms["market_score"] = np.nan
    return ms[["date","d","market_state","market_score"]].sort_values("date")

def try_load_sector_map(tickers: list[str]) -> tuple[dict, bool]:
    if not SECTOR_ENABLE:
        return {}, False
    try:
        txt = read_text(SECTOR_MAP_KEY)
        sm = pd.read_csv(io.StringIO(txt))
        if "ticker" not in sm.columns or "sector" not in sm.columns:
            return {}, False
        sm["ticker"] = sm["ticker"].astype(str)
        sm["sector"] = sm["sector"].astype(str)
        sm = sm[sm["ticker"].isin(tickers)]
        return dict(zip(sm["ticker"], sm["sector"])), True
    except Exception:
        return {}, False

def build_close_matrix(tickers: list[str], start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
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
    if not closes:
        return pd.DataFrame()
    close_df = pd.concat(closes, ignore_index=True)
    close_df = close_df.rename(columns={"d":"date"})
    return close_df.pivot(index="date", columns="ticker", values="close").sort_index()

def compute_rs_table(close_pivot: pd.DataFrame) -> pd.DataFrame:
    if close_pivot.empty:
        return pd.DataFrame()
    ret1 = close_pivot.pct_change(RS_WIN1, fill_method=None)
    ret2 = close_pivot.pct_change(RS_WIN2, fill_method=None)
    r1 = ret1.rank(axis=1, pct=True)
    r2 = ret2.rank(axis=1, pct=True)
    rs = 0.6 * r1 + 0.4 * r2
    rs["rs_threshold"] = rs.quantile(RS_TOP_Q, axis=1, numeric_only=True)
    return rs

def compute_sector_filters(rs_row: pd.Series, sector_map: dict) -> tuple[set[str], dict]:
    if not sector_map:
        tickers = [c for c in rs_row.index if c != "rs_threshold"]
        return set(tickers), {"sector_enabled": False}

    tickers = [c for c in rs_row.index if c != "rs_threshold"]
    df = pd.DataFrame({"ticker": tickers, "rs": rs_row[tickers].values})
    df["sector"] = df["ticker"].map(sector_map).fillna("UNKNOWN")

    sector_strength = df.groupby("sector")["rs"].mean().sort_values(ascending=False)
    if len(sector_strength) <= 1:
        top_sectors = set(sector_strength.index.tolist())
    else:
        thr = sector_strength.quantile(SECTOR_TOP_Q)
        top_sectors = set(sector_strength[sector_strength >= thr].index.tolist())

    df["in_top_sector"] = df["sector"].isin(top_sectors)
    df["rs_in_sector_pct"] = df.groupby("sector")["rs"].rank(pct=True)
    df["in_sector_top"] = df["rs_in_sector_pct"] >= IN_SECTOR_TOP_Q

    allowed = set(df[(df["in_top_sector"]) & (df["in_sector_top"])]["ticker"].tolist())
    dbg = {
        "sector_enabled": True,
        "sectors_total": int(sector_strength.shape[0]),
        "top_sectors": int(len(top_sectors)),
        "allowed_after_sector": int(len(allowed)),
    }
    return allowed, dbg

def compute_consolidation(df: pd.DataFrame) -> pd.Series:
    roll_hi = df["high"].rolling(CONS_LOOKBACK, min_periods=CONS_LOOKBACK).max()
    roll_lo = df["low"].rolling(CONS_LOOKBACK, min_periods=CONS_LOOKBACK).min()
    rng = (roll_hi - roll_lo) / df["close"]
    return rng

def generate_trades_for_ticker(df: pd.DataFrame, ticker: str, ms_map: dict):
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

    watch = False
    cooldown_until = -1

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
    partial_done = False
    r_unit = None

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

        if in_pos:
            hold += 1
            lo, hi, cl = float(row["low"]), float(row["high"]), float(row["close"])

            # move to BE once price moves sufficiently
            if (not be_moved) and hi >= entry * (1.0 + MOVE_BE_PCT):
                stop = max(stop, entry)
                be_moved = True

            # trailing after BE
            if be_moved:
                a = row["atr"]
                e = row["ema20"]
                if not pd.isna(a) and not pd.isna(e):
                    if TRAIL_MODE == "ema_atr":
                        trail = float(e) - TRAIL_ATR_MULT * float(a)
                    else:
                        trail = float(cl) - TRAIL_ATR_MULT * float(a)
                    stop = max(stop, trail)

            # partial take profit
            if ENABLE_PARTIAL_TP and (not partial_done) and (r_unit is not None) and (r_unit > 0):
                target_partial = entry + PARTIAL_TP_R * r_unit
                if hi >= target_partial:
                    partial_done = True
                    # after partial, force BE
                    stop = max(stop, entry)
                    be_moved = True
                    trades.append({
                        "ticker": ticker,
                        "entry_time": entry_time,
                        "exit_time": t,
                        "entry": float(entry),
                        "exit": float(target_partial),
                        "exit_reason": f"PARTIAL_{PARTIAL_TP_R}R",
                        "hold_days": int(hold),
                        "be_moved": bool(be_moved),
                        "score": float(watch_score) if watch_score is not None else np.nan,
                        "entry_market_state": entry_mstate,
                        "size_mult": float(PARTIAL_FRACTION),
                    })

            # optional far TP (rare if trailing/partial works)
            if tp is not None and TP_ATR and TP_ATR > 0 and hi >= tp:
                rem = 1.0 - (PARTIAL_FRACTION if partial_done else 0.0)
                if rem > 0:
                    trades.append({
                        "ticker": ticker,
                        "entry_time": entry_time,
                        "exit_time": t,
                        "entry": float(entry),
                        "exit": float(tp),
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

            if lo <= stop:
                rem = 1.0 - (PARTIAL_FRACTION if partial_done else 0.0)
                if rem > 0:
                    trades.append({
                        "ticker": ticker,
                        "entry_time": entry_time,
                        "exit_time": t,
                        "entry": float(entry),
                        "exit": float(stop),
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
                e = row["ema20"]
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
                    r_unit = max(0.0, entry - initial_stop)

                    in_pos = True
                    watch = False

    return pd.DataFrame(trades), diag

def simulate_portfolio_daily(trades: pd.DataFrame,
                             market_state: pd.DataFrame,
                             close_pivot: pd.DataFrame,
                             rs_table: pd.DataFrame,
                             sector_map: dict,
                             sector_loaded: bool):
    if trades.empty or close_pivot.empty:
        return pd.DataFrame(columns=["date","equity"]), pd.DataFrame(), {"opened": 0, "open_attempts": 0, "rs_reject": 0, "sector_reject": 0}

    trades = trades.copy()
    trades["entry_time"] = pd.to_datetime(trades["entry_time"], utc=True)
    trades["exit_time"]  = pd.to_datetime(trades["exit_time"], utc=True)
    if "size_mult" not in trades.columns:
        trades["size_mult"] = 1.0
    trades["size_mult"] = pd.to_numeric(trades["size_mult"], errors="coerce").fillna(1.0)

    ms = market_state.copy()
    ms["d"] = pd.to_datetime(ms["d"], utc=True)
    ms_map = dict(zip(ms["d"], ms["market_state"]))
    ms_score_map = dict(zip(ms["d"], ms["market_score"]))

    cal = close_pivot.index
    trades["d_entry"] = trades["entry_time"].dt.normalize()
    trades["d_exit"]  = trades["exit_time"].dt.normalize()

    trades["score"] = pd.to_numeric(trades.get("score", 0.0), errors="coerce").fillna(0.0)
    trades = trades.sort_values(["entry_time","score"], ascending=[True, False]).reset_index(drop=True)
    trades["trade_id"] = np.arange(len(trades))

    entries_by_day = {d: df for d, df in trades.groupby("d_entry")}
    exits_by_day = {d: df for d, df in trades.groupby("d_exit")}

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
                eq += p["notional"] * (float(px) / p["entry_px"])
        return float(eq)

    def used_util(current_equity):
        notional_sum = sum(p["notional"] for p in positions.values())
        return (notional_sum / current_equity) if current_equity > 0 else 0.0

    rs_reject = 0
    sector_reject = 0
    open_attempts = 0
    opened = 0
    opened_by_day = 0
    skip_same_ticker = 0
    skip_day_limit = 0
    sized_boosted = 0

    for d in cal:
        opened_by_day = 0
        d_utc = as_utc(pd.Timestamp(d))
        eq = compute_equity(d)

        mstate = ms_map.get(pd.Timestamp(d), "neutral")
        mscore = ms_score_map.get(pd.Timestamp(d), np.nan)
        if pd.isna(mscore):
            mscore = 5.0

        if DYNEXP_ENABLE:
            scale = float(np.clip(mscore / 10.0, SCORE_MIN_EXPO, SCORE_MAX_EXPO))
        else:
            scale = 1.0

        eff_max_positions = max(1, int(round(BASE_MAX_POSITIONS * scale)))
        eff_capital_util  = float(BASE_CAPITAL_UTIL * scale)
        eff_pos_fraction  = float(BASE_POS_FRACTION * scale)

        # forced exit risk_off
        if mstate in FORCE_EXIT_STATES and positions:
            prices = close_pivot.loc[d]
            to_close = list(positions.keys())
            for tid in to_close:
                p = positions.pop(tid)
                px = prices.get(p["ticker"], np.nan)
                if pd.isna(px):
                    px = p["entry_px"]
                entry_adj, exit_adj = apply_costs(p["entry_px"], float(px))
                ret = (exit_adj / entry_adj) - 1.0
                cash += p["notional"] * (1.0 + ret)
                events.append((d_utc, "FORCE_EXIT", p["ticker"], int(tid), "MARKET_RISK_OFF"))
            eq = compute_equity(d)

        # scheduled exits
        if pd.Timestamp(d) in exits_by_day and positions:
            prices = close_pivot.loc[d]
            for _, r in exits_by_day[pd.Timestamp(d)].iterrows():
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
                events.append((d_utc, "CLOSE", p["ticker"], tid, str(r.get("exit_reason", ""))))
            eq = compute_equity(d)

        # opens
        if pd.Timestamp(d) in entries_by_day:
            if mstate in ALLOW_ENTRY_STATES:
                batch = entries_by_day[pd.Timestamp(d)].copy()

                # one open per ticker per day
                if ONE_OPEN_PER_TICKER_PER_DAY and not batch.empty:
                    batch = batch.sort_values("score", ascending=False)
                    batch = batch.drop_duplicates(subset=["ticker"], keep="first")

                # ALWAYS create rs col
                batch["rs"] = np.nan

                # RS filter for day if available
                if (not rs_table.empty) and (pd.Timestamp(d) in rs_table.index):
                    rs_row = rs_table.loc[pd.Timestamp(d)]
                    rs_thr = float(rs_row.get("rs_threshold", np.nan))
                    batch["rs"] = batch["ticker"].map(lambda x: float(rs_row.get(x, np.nan)))
                    if not np.isnan(rs_thr):
                        pre_n = len(batch)
                        batch = batch[batch["rs"] >= rs_thr].copy()
                        rs_reject += int(pre_n - len(batch))

                # sector filter (optional)
                if sector_loaded and (not rs_table.empty) and (pd.Timestamp(d) in rs_table.index):
                    allowed_by_sector, _ = compute_sector_filters(rs_table.loc[pd.Timestamp(d)], sector_map)
                    pre_n = len(batch)
                    batch = batch[batch["ticker"].isin(allowed_by_sector)].copy()
                    sector_reject += int(pre_n - len(batch))

                if batch.empty:
                    continue

                # score sizing threshold inside the day (mild)
                if SCORE_SIZING_ENABLE:
                    thr = batch["score"].quantile(SCORE_SIZING_TOP_Q) if len(batch) > 1 else batch["score"].iloc[0]
                    batch["size_mult"] = np.where(batch["score"] >= thr, SCORE_SIZING_MULT, 1.0)
                else:
                    batch["size_mult"] = 1.0

                current_equity = compute_equity(d)
                free_slots = eff_max_positions - len(positions)
                if free_slots > 0 and current_equity > 0:
                    batch = batch.sort_values(["score","rs"], ascending=False)

                    prices = close_pivot.loc[d]
                    opened_tickers_today = set()

                    for _, r in batch.iterrows():
                        open_attempts += 1

                        if opened_by_day >= MAX_OPENS_PER_DAY:
                            skip_day_limit += 1
                            break

                        if free_slots <= 0:
                            break

                        tkr = r["ticker"]
                        if ONE_OPEN_PER_TICKER_PER_DAY and tkr in opened_tickers_today:
                            skip_same_ticker += 1
                            continue

                        current_equity = compute_equity(d)
                        current_util = used_util(current_equity)
                        free_util = max(0.0, eff_capital_util - current_util)
                        if free_util <= 0:
                            break

                        size_mult = float(r.get("size_mult", 1.0))
                        pos_frac = eff_pos_fraction * size_mult
                        if pos_frac <= 0:
                            continue
                        if pos_frac > free_util:
                            continue

                        px = prices.get(tkr, np.nan)
                        if pd.isna(px):
                            continue

                        entry_adj, _ = apply_costs(float(px), float(px))
                        notional = pos_frac * current_equity
                        cash -= notional

                        tid = int(r["trade_id"])
                        positions[tid] = {"ticker": tkr, "entry_px": float(entry_adj), "notional": float(notional)}
                        if size_mult > 1.0:
                            sized_boosted += 1
                        opened += 1
                        opened_by_day += 1
                        opened_tickers_today.add(tkr)

                        events.append((d_utc, "OPEN", tkr, tid,
                                       f"mscore={mscore:.2f},scale={scale:.2f},rs={float(r.get('rs',np.nan)):.3f},size_mult={size_mult:.2f}"))
                        free_slots -= 1

        equity_curve.append((d_utc, compute_equity(d)))

    equity_df = pd.DataFrame(equity_curve, columns=["date","equity"])
    events_df = pd.DataFrame(events, columns=["time","event","ticker","trade_id","reason"])

    debug = {
        "rs_reject": int(rs_reject),
        "sector_reject": int(sector_reject),
        "open_attempts": int(open_attempts),
        "opened": int(opened),
        "skip_same_ticker": int(skip_same_ticker),
        "skip_day_limit": int(skip_day_limit),
        "sized_boosted": int(sized_boosted),
    }
    return equity_df, events_df, debug

def main():
    ms = load_market_state()
    tickers = list_tickers(DATA_PREFIX)
    print("Tickers:", len(tickers))

    start = ms["d"].min()
    end = ms["d"].max()

    close_pivot = build_close_matrix(tickers, start, end)
    if close_pivot.empty:
        print("No close data found.")
        return

    rs_table = compute_rs_table(close_pivot)

    sector_map, sector_loaded = try_load_sector_map(tickers)
    print("Sector map loaded:", sector_loaded, "count:", len(sector_map))

    ms_map = dict(zip(ms["d"], ms["market_state"]))

    all_trades = []
    diags = []

    for i, tkr in enumerate(tickers, 1):
        try:
            df = read_parquet(f"{DATA_PREFIX}{tkr}.parquet")
            if df is None or df.empty:
                continue
            tr, diag = generate_trades_for_ticker(df, tkr, ms_map)
            diags.append(diag)
            if not tr.empty:
                all_trades.append(tr)
            if i % 50 == 0:
                print(f"Processed {i}/{len(tickers)}")
        except Exception as e:
            print("Error:", tkr, e)

    trades = pd.concat(all_trades, ignore_index=True) if all_trades else pd.DataFrame()
    diag_df = pd.DataFrame(diags)

    equity_df, events_df, sim_debug = simulate_portfolio_daily(trades, ms, close_pivot, rs_table, sector_map, sector_loaded)

    env_debug = {
        "RS_WIN1": os.getenv("RS_WIN1"),
        "RS_WIN2": os.getenv("RS_WIN2"),
        "RS_TOP_Q": os.getenv("RS_TOP_Q"),
        "DYNEXP_ENABLE": os.getenv("DYNEXP_ENABLE"),
        "SCORE_MIN_EXPO": os.getenv("SCORE_MIN_EXPO"),
        "SCORE_MAX_EXPO": os.getenv("SCORE_MAX_EXPO"),
        "ENABLE_PARTIAL_TP": os.getenv("ENABLE_PARTIAL_TP"),
        "PARTIAL_TP_R": os.getenv("PARTIAL_TP_R"),
        "PARTIAL_FRACTION": os.getenv("PARTIAL_FRACTION"),
        "TRAIL_ATR_MULT": os.getenv("TRAIL_ATR_MULT"),
        "SCORE_SIZING_ENABLE": os.getenv("SCORE_SIZING_ENABLE"),
        "SCORE_SIZING_TOP_Q": os.getenv("SCORE_SIZING_TOP_Q"),
        "SCORE_SIZING_MULT": os.getenv("SCORE_SIZING_MULT"),
        "MAX_OPENS_PER_DAY": os.getenv("MAX_OPENS_PER_DAY"),
    }

    stats = {
        "mode": "POOS DAILY v3.2 MODERATE (partial TP + mild score sizing + RS + dyn exp)",
        "tickers_tested": int(len(tickers)),
        "raw_trades": int(len(trades)) if not trades.empty else 0,
        "final_equity": float(equity_df["equity"].iloc[-1]) if not equity_df.empty else 1.0,
        "sim_debug": sim_debug,
        "setup_params": {
            "enable_partial_tp": bool(ENABLE_PARTIAL_TP),
            "partial_tp_r": float(PARTIAL_TP_R),
            "partial_fraction": float(PARTIAL_FRACTION),
            "trail_mode": str(TRAIL_MODE),
            "trail_atr_mult": float(TRAIL_ATR_MULT),
            "score_sizing_enable": bool(SCORE_SIZING_ENABLE),
            "score_sizing_top_q": float(SCORE_SIZING_TOP_Q),
            "score_sizing_mult": float(SCORE_SIZING_MULT),
            "max_opens_per_day": int(MAX_OPENS_PER_DAY),
            "rs_win1": int(RS_WIN1),
            "rs_win2": int(RS_WIN2),
            "rs_top_q": float(RS_TOP_Q),
        },
        "env_debug": env_debug,
    }

    put_text(f"{RESULT_PREFIX}/stats.json", json.dumps(stats, indent=2))
    put_text(f"{RESULT_PREFIX}/trades.csv", trades.to_csv(index=False) if not trades.empty else "ticker,entry_time,exit_time,entry,exit,exit_reason,hold_days,be_moved,score,entry_market_state,size_mult\n")
    put_text(f"{RESULT_PREFIX}/equity.csv", equity_df.to_csv(index=False) if not equity_df.empty else "date,equity\n")
    put_text(f"{RESULT_PREFIX}/events.csv", events_df.to_csv(index=False) if not events_df.empty else "time,event,ticker,trade_id,reason\n")
    if not diag_df.empty:
        put_text(f"{RESULT_PREFIX}/diag.csv", diag_df.to_csv(index=False))

    plt.figure()
    if not equity_df.empty:
        plt.plot(pd.to_datetime(equity_df["date"]), equity_df["equity"])
        plt.xticks(rotation=30, ha="right")
        plt.title("POOS Daily Equity (v3.2 moderate)")
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
