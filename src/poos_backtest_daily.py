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

# ===================== R2 =====================
R2_ENDPOINT = os.environ["R2_ENDPOINT"]
R2_KEY = os.environ["R2_ACCESS_KEY_ID"]
R2_SECRET = os.environ["R2_SECRET_ACCESS_KEY"]
R2_BUCKET = os.environ["R2_BUCKET"]

DATA_PREFIX = os.getenv("DATA_PREFIX", "yahoo/1d/")
MARKET_STATE_KEY = os.getenv("MARKET_STATE_KEY", "results/poos_market_engine_v1/market_state.csv")

# optional: ticker,sector (csv)
SECTOR_MAP_KEY = os.getenv("SECTOR_MAP_KEY", "results/poos_meta/sector_map.csv")

RESULT_PREFIX = os.getenv("RESULT_PREFIX", "results/poos_daily_v3_rs_dynexp")

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
TP_ATR = float(os.getenv("TP_ATR", "8.0"))
MOVE_BE_PCT = float(os.getenv("MOVE_BE_PCT", "0.01"))

# trailing
TRAIL_MODE = os.getenv("TRAIL_MODE", "ema_atr")  # ema_atr / close_atr
TRAIL_ATR_MULT = float(os.getenv("TRAIL_ATR_MULT", "2.2"))

# partial tp
ENABLE_PARTIAL_TP = os.getenv("ENABLE_PARTIAL_TP", "0") == "1"
PARTIAL_TP_R = float(os.getenv("PARTIAL_TP_R", "2.0"))
PARTIAL_FRACTION = float(os.getenv("PARTIAL_FRACTION", "0.50"))

MAX_HOLD_DAYS = int(os.getenv("MAX_HOLD_DAYS", "45"))
COOLOFF_DAYS  = int(os.getenv("COOLOFF_DAYS", "10"))

TAIL_DAYS = int(os.getenv("TAIL_DAYS", "504"))

# ===================== RS + Dynamic exposure (NEW) =====================
RS_WIN1 = int(os.getenv("RS_WIN1", "63"))
RS_WIN2 = int(os.getenv("RS_WIN2", "126"))
RS_TOP_Q = float(os.getenv("RS_TOP_Q", "0.70"))  # 0.70 => top 30%

# optional sector filters (enabled only if sector_map loaded)
SECTOR_ENABLE = os.getenv("SECTOR_ENABLE", "1") == "1"
SECTOR_TOP_Q = float(os.getenv("SECTOR_TOP_Q", "0.50"))  # top 50% sectors by avg RS
IN_SECTOR_TOP_Q = float(os.getenv("IN_SECTOR_TOP_Q", "0.60"))  # ticker must be top 40% in its sector

# dynamic exposure by market_score
DYNEXP_ENABLE = os.getenv("DYNEXP_ENABLE", "1") == "1"
SCORE_MIN_EXPO = float(os.getenv("SCORE_MIN_EXPO", "0.20"))  # even if score low but still allowed, keep at least 20% expo
SCORE_MAX_EXPO = float(os.getenv("SCORE_MAX_EXPO", "1.00"))

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
    # ensure market_score exists
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
    """
    Returns DataFrame indexed by date with columns:
      - per-ticker RS score (float) in [0..1] (rank-normalized)
      - rs_threshold (per date quantile)
    """
    if close_pivot.empty:
        return pd.DataFrame()

    ret1 = close_pivot.pct_change(RS_WIN1)
    ret2 = close_pivot.pct_change(RS_WIN2)

    # ranks per date across tickers (ignore NaNs)
    r1 = ret1.rank(axis=1, pct=True)
    r2 = ret2.rank(axis=1, pct=True)

    rs = 0.6 * r1 + 0.4 * r2
    rs["rs_threshold"] = rs.quantile(RS_TOP_Q, axis=1, numeric_only=True)
    return rs

def compute_sector_filters(rs_row: pd.Series, sector_map: dict) -> tuple[set[str], dict]:
    """
    Given rs for a single day (tickers -> score), returns allowed tickers set.
    Also returns debug dict.
    """
    # if no sector map, allow all
    if not sector_map:
        return set(rs_row.index.tolist()), {"sector_enabled": False}

    tickers = [c for c in rs_row.index if c != "rs_threshold"]
    df = pd.DataFrame({"ticker": tickers, "rs": rs_row[tickers].values})
    df["sector"] = df["ticker"].map(sector_map).fillna("UNKNOWN")

    # sector strength = mean rs
    sector_strength = df.groupby("sector")["rs"].mean().sort_values(ascending=False)
    # keep top sectors by quantile
    if len(sector_strength) <= 1:
        top_sectors = set(sector_strength.index.tolist())
    else:
        thr = sector_strength.quantile(SECTOR_TOP_Q)
        top_sectors = set(sector_strength[sector_strength >= thr].index.tolist())

    df["in_top_sector"] = df["sector"].isin(top_sectors)

    # within-sector rank
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

        # manage
        if in_pos:
            hold += 1
            lo, hi, cl = float(row["low"]), float(row["high"]), float(row["close"])

            # move BE
            if (not be_moved) and hi >= entry * (1.0 + MOVE_BE_PCT):
                stop = max(stop, entry)
                be_moved = True

            # trail if BE already moved (keeps it POOS-like; not trailing immediately)
            if be_moved:
                a = row["atr"]
                e = row["ema20"]
                if not pd.isna(a) and not pd.isna(e):
                    if TRAIL_MODE == "ema_atr":
                        trail = float(e) - TRAIL_ATR_MULT * float(a)
                    else:
                        trail = float(cl) - TRAIL_ATR_MULT * float(a)
                    stop = max(stop, trail)

            # partial
            if ENABLE_PARTIAL_TP and (not partial_done) and (r_unit is not None) and (r_unit > 0):
                target_partial = entry + PARTIAL_TP_R * r_unit
                if hi >= target_partial:
                    partial_done = True
                    # record partial as separate trade row
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

            # TP
            if tp is not None and hi >= tp:
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

            # STOP
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

            # TIME
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

        # setup -> watch
        if not in_pos:
            if not watch:
                if (mstate in ALLOW_ENTRY_STATES) and bool(row["impulse_recent"]) and bool(row["trend_ok"]):
                    vol_ratio = float(row["volume"] / row["vol_ma"]) if row["vol_ma"] and not pd.isna(row["vol_ma"]) else 0.0
                    burst = max(float(row["ret1"]) if not pd.isna(row["ret1"]) else 0.0,
                                float(row["ret2"]) if not pd.isna(row["ret2"]) else 0.0)
                    watch_score = float(burst) * float(vol_ratio)
                    watch = True
            else:
                # invalidate watch if market goes off
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

                # entry on pullback to EMA20
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
        return pd.DataFrame(columns=["date","equity"]), pd.DataFrame()

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

    # debug counters
    rs_reject = 0
    sector_reject = 0
    open_attempts = 0
    opened = 0

    for d in cal:
        d_utc = as_utc(pd.Timestamp(d))
        eq = compute_equity(d)

        mstate = ms_map.get(pd.Timestamp(d), "neutral")
        mscore = ms_score_map.get(pd.Timestamp(d), np.nan)
        if pd.isna(mscore):
            mscore = 5.0  # neutral default

        # dynamic scale from score 0..10 -> 0..1
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

                # RS filter for the day
                if (not rs_table.empty) and (pd.Timestamp(d) in rs_table.index):
                    rs_row = rs_table.loc[pd.Timestamp(d)]
                    rs_thr = float(rs_row.get("rs_threshold", np.nan))
                    if not np.isnan(rs_thr):
                        batch["rs"] = batch["ticker"].map(lambda x: float(rs_row.get(x, np.nan)))
                        pre_n = len(batch)
                        batch = batch[batch["rs"] >= rs_thr].copy()
                        rs_reject += int(pre_n - len(batch))
                else:
                    batch["rs"] = np.nan

                # Sector leadership filter (only if map exists)
                sector_dbg = {}
                if sector_loaded and (not rs_table.empty) and (pd.Timestamp(d) in rs_table.index):
                    allowed_by_sector, sector_dbg = compute_sector_filters(rs_table.loc[pd.Timestamp(d)], sector_map)
                    pre_n = len(batch)
                    batch = batch[batch["ticker"].isin(allowed_by_sector)].copy()
                    sector_reject += int(pre_n - len(batch))

                current_equity = compute_equity(d)

                free_slots = eff_max_positions - len(positions)
                if free_slots > 0 and current_equity > 0:
                    # strongest setups first
                    batch = batch.sort_values(["score","rs"], ascending=False)
                    prices = close_pivot.loc[d]

                    for _, r in batch.iterrows():
                        open_attempts += 1
                        if free_slots <= 0:
                            break

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

                        px = prices.get(r["ticker"], np.nan)
                        if pd.isna(px):
                            continue

                        entry_adj, _ = apply_costs(float(px), float(px))
                        notional = pos_frac * current_equity
                        cash -= notional

                        tid = int(r["trade_id"])
                        positions[tid] = {"ticker": r["ticker"], "entry_px": float(entry_adj), "notional": float(notional)}
                        events.append((d_utc, "OPEN", r["ticker"], tid,
                                       f"mscore={mscore:.2f},scale={scale:.2f},rs={float(r.get('rs',np.nan)):.3f}"))
                        opened += 1
                        free_slots -= 1

        eq_end = compute_equity(d)
        equity_curve.append((d_utc, eq_end))

    equity_df = pd.DataFrame(equity_curve, columns=["date","equity"])
    events_df = pd.DataFrame(events, columns=["time","event","ticker","trade_id","reason"])

    debug = {
        "rs_reject": int(rs_reject),
        "sector_reject": int(sector_reject),
        "open_attempts": int(open_attempts),
        "opened": int(opened),
    }
    return equity_df, events_df, debug

def main():
    ms = load_market_state()
    tickers = list_tickers(DATA_PREFIX)
    print("Tickers:", len(tickers))

    # build calendar based on available closes
    # (use the overlapping date range of all tickers we have)
    # We'll just use the union calendar from pivot; NaNs are fine for non-traded tickers on that day.
    # Determine rough range from market_state dates
    start = ms["d"].min()
    end = ms["d"].max()

    close_pivot = build_close_matrix(tickers, start, end)
    if close_pivot.empty:
        print("No close data found.")
        return

    # RS table across your 250 tickers
    rs_table = compute_rs_table(close_pivot)

    # optional sector map
    sector_map, sector_loaded = try_load_sector_map(tickers)
    print("Sector map loaded:", sector_loaded, "count:", len(sector_map))

    # market state map (for setup stage)
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

    # RS snapshot (last date): which tickers are leaders now
    rs_snapshot = pd.DataFrame()
    last_d = close_pivot.index.max()
    if (not rs_table.empty) and (last_d in rs_table.index):
        row = rs_table.loc[last_d].drop(labels=["rs_threshold"], errors="ignore")
        rs_snapshot = row.rename("rs").to_frame().reset_index().rename(columns={"index":"ticker"})
        rs_snapshot = rs_snapshot.sort_values("rs", ascending=False)
        thr = float(rs_table.loc[last_d].get("rs_threshold", np.nan))
        rs_snapshot["is_top"] = rs_snapshot["rs"] >= thr

    env_debug = {
        # setup
        "IMPULSE_RET1_MIN": os.getenv("IMPULSE_RET1_MIN"),
        "IMPULSE_RET2_MIN": os.getenv("IMPULSE_RET2_MIN"),
        "VOL_MULT_IMP": os.getenv("VOL_MULT_IMP"),
        "IMPULSE_MEMORY_DAYS": os.getenv("IMPULSE_MEMORY_DAYS"),
        "CONS_MAX_RANGE_PCT": os.getenv("CONS_MAX_RANGE_PCT"),
        "MAX_HOLD_DAYS": os.getenv("MAX_HOLD_DAYS"),
        "STOP_ATR": os.getenv("STOP_ATR"),
        "TP_ATR": os.getenv("TP_ATR"),
        "TRAIL_ATR_MULT": os.getenv("TRAIL_ATR_MULT"),
        "ENABLE_PARTIAL_TP": os.getenv("ENABLE_PARTIAL_TP"),
        # portfolio
        "MAX_POSITIONS": os.getenv("MAX_POSITIONS"),
        "CAPITAL_UTIL": os.getenv("CAPITAL_UTIL"),
        "POS_FRACTION": os.getenv("POS_FRACTION"),
        # RS/dynexp
        "RS_WIN1": os.getenv("RS_WIN1"),
        "RS_WIN2": os.getenv("RS_WIN2"),
        "RS_TOP_Q": os.getenv("RS_TOP_Q"),
        "DYNEXP_ENABLE": os.getenv("DYNEXP_ENABLE"),
        "SCORE_MIN_EXPO": os.getenv("SCORE_MIN_EXPO"),
        "SECTOR_MAP_KEY": os.getenv("SECTOR_MAP_KEY"),
        "SECTOR_ENABLE": os.getenv("SECTOR_ENABLE"),
        "SECTOR_TOP_Q": os.getenv("SECTOR_TOP_Q"),
        "IN_SECTOR_TOP_Q": os.getenv("IN_SECTOR_TOP_Q"),
    }

    stats = {
        "mode": "POOS DAILY v3 (POOS-like setups) + MARKET GATING + RS TOP + DYN EXPOSURE" + (" + SECTOR" if sector_loaded else ""),
        "tickers_tested": int(len(tickers)),
        "raw_trades": int(len(trades)) if not trades.empty else 0,
        "final_equity": float(equity_df["equity"].iloc[-1]) if not equity_df.empty else 1.0,
        "base": {
            "max_positions": int(BASE_MAX_POSITIONS),
            "capital_util": float(BASE_CAPITAL_UTIL),
            "pos_fraction": float(BASE_POS_FRACTION),
        },
        "costs": {
            "slippage_bps_per_side": float(SLIPPAGE_BPS),
            "commission_bps_per_side": float(COMMISSION_BPS),
        },
        "market_gating": {
            "allow_entry_states": ALLOW_ENTRY_STATES,
            "force_exit_states": FORCE_EXIT_STATES,
            "market_state_counts": ms["market_state"].value_counts().to_dict(),
        },
        "rs_filter": {
            "rs_win1": int(RS_WIN1),
            "rs_win2": int(RS_WIN2),
            "rs_top_q": float(RS_TOP_Q),
        },
        "sector_filter": {
            "enabled": bool(sector_loaded),
            "sector_top_q": float(SECTOR_TOP_Q),
            "in_sector_top_q": float(IN_SECTOR_TOP_Q),
            "sector_map_key": SECTOR_MAP_KEY,
            "sector_map_loaded_rows": int(len(sector_map)),
        },
        "dyn_exposure": {
            "enabled": bool(DYNEXP_ENABLE),
            "score_min_expo": float(SCORE_MIN_EXPO),
            "score_max_expo": float(SCORE_MAX_EXPO),
        },
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
        "diag_sum": {
            "impulses_total": int(diag_df["impulses"].sum()) if not diag_df.empty else 0,
            "impulse_recent_days_total": int(diag_df["impulse_recent_days"].sum()) if not diag_df.empty else 0,
            "tight_days_total": int(diag_df["tight_days"].sum()) if not diag_df.empty else 0,
        },
        "sim_debug": sim_debug,
        "env_debug": env_debug,
    }

    # outputs
    put_text(f"{RESULT_PREFIX}/stats.json", json.dumps(stats, indent=2))
    put_text(f"{RESULT_PREFIX}/trades.csv",
             trades.to_csv(index=False) if not trades.empty
             else "ticker,entry_time,exit_time,entry,exit,exit_reason,hold_days,be_moved,score,entry_market_state,size_mult\n")
    put_text(f"{RESULT_PREFIX}/equity.csv", equity_df.to_csv(index=False) if not equity_df.empty else "date,equity\n")
    put_text(f"{RESULT_PREFIX}/events.csv", events_df.to_csv(index=False) if not events_df.empty else "time,event,ticker,trade_id,reason\n")
    put_text(f"{RESULT_PREFIX}/diag.csv",
             diag_df.to_csv(index=False) if not diag_df.empty else "ticker,rows,impulses,impulse_recent_days,tight_days\n")
    if not rs_snapshot.empty:
        put_text(f"{RESULT_PREFIX}/rs_snapshot.csv", rs_snapshot.to_csv(index=False))
    put_text(f"{RESULT_PREFIX}/market_score.csv", ms.to_csv(index=False))

    # equity plot
    plt.figure()
    if not equity_df.empty:
        plt.plot(pd.to_datetime(equity_df["date"]), equity_df["equity"])
        plt.xticks(rotation=30, ha="right")
        plt.title("POOS Daily Equity (RS top + dyn exposure, costs)")
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
