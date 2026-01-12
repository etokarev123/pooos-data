# --- отличия от предыдущей версии ---
# 1) добавлен IMPULSE_MEMORY_DAYS (по умолчанию 7)
# 2) watch стартует по impulse_recent, а не только в день импульса
# 3) сохраняем диагностику: сколько импульсов и сколько из них в risk_on

import os, io, json
import pandas as pd
import numpy as np
import boto3
from botocore.config import Config
from dotenv import load_dotenv

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

load_dotenv()

R2_ENDPOINT = os.environ["R2_ENDPOINT"]
R2_KEY = os.environ["R2_ACCESS_KEY_ID"]
R2_SECRET = os.environ["R2_SECRET_ACCESS_KEY"]
R2_BUCKET = os.environ["R2_BUCKET"]

DATA_PREFIX = os.getenv("DATA_PREFIX", "yahoo/1d/")
MARKET_STATE_KEY = os.getenv("MARKET_STATE_KEY", "results/poos_market_engine_v1/market_state.csv")

IMP_LOOKBACK_DAYS = int(os.getenv("IMP_LOOKBACK_DAYS", "20"))
IMP_MIN_RET = float(os.getenv("IMP_MIN_RET", "0.30"))
VOL_MA_DAYS = int(os.getenv("VOL_MA_DAYS", "20"))
VOL_MULT = float(os.getenv("VOL_MULT", "2.0"))

# ВАЖНО: память импульса
IMPULSE_MEMORY_DAYS = int(os.getenv("IMPULSE_MEMORY_DAYS", "7"))

USE_TREND_FILTER = os.getenv("USE_TREND_FILTER", "1") == "1"
ENTRY_EMA = int(os.getenv("ENTRY_EMA", "20"))

ATR_PERIOD = int(os.getenv("ATR_PERIOD", "14"))
STOP_ATR = float(os.getenv("STOP_ATR", "1.0"))
TP_ATR = float(os.getenv("TP_ATR", "1.5"))
MOVE_BE_PCT = float(os.getenv("MOVE_BE_PCT", "0.01"))

MAX_HOLD_DAYS = int(os.getenv("MAX_HOLD_DAYS", "15"))
COOLOFF_DAYS = int(os.getenv("COOLOFF_DAYS", "10"))
TAIL_DAYS = int(os.getenv("TAIL_DAYS", "504"))

MAX_POSITIONS = int(os.getenv("MAX_POSITIONS", "40"))
CAPITAL_UTIL  = float(os.getenv("CAPITAL_UTIL", "0.75"))
POS_FRACTION  = float(os.getenv("POS_FRACTION", "")) if os.getenv("POS_FRACTION") else (CAPITAL_UTIL / MAX_POSITIONS)

SLIPPAGE_BPS   = float(os.getenv("SLIPPAGE_BPS", "5"))
COMMISSION_BPS = float(os.getenv("COMMISSION_BPS", "1"))

ALLOW_ENTRY_STATES = [s.strip() for s in os.getenv("ALLOW_ENTRY_STATES", "risk_on").split(",") if s.strip()]
FORCE_EXIT_STATES  = [s.strip() for s in os.getenv("FORCE_EXIT_STATES", "risk_off").split(",") if s.strip()]

RESULT_PREFIX = os.getenv("RESULT_PREFIX", "results/poos_daily_v2_market_gated")

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

def generate_trades(df: pd.DataFrame, ticker: str, ms_map: dict) -> tuple[pd.DataFrame, dict]:
    df = df.sort_values("date").dropna(subset=["open","high","low","close","volume"]).copy()
    df["date"] = pd.to_datetime(df["date"], utc=True)

    if TAIL_DAYS > 0 and len(df) > TAIL_DAYS:
        df = df.iloc[-TAIL_DAYS:].copy()

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

    # ✅ POOS memory: импульс был недавно
    df["impulse_recent"] = (
        df["is_impulse"]
        .rolling(IMPULSE_MEMORY_DAYS, min_periods=1)
        .max()
        .fillna(0)
        .astype(int)
    )

    in_pos = False
    watch = False
    cooldown_until = -1

    entry = stop = tp = initial_stop = None
    be_moved = False
    hold = 0
    entry_time = None
    entry_mstate = None
    watch_score = None

    trades = []

    # diagnostics
    diag = {
        "rows": int(len(df)),
        "impulses": int(df["is_impulse"].sum()),
        "impulse_recent_days": int(df["impulse_recent"].sum())
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
            lo, hi = float(row["low"]), float(row["high"])

            if (not be_moved) and hi >= entry * (1.0 + MOVE_BE_PCT):
                stop = entry
                be_moved = True

            if lo <= stop:
                exit_px = stop
                trades.append({
                    "ticker": ticker, "entry_time": entry_time, "exit_time": t,
                    "entry": float(entry), "exit": float(exit_px),
                    "stop_init": float(initial_stop), "tp": float(tp),
                    "exit_reason": "STOP", "hold_days": int(hold),
                    "be_moved": bool(be_moved), "score": float(watch_score) if watch_score else np.nan,
                    "entry_market_state": entry_mstate
                })
                in_pos = False
                cooldown_until = i + COOLOFF_DAYS
                continue

            if hi >= tp:
                exit_px = tp
                trades.append({
                    "ticker": ticker, "entry_time": entry_time, "exit_time": t,
                    "entry": float(entry), "exit": float(exit_px),
                    "stop_init": float(initial_stop), "tp": float(tp),
                    "exit_reason": "TP", "hold_days": int(hold),
                    "be_moved": bool(be_moved), "score": float(watch_score) if watch_score else np.nan,
                    "entry_market_state": entry_mstate
                })
                in_pos = False
                cooldown_until = i + COOLOFF_DAYS
                continue

            if hold >= MAX_HOLD_DAYS:
                exit_px = float(row["close"])
                trades.append({
                    "ticker": ticker, "entry_time": entry_time, "exit_time": t,
                    "entry": float(entry), "exit": float(exit_px),
                    "stop_init": float(initial_stop), "tp": float(tp),
                    "exit_reason": "TIME", "hold_days": int(hold),
                    "be_moved": bool(be_moved), "score": float(watch_score) if watch_score else np.nan,
                    "entry_market_state": entry_mstate
                })
                in_pos = False
                cooldown_until = i + COOLOFF_DAYS
                continue

        if not in_pos:
            if not watch:
                # ✅ ВАЖНО: watch можно начать, если импульс был недавно, а рынок сейчас risk_on
                if (mstate in ALLOW_ENTRY_STATES) and bool(row["impulse_recent"]):
                    vol_ratio = float(row["volume"] / row["vol_ma"]) if row["vol_ma"] and not pd.isna(row["vol_ma"]) else 0.0
                    watch_score = float(row["imp_ret"]) * vol_ratio
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
                    tp = entry + TP_ATR * float(a)
                    in_pos = True
                    watch = False
                    be_moved = False
                    hold = 0

    return pd.DataFrame(trades), diag

def simulate_portfolio(trades: pd.DataFrame, market_state: pd.DataFrame):
    if trades.empty:
        return pd.DataFrame(columns=["time","equity"]), pd.DataFrame()

    trades = trades.copy()
    trades["entry_time"] = pd.to_datetime(trades["entry_time"], utc=True)
    trades["exit_time"]  = pd.to_datetime(trades["exit_time"], utc=True)

    ms = market_state.copy()
    ms_map = dict(zip(ms["d"], ms["market_state"]))

    tickers = sorted(trades["ticker"].unique().tolist())
    start = trades["entry_time"].min().normalize()
    end = trades["exit_time"].max().normalize()

    close_map = {}
    for t in tickers:
        try:
            df = read_parquet(f"{DATA_PREFIX}{t}.parquet")
            df["date"] = pd.to_datetime(df["date"], utc=True)
            df["d"] = df["date"].dt.normalize()
            df = df[(df["d"] >= start) & (df["d"] <= end)]
            close_map.update({(t, d): float(c) for d, c in zip(df["d"], df["close"])})
        except Exception as e:
            print("close load err", t, e)

    trades["score"] = pd.to_numeric(trades.get("score", 0.0), errors="coerce").fillna(0.0)
    trades = trades.sort_values(["entry_time","score"], ascending=[True, False]).reset_index(drop=True)
    trades["trade_id"] = np.arange(len(trades))

    equity = 1.0
    events = []
    open_pos = {}

    entry_times = trades["entry_time"].sort_values().unique()

    def close_positions_by_time(tstamp):
        nonlocal equity
        to_close = [tid for tid, pos in open_pos.items() if pos["close_time"] <= tstamp]
        for tid in sorted(to_close, key=lambda tid: open_pos[tid]["close_time"]):
            pos = open_pos.pop(tid)
            equity *= (1.0 + pos["pos_frac"] * pos["ret"])
            events.append((pos["close_time"], equity, "CLOSE", pos["ticker"], tid, pos["close_reason"]))

    def force_exit_on_day(d_norm, close_time):
        nonlocal equity
        if ms_map.get(d_norm, "neutral") not in FORCE_EXIT_STATES:
            return
        to_close = list(open_pos.keys())
        for tid in sorted(to_close, key=lambda x: open_pos[x]["close_time"]):
            pos = open_pos.pop(tid)
            px_close = close_map.get((pos["ticker"], d_norm), pos["planned_exit_px"])
            entry_adj, exit_adj = apply_costs(pos["entry_px"], float(px_close))
            ret = (exit_adj / entry_adj) - 1.0
            equity *= (1.0 + pos["pos_frac"] * ret)
            events.append((close_time, equity, "FORCE_EXIT", pos["ticker"], tid, "MARKET_RISK_OFF"))

    for et in entry_times:
        close_positions_by_time(et)
        d_et = pd.to_datetime(et, utc=True).normalize()
        force_exit_on_day(d_et, et)

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
            planned_exit_px = float(r["exit"])
            entry_adj, exit_adj = apply_costs(entry_px, planned_exit_px)
            planned_ret = (exit_adj / entry_adj) - 1.0

            open_pos[int(r["trade_id"])] = {
                "ticker": r["ticker"],
                "entry_time": r["entry_time"],
                "close_time": r["exit_time"],
                "planned_exit_px": planned_exit_px,
                "entry_px": entry_px,
                "ret": planned_ret,
                "pos_frac": POS_FRACTION,
                "close_reason": r["exit_reason"],
            }
            events.append((r["entry_time"], equity, "OPEN", r["ticker"], int(r["trade_id"]), ""))

        force_exit_on_day(d_et, et)

    close_positions_by_time(pd.Timestamp.max.tz_localize("UTC"))

    events_df = pd.DataFrame(events, columns=["time","equity","event","ticker","trade_id","reason"])
    equity_df = events_df[events_df["event"].isin(["CLOSE","FORCE_EXIT"])][["time","equity"]].reset_index(drop=True)
    return equity_df, events_df

def main():
    ms = load_market_state()
    ms_map = dict(zip(ms["d"], ms["market_state"]))

    tickers = [t for t in list_tickers(DATA_PREFIX)]
    print("Tickers:", len(tickers))

    all_trades = []
    diags = []

    for i, t in enumerate(tickers, 1):
        try:
            df = read_parquet(f"{DATA_PREFIX}{t}.parquet")
            if df is None or df.empty:
                continue
            tr, diag = generate_trades(df, t, ms_map)
            diag["ticker"] = t
            diags.append(diag)
            if not tr.empty:
                all_trades.append(tr)
            if i % 50 == 0:
                print(f"Processed {i}/{len(tickers)}")
        except Exception as e:
            print("Error:", t, e)

    trades = pd.concat(all_trades, ignore_index=True) if all_trades else pd.DataFrame()
    diag_df = pd.DataFrame(diags)

    equity_df, events_df = simulate_portfolio(trades, ms)

    stats = {
        "mode": "POOS DAILY + MARKET ENGINE GATING (impulse memory fixed)",
        "tickers_tested": int(len(tickers)),
        "trades": int(len(trades)) if not trades.empty else 0,
        "final_equity": float(equity_df["equity"].iloc[-1]) if not equity_df.empty else 1.0,
        "impulse_memory_days": int(IMPULSE_MEMORY_DAYS),
        "allow_entry_states": ALLOW_ENTRY_STATES,
        "force_exit_states": FORCE_EXIT_STATES,
        "imp_lookback_days": int(IMP_LOOKBACK_DAYS),
        "imp_min_ret": float(IMP_MIN_RET),
        "vol_mult": float(VOL_MULT),
        "market_state_counts": ms["market_state"].value_counts().to_dict(),
        "impulse_diag_sum": {
            "impulses_total": int(diag_df["impulses"].sum()) if not diag_df.empty else 0,
            "impulse_recent_days_total": int(diag_df["impulse_recent_days"].sum()) if not diag_df.empty else 0,
        }
    }

    put_text(f"{RESULT_PREFIX}/stats.json", json.dumps(stats, indent=2))
    put_text(f"{RESULT_PREFIX}/trades.csv", trades.to_csv(index=False) if not trades.empty else "ticker,entry_time,exit_time,entry,exit,stop_init,tp,exit_reason,hold_days,be_moved,score,entry_market_state\n")
    put_text(f"{RESULT_PREFIX}/equity.csv", equity_df.to_csv(index=False) if not equity_df.empty else "time,equity\n")
    put_text(f"{RESULT_PREFIX}/events.csv", events_df.to_csv(index=False) if not events_df.empty else "time,equity,event,ticker,trade_id,reason\n")
    put_text(f"{RESULT_PREFIX}/diag.csv", diag_df.to_csv(index=False) if not diag_df.empty else "ticker,rows,impulses,impulse_recent_days\n")

    plt.figure()
    if not equity_df.empty:
        plt.plot(pd.to_datetime(equity_df["time"]), equity_df["equity"])
        plt.xticks(rotation=30, ha="right")
        plt.title("POOS Daily Equity (Market-Gated, impulse-memory)")
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
