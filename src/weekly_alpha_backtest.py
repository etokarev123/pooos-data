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

# ---------------- R2 ----------------
R2_ENDPOINT = os.environ["R2_ENDPOINT"]
R2_KEY = os.environ["R2_ACCESS_KEY_ID"]
R2_SECRET = os.environ["R2_SECRET_ACCESS_KEY"]
R2_BUCKET = os.environ["R2_BUCKET"]

DATA_PREFIX = os.getenv("DATA_PREFIX", "yahoo/1d/")

# ---------------- Strategy params ----------------
N_LONG = int(os.getenv("N_LONG", "20"))
N_SHORT = int(os.getenv("N_SHORT", "20"))
GROSS_LEVERAGE = float(os.getenv("GROSS_LEVERAGE", "2.0"))  # 2.0 => 100% long + 100% short, net ~0
COST_BPS = float(os.getenv("COST_BPS", "10"))               # cost per 1.0 turnover (bps)
TAIL_WEEKS = int(os.getenv("TAIL_WEEKS", "0"))              # 0 = all available; else last N weeks

MIN_WEEKS_HISTORY = int(os.getenv("MIN_WEEKS_HISTORY", "60"))  # require at least this many weekly bars/ticker

RESULT_PREFIX = os.getenv("RESULT_PREFIX", "results/alpha_ls_weekly_v1")

# Factor lookbacks (weekly)
LB_4W = int(os.getenv("LB_4W", "4"))
LB_12W = int(os.getenv("LB_12W", "12"))
LB_VOL = int(os.getenv("LB_VOL", "12"))
LB_VOLU = int(os.getenv("LB_VOLU", "12"))

# "POOS-like" impulse factor (weekly): big move + volume spike
POOS_LB = int(os.getenv("POOS_LB", "8"))
POOS_MIN_RET = float(os.getenv("POOS_MIN_RET", "0.20"))  # +20% over 8 weeks
POOS_VOL_MULT = float(os.getenv("POOS_VOL_MULT", "1.5")) # volume this week > 1.5x avg(12w)

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
            k = obj["Key"]
            if k.endswith(".parquet"):
                out.append(k.split("/")[-1].replace(".parquet", ""))
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

def to_weekly(df_daily: pd.DataFrame) -> pd.DataFrame:
    """
    Build weekly bars from daily (RTH) bars.
    Week = Fri close (W-FRI).
    """
    df = df_daily.copy()
    df["date"] = pd.to_datetime(df["date"], utc=True)
    df = df.sort_values("date").dropna(subset=["open","high","low","close","volume"])
    df = df.set_index("date")

    # Resample to W-FRI (end of week Friday)
    o = df["open"].resample("W-FRI").first()
    h = df["high"].resample("W-FRI").max()
    l = df["low"].resample("W-FRI").min()
    c = df["close"].resample("W-FRI").last()
    v = df["volume"].resample("W-FRI").sum()

    w = pd.DataFrame({"open": o, "high": h, "low": l, "close": c, "volume": v}).dropna()
    w = w[w["volume"] > 0]
    w = w.reset_index().rename(columns={"date": "week"})
    return w

def zscore_cross_section(x: pd.Series) -> pd.Series:
    x = pd.to_numeric(x, errors="coerce")
    m = x.mean()
    s = x.std(ddof=0)
    if s == 0 or np.isnan(s):
        return x * 0.0
    return (x - m) / s

def build_weekly_panel(tickers: list[str]) -> pd.DataFrame:
    """
    Returns a panel with columns:
    week, ticker, close, volume, ret_1w, mom_4w, mom_12w, vol_12w, volu_z, poos_flag
    """
    rows = []
    for i, t in enumerate(tickers, 1):
        try:
            df = read_parquet(f"{DATA_PREFIX}{t}.parquet")
            if df is None or df.empty:
                continue

            need = {"date","open","high","low","close","volume"}
            if not need.issubset(set(df.columns)):
                continue

            w = to_weekly(df)
            if len(w) < MIN_WEEKS_HISTORY:
                continue

            w["ticker"] = t
            rows.append(w)

            if i % 50 == 0:
                print(f"Loaded {i}/{len(tickers)} tickers")
        except Exception as e:
            print("Error loading", t, ":", e)

    if not rows:
        return pd.DataFrame()

    panel = pd.concat(rows, ignore_index=True)
    panel = panel.sort_values(["ticker","week"]).reset_index(drop=True)

    # 1w returns by ticker
    panel["ret_1w"] = panel.groupby("ticker")["close"].pct_change(1)

    # momentum
    panel["mom_4w"] = panel.groupby("ticker")["close"].pct_change(LB_4W)
    panel["mom_12w"] = panel.groupby("ticker")["close"].pct_change(LB_12W)

    # volatility (std of weekly returns)
    panel["vol_12w"] = panel.groupby("ticker")["ret_1w"].rolling(LB_VOL, min_periods=LB_VOL).std(ddof=0).reset_index(level=0, drop=True)

    # volume z-ish: current volume relative to avg
    vol_ma = panel.groupby("ticker")["volume"].rolling(LB_VOLU, min_periods=LB_VOLU).mean().reset_index(level=0, drop=True)
    panel["vol_ratio"] = panel["volume"] / vol_ma

    # POOS-like impulse flag: big move in POOS_LB weeks + volume spike this week
    panel["poos_ret"] = panel.groupby("ticker")["close"].pct_change(POOS_LB)
    panel["poos_flag"] = ((panel["poos_ret"] >= POOS_MIN_RET) & (panel["vol_ratio"] >= POOS_VOL_MULT)).astype(int)

    return panel

def run_backtest(panel: pd.DataFrame):
    """
    Weekly rebalance:
    - At week t close, compute score using info up to t (no lookahead)
    - Hold from t -> t+1 close (using ret_1w at t+1)
    Portfolio return at week t+1 = sum(weights_t * ret_1w_{t+1})
    Costs: COST_BPS * turnover, applied at rebalance time t (on equity)
    """
    panel = panel.copy()
    panel["week"] = pd.to_datetime(panel["week"], utc=True)

    # Drop rows without features
    feats = ["mom_4w","mom_12w","vol_12w","vol_ratio","poos_flag"]
    panel = panel.dropna(subset=["ret_1w"] + feats)

    # Optional tail window
    weeks = sorted(panel["week"].unique())
    if TAIL_WEEKS > 0 and len(weeks) > TAIL_WEEKS:
        keep = set(weeks[-TAIL_WEEKS:])
        panel = panel[panel["week"].isin(keep)].copy()
        weeks = sorted(panel["week"].unique())

    # Build per-week cross-sectional scores (z-scored)
    # Stronger momentum good, higher vol_ratio good, poos_flag good, lower vol_12w mildly good (risk control)
    # (We keep it simple + robust)
    def score_week(dfw: pd.DataFrame) -> pd.DataFrame:
        dfw = dfw.copy()
        dfw["z_m4"] = zscore_cross_section(dfw["mom_4w"])
        dfw["z_m12"] = zscore_cross_section(dfw["mom_12w"])
        dfw["z_volu"] = zscore_cross_section(np.log1p(dfw["vol_ratio"].clip(lower=0)))
        dfw["z_risk"] = -zscore_cross_section(dfw["vol_12w"])  # prefer lower vol a bit
        dfw["z_poos"] = zscore_cross_section(dfw["poos_flag"].astype(float))

        # weights of factors (simple starting point)
        dfw["score"] = (
            0.35 * dfw["z_m4"] +
            0.35 * dfw["z_m12"] +
            0.15 * dfw["z_volu"] +
            0.10 * dfw["z_risk"] +
            0.05 * dfw["z_poos"]
        )
        return dfw

    panel = panel.groupby("week", group_keys=False).apply(score_week)

    # Weights: equal-weight long & short, gross = GROSS_LEVERAGE
    long_w = (GROSS_LEVERAGE / 2.0) / N_LONG
    short_w = -(GROSS_LEVERAGE / 2.0) / N_SHORT

    equity = 1.0
    curve = []
    holdings_rows = []

    prev_w = {}  # ticker -> weight

    # For return at week t+1, we need next week's ret_1w.
    # We'll form weights at week t, apply to week t+1 returns.
    weeks = sorted(panel["week"].unique())
    if len(weeks) < 2:
        return pd.DataFrame(), pd.DataFrame(), {}

    for idx in range(len(weeks) - 1):
        w = weeks[idx]
        w_next = weeks[idx + 1]

        dfw = panel[panel["week"] == w].copy()
        dfn = panel[panel["week"] == w_next][["ticker","ret_1w"]].copy()

        if dfw.empty or dfn.empty:
            continue

        dfw = dfw.sort_values("score", ascending=False)
        longs = dfw.head(N_LONG)["ticker"].tolist()
        shorts = dfw.tail(N_SHORT)["ticker"].tolist()

        w_map = {t: long_w for t in longs}
        w_map.update({t: short_w for t in shorts})

        # turnover = sum(|w_new - w_old|) / 2 (standard)
        all_names = set(prev_w.keys()) | set(w_map.keys())
        turnover = 0.0
        for t in all_names:
            turnover += abs(w_map.get(t, 0.0) - prev_w.get(t, 0.0))
        turnover *= 0.5

        # apply costs at rebalance (bps on turnover)
        cost = turnover * (COST_BPS / 10000.0)
        equity *= (1.0 - cost)

        # realized portfolio return over next week
        ret_next = dfn.set_index("ticker")["ret_1w"].to_dict()
        port_ret = 0.0
        for t, wt in w_map.items():
            r = ret_next.get(t, 0.0)
            port_ret += wt * r

        equity *= (1.0 + port_ret)
        curve.append((w_next, equity, port_ret, turnover, cost))

        # log holdings at rebalance week
        holdings_rows.extend([(w, t, w_map[t]) for t in sorted(w_map.keys())])

        prev_w = w_map

    equity_df = pd.DataFrame(curve, columns=["time","equity","port_ret","turnover","cost"])
    holdings_df = pd.DataFrame(holdings_rows, columns=["rebalance_week","ticker","weight"])

    stats = {
        "mode": "Weekly Alpha Long/Short Market Neutral",
        "tickers_in_panel": int(panel["ticker"].nunique()),
        "weeks": int(equity_df.shape[0]),
        "N_LONG": int(N_LONG),
        "N_SHORT": int(N_SHORT),
        "gross_leverage": float(GROSS_LEVERAGE),
        "cost_bps_per_turnover": float(COST_BPS),
        "final_equity": float(equity_df["equity"].iloc[-1]) if not equity_df.empty else 1.0,
        "avg_weekly_ret": float(equity_df["port_ret"].mean()) if not equity_df.empty else 0.0,
        "std_weekly_ret": float(equity_df["port_ret"].std(ddof=0)) if not equity_df.empty else 0.0,
        "avg_turnover": float(equity_df["turnover"].mean()) if not equity_df.empty else 0.0,
        "tail_weeks": int(TAIL_WEEKS),
        "factors": {
            "mom_4w": LB_4W,
            "mom_12w": LB_12W,
            "vol_12w": LB_VOL,
            "volu_ma": LB_VOLU,
            "poos_lb": POOS_LB,
            "poos_min_ret": POOS_MIN_RET,
            "poos_vol_mult": POOS_VOL_MULT
        }
    }

    # Add simple Sharpe-like (weekly, not annualized/then annualized)
    if not equity_df.empty and stats["std_weekly_ret"] > 0:
        weekly_sharpe = stats["avg_weekly_ret"] / stats["std_weekly_ret"]
        stats["weekly_sharpe"] = float(weekly_sharpe)
        stats["annualized_sharpe_approx"] = float(weekly_sharpe * math.sqrt(52))
    else:
        stats["weekly_sharpe"] = 0.0
        stats["annualized_sharpe_approx"] = 0.0

    return equity_df, holdings_df, stats

def main():
    tickers = list_tickers(DATA_PREFIX)
    print("Daily tickers in R2:", len(tickers))
    if not tickers:
        raise SystemExit("No daily data found in R2. Ensure yahoo/1d exists.")

    panel = build_weekly_panel(tickers)
    if panel.empty:
        raise SystemExit("Weekly panel is empty. Not enough data or schema mismatch.")

    print("Panel tickers:", panel["ticker"].nunique(), "rows:", len(panel))

    equity_df, holdings_df, stats = run_backtest(panel)

    # Save to R2
    put_text(f"{RESULT_PREFIX}/stats.json", json.dumps(stats, indent=2))
    put_text(f"{RESULT_PREFIX}/equity.csv", equity_df.to_csv(index=False) if not equity_df.empty else "time,equity,port_ret,turnover,cost\n")
    put_text(f"{RESULT_PREFIX}/holdings.csv", holdings_df.to_csv(index=False) if not holdings_df.empty else "rebalance_week,ticker,weight\n")

    # Plot
    plt.figure()
    if not equity_df.empty:
        plt.plot(pd.to_datetime(equity_df["time"]), equity_df["equity"])
        plt.xticks(rotation=30, ha="right")
        plt.title("Weekly Alpha L/S Equity (Market Neutral)")
        plt.tight_layout()
    else:
        plt.text(0.5, 0.5, "No data", ha="center", va="center")
        plt.axis("off")

    buf = io.BytesIO()
    plt.savefig(buf, format="png", dpi=160)
    plt.close()
    put_bytes(f"{RESULT_PREFIX}/equity.png", buf.getvalue(), content_type="image/png")

    print("Saved to R2:", RESULT_PREFIX)
    print("Stats:", stats)

if __name__ == "__main__":
    main()
