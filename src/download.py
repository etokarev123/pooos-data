import os
import io
import time
import pandas as pd
import yfinance as yf
import boto3
from botocore.config import Config
from dotenv import load_dotenv

load_dotenv()

# ---------- ENV ----------
R2_ENDPOINT = os.environ["R2_ENDPOINT"]
R2_KEY = os.environ["R2_ACCESS_KEY_ID"]
R2_SECRET = os.environ["R2_SECRET_ACCESS_KEY"]
R2_BUCKET = os.environ["R2_BUCKET"]

PERIOD = os.getenv("PERIOD", "730d")
INTERVAL = os.getenv("INTERVAL", "1h")
BATCH_SIZE = int(os.getenv("BATCH_SIZE", "20"))
SLEEP_SEC = int(os.getenv("SLEEP_SEC", "5"))
RETRIES = int(os.getenv("RETRIES", "3"))

# If RESUME=1, script checks if object exists in R2 and skips it
RESUME = os.getenv("RESUME", "1") == "1"

# ---------- R2 Client ----------
s3 = boto3.client(
    "s3",
    endpoint_url=R2_ENDPOINT,
    aws_access_key_id=R2_KEY,
    aws_secret_access_key=R2_SECRET,
    region_name="auto",
    config=Config(signature_version="s3v4", retries={"max_attempts": 10, "mode": "standard"}),
)

def chunks(lst, n):
    for i in range(0, len(lst), n):
        yield lst[i:i+n]

def object_exists(key: str) -> bool:
    try:
        s3.head_object(Bucket=R2_BUCKET, Key=key)
        return True
    except Exception:
        return False

def upload_df(ticker: str, df: pd.DataFrame):
    buf = io.BytesIO()
    df.to_parquet(buf, index=False)
    key = f"yahoo/1h/{ticker}.parquet"
    s3.put_object(Bucket=R2_BUCKET, Key=key, Body=buf.getvalue())

# ---------- Read tickers ----------
with open("tickers.txt", "r", encoding="utf-8") as f:
    tickers = [x.strip() for x in f if x.strip() and not x.startswith("#")]

print("Tickers:", len(tickers))
print("PERIOD:", PERIOD, "INTERVAL:", INTERVAL, "BATCH_SIZE:", BATCH_SIZE, "RESUME:", RESUME)

# ---------- Download loop ----------
for batch in chunks(tickers, BATCH_SIZE):
    print("Downloading batch:", batch)

    # optional resume: skip tickers that already exist in R2
    if RESUME:
        batch = [t for t in batch if not object_exists(f"yahoo/1h/{t}.parquet")]
        if not batch:
            print("All tickers in this batch already exist. Skipping.")
            continue
        print("After RESUME filter:", batch)

    data = None
    last_err = None
    for attempt in range(1, RETRIES + 1):
        try:
            data = yf.download(
                tickers=" ".join(batch),
                period=PERIOD,
                interval=INTERVAL,
                group_by="ticker",
                threads=False,
                progress=False,
            )
            break
        except Exception as e:
            last_err = e
            print("Yahoo error attempt", attempt, ":", e)
            time.sleep(2 * attempt)

    if data is None:
        print("Batch failed permanently:", batch, "error:", last_err)
        time.sleep(SLEEP_SEC)
        continue

    # If multiple tickers -> MultiIndex columns. If one ticker -> normal columns.
    multi = isinstance(data.columns, pd.MultiIndex)

    for ticker in batch:
        try:
            # handle missing ticker in returned dataset
            if multi:
                if ticker not in data.columns.get_level_values(0):
                    print("Missing in batch result:", ticker)
                    continue
                df = data[ticker].reset_index()
            else:
                # single ticker case
                df = data.reset_index()

            # Normalize datetime column name
            if "Datetime" in df.columns:
                df = df.rename(columns={"Datetime": "date"})
            elif "Date" in df.columns:
                df = df.rename(columns={"Date": "date"})
            elif "index" in df.columns:
                df = df.rename(columns={"index": "date"})

            # Normalize OHLCV names
            df = df.rename(columns={
                "Open": "open",
                "High": "high",
                "Low": "low",
                "Close": "close",
                "Volume": "volume"
            })

            if "date" not in df.columns:
                print("No date column for:", ticker, "cols:", list(df.columns))
                continue

            need = ["date", "open", "high", "low", "close", "volume"]
            df = df[[c for c in need if c in df.columns]].dropna()

            if df.empty or len(df) < 200:
                print("Too small/empty:", ticker, "rows:", len(df))
                continue

            df["date"] = pd.to_datetime(df["date"], utc=True)

            upload_df(ticker, df)
            print("Uploaded:", ticker, "rows:", len(df))

        except Exception as e:
            print("Error ticker:", ticker, "err:", e)

    time.sleep(SLEEP_SEC)

print("DONE")
