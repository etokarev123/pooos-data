import os
import io
import time
import pandas as pd
import yfinance as yf
import boto3
from botocore.config import Config
from dotenv import load_dotenv

load_dotenv()

# ---- ENV ----
R2_ENDPOINT = os.environ["R2_ENDPOINT"]
R2_KEY = os.environ["R2_ACCESS_KEY_ID"]
R2_SECRET = os.environ["R2_SECRET_ACCESS_KEY"]
R2_BUCKET = os.environ["R2_BUCKET"]

PERIOD = os.getenv("PERIOD", "730d")
INTERVAL = os.getenv("INTERVAL", "1h")
BATCH_SIZE = int(os.getenv("BATCH_SIZE", "20"))
SLEEP = int(os.getenv("SLEEP_SEC", "5"))

# ---- R2 Client ----
s3 = boto3.client(
    "s3",
    endpoint_url=R2_ENDPOINT,
    aws_access_key_id=R2_KEY,
    aws_secret_access_key=R2_SECRET,
    region_name="auto",
    config=Config(signature_version="s3v4"),
)

# ---- Read tickers ----
with open("tickers.txt") as f:
    tickers = [x.strip() for x in f if x.strip()]

print("Tickers:", len(tickers))

# ---- helper ----
def chunks(lst, n):
    for i in range(0, len(lst), n):
        yield lst[i:i+n]

def upload_df(ticker, df):
    buf = io.BytesIO()
    df.to_parquet(buf, index=False)
    key = f"yahoo/1h/{ticker}.parquet"
    s3.put_object(Bucket=R2_BUCKET, Key=key, Body=buf.getvalue())

# ---- download loop ----
for batch in chunks(tickers, BATCH_SIZE):
    print("Downloading:", batch)

    try:
        data = yf.download(
            tickers=" ".join(batch),
            period=PERIOD,
            interval=INTERVAL,
            group_by="ticker",
            threads=False,
            progress=False,
        )
    except Exception as e:
        print("Yahoo error:", e)
        time.sleep(10)
        continue

    for ticker in batch:
        try:
            if ticker not in data.columns.get_level_values(0):
                print("Missing:", ticker)
                continue

            df = data[ticker].reset_index()
            df = df.rename(columns={
                "Datetime": "date",
                "Open": "open",
                "High": "high",
                "Low": "low",
                "Close": "close",
                "Volume": "volume"
            })

            df = df[["date", "open", "high", "low", "close", "volume"]].dropna()
            df["date"] = pd.to_datetime(df["date"], utc=True)

            if len(df) < 100:
                print("Too small:", ticker)
                continue

            upload_df(ticker, df)
            print("Uploaded:", ticker, len(df))

        except Exception as e:
            print("Error", ticker, e)

    time.sleep(SLEEP)

print("DONE")
