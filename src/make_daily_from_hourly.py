import os
import io
import pandas as pd
import boto3
from botocore.config import Config
from dotenv import load_dotenv

load_dotenv()

R2_ENDPOINT = os.environ["R2_ENDPOINT"]
R2_KEY = os.environ["R2_ACCESS_KEY_ID"]
R2_SECRET = os.environ["R2_SECRET_ACCESS_KEY"]
R2_BUCKET = os.environ["R2_BUCKET"]

SRC_PREFIX = os.getenv("SRC_PREFIX", "yahoo/1h/")
DST_PREFIX = os.getenv("DST_PREFIX", "yahoo/1d/")

# Regular Trading Hours (New York)
RTH_START = os.getenv("RTH_START", "09:30")
RTH_END = os.getenv("RTH_END", "16:00")

MAX_TICKERS = int(os.getenv("MAX_TICKERS", "0"))  # 0 = all

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

def put_parquet(key: str, df: pd.DataFrame):
    buf = io.BytesIO()
    df.to_parquet(buf, index=False)
    s3.put_object(Bucket=R2_BUCKET, Key=key, Body=buf.getvalue())

def make_daily(df_1h: pd.DataFrame) -> pd.DataFrame:
    # Expect columns: date (utc), open, high, low, close, volume
    df = df_1h.copy()
    df["date"] = pd.to_datetime(df["date"], utc=True)
    df = df.sort_values("date")

    # Convert to NY time and filter RTH
    df["ny"] = df["date"].dt.tz_convert("America/New_York")
    df = df.set_index("ny")

    # Keep only weekdays and RTH window
    df = df[df.index.dayofweek < 5]
    df = df.between_time(RTH_START, RTH_END, inclusive="both")

    if df.empty:
        return pd.DataFrame()

    df["day"] = df.index.date

    g = df.groupby("day", sort=True)
    daily = pd.DataFrame({
        "date": pd.to_datetime(list(g.groups.keys())).tz_localize("America/New_York").tz_convert("UTC"),
        "open": g["open"].first().values,
        "high": g["high"].max().values,
        "low": g["low"].min().values,
        "close": g["close"].last().values,
        "volume": g["volume"].sum().values,
    })

    daily = daily.dropna()
    daily = daily.sort_values("date")

    # Basic sanity
    daily = daily[daily["volume"] > 0]
    return daily

def main():
    tickers = list_tickers(SRC_PREFIX)
    if MAX_TICKERS > 0:
        tickers = tickers[:MAX_TICKERS]

    print("Hourly tickers:", len(tickers))
    made = 0
    skipped = 0

    for i, t in enumerate(tickers, 1):
        try:
            key = f"{SRC_PREFIX}{t}.parquet"
            df_1h = read_parquet(key)
            if df_1h is None or df_1h.empty:
                skipped += 1
                continue

            need = {"date", "open", "high", "low", "close", "volume"}
            if not need.issubset(set(df_1h.columns)):
                print("Bad schema:", t, "cols=", list(df_1h.columns))
                skipped += 1
                continue

            daily = make_daily(df_1h)
            if daily.empty or len(daily) < 200:
                print("Too small daily:", t, "rows=", len(daily))
                skipped += 1
                continue

            out_key = f"{DST_PREFIX}{t}.parquet"
            put_parquet(out_key, daily)
            made += 1

            if i % 25 == 0:
                print(f"Processed {i}/{len(tickers)} | daily_ok={made} skipped={skipped}")

        except Exception as e:
            print("Error:", t, e)
            skipped += 1

    print("DONE daily build. daily_ok=", made, "skipped=", skipped)

if __name__ == "__main__":
    main()
