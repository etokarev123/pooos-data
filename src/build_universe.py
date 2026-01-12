import os
import boto3
from dotenv import load_dotenv

load_dotenv()

R2_ENDPOINT = os.environ["R2_ENDPOINT"]
R2_KEY = os.environ["R2_ACCESS_KEY_ID"]
R2_SECRET = os.environ["R2_SECRET_ACCESS_KEY"]
R2_BUCKET = os.environ["R2_BUCKET"]

s3 = boto3.client(
    "s3",
    endpoint_url=R2_ENDPOINT,
    aws_access_key_id=R2_KEY,
    aws_secret_access_key=R2_SECRET,
    region_name="auto",
)

resp = s3.list_objects_v2(Bucket=R2_BUCKET, Prefix="yahoo/1h/")

tickers = []
for obj in resp.get("Contents", []):
    key = obj["Key"]
    if key.endswith(".parquet"):
        t = key.split("/")[-1].replace(".parquet", "")
        tickers.append(t)

tickers = sorted(set(tickers))

# Save to local file
with open("universe.txt", "w") as f:
    for t in tickers:
        f.write(t + "\n")

# Upload to R2
s3.put_object(
    Bucket=R2_BUCKET,
    Key="universe.txt",
    Body="\n".join(tickers).encode(),
)

print("Universe size:", len(tickers))
print(tickers[:20])
