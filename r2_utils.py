from __future__ import annotations

"""
Cloudflare R2 upload utility.
Reads credentials from environment variables.

Required:
  R2_ACCOUNT_ID
  R2_ACCESS_KEY_ID
  R2_SECRET_ACCESS_KEY
  R2_BUCKET_NAME  (or legacy R2_BUCKET)
  R2_PUBLIC_BASE_URL
"""

import os


def upload_to_r2(local_path: str, r2_key: str) -> str:
    import boto3
    from botocore.config import Config

    account_id = os.environ["R2_ACCOUNT_ID"]
    access_key = os.environ["R2_ACCESS_KEY_ID"]
    secret_key = os.environ["R2_SECRET_ACCESS_KEY"]
    bucket = os.getenv("R2_BUCKET_NAME") or os.getenv("R2_BUCKET")
    if not bucket:
        raise RuntimeError("Missing R2_BUCKET_NAME or R2_BUCKET environment variable.")
    public_base = os.environ["R2_PUBLIC_BASE_URL"].rstrip("/")

    endpoint = f"https://{account_id}.r2.cloudflarestorage.com"

    s3 = boto3.client(
        "s3",
        endpoint_url=endpoint,
        aws_access_key_id=access_key,
        aws_secret_access_key=secret_key,
        config=Config(signature_version="s3v4"),
        region_name="auto",
    )

    content_type = "video/mp4"
    if local_path.endswith(".webm"):
        content_type = "video/webm"

    with open(local_path, "rb") as f:
        s3.put_object(
            Bucket=bucket,
            Key=r2_key,
            Body=f,
            ContentType=content_type,
            CacheControl="public, max-age=31536000",
        )

    return f"{public_base}/{r2_key}"
