#!/usr/bin/env python3
"""
S3 Download Script - Downloads files or directories from S3 with streaming support.
Accepts credentials as command-line arguments.
"""
import argparse
import os
import sys
from pathlib import Path

import boto3
from botocore.exceptions import ClientError


def create_s3_client(endpoint_url=None):
    """Create and return an S3 client using environment variables."""
    access_key = os.environ.get("AWS_ACCESS_KEY_ID")
    secret_key = os.environ.get("AWS_SECRET_ACCESS_KEY")

    if not access_key or not secret_key:
        print(
            "Error: AWS credentials not found in environment variables", file=sys.stderr
        )
        sys.exit(1)

    config_kwargs = {
        "aws_access_key_id": access_key,
        "aws_secret_access_key": secret_key,
    }

    if endpoint_url:
        config_kwargs["endpoint_url"] = endpoint_url

    return boto3.client("s3", **config_kwargs)


def download_file(s3_client, bucket, s3_key, local_path):
    """Download a single file from S3 with streaming."""
    try:
        os.makedirs(os.path.dirname(local_path), exist_ok=True)

        response = s3_client.head_object(Bucket=bucket, Key=s3_key)
        file_size = response.get("ContentLength", 0)

        print(f"Downloading s3://{bucket}/{s3_key} to {local_path} ({file_size} bytes)")

        with open(local_path, "wb") as file_data:
            s3_client.download_fileobj(bucket, s3_key, file_data)

        print(f"✓ Downloaded {s3_key}")
        return True
    except ClientError as e:
        print(f"✗ Error downloading {s3_key}: {e}", file=sys.stderr)
        return False
    except Exception as e:
        print(f"✗ Unexpected error downloading {s3_key}: {e}", file=sys.stderr)
        return False


def is_s3_file(s3_client, bucket, s3_path):
    """Check if S3 path is a file (object) or directory (prefix)."""
    try:
        s3_client.head_object(Bucket=bucket, Key=s3_path)
        return True
    except ClientError as e:
        if e.response["Error"]["Code"] == "404":
            # Object doesn't exist, might be a prefix/directory
            return False
        raise


def download_directory(s3_client, bucket, s3_prefix, local_dir):
    """Recursively download all objects with a given prefix from S3."""
    try:
        if s3_prefix and not s3_prefix.endswith("/"):
            s3_prefix += "/"

        paginator = s3_client.get_paginator("list_objects_v2")
        pages = paginator.paginate(Bucket=bucket, Prefix=s3_prefix)

        success_count = 0
        error_count = 0
        found_objects = False

        for page in pages:
            if "Contents" not in page:
                continue

            found_objects = True
            for obj in page["Contents"]:
                s3_key = obj["Key"]

                if s3_key.endswith("/"):
                    continue

                relative_path = s3_key[len(s3_prefix) :] if s3_prefix else s3_key
                local_file = os.path.join(local_dir, relative_path)

                if download_file(s3_client, bucket, s3_key, local_file):
                    success_count += 1
                else:
                    error_count += 1

        if not found_objects:
            print(f"No objects found with prefix: {s3_prefix}", file=sys.stderr)
            return False

        print(f"\nDownload complete: {success_count} succeeded, {error_count} failed")
        return error_count == 0
    except ClientError as e:
        print(f"Error listing objects: {e}", file=sys.stderr)
        return False
    except Exception as e:
        print(f"Unexpected error: {e}", file=sys.stderr)
        return False


def main():
    parser = argparse.ArgumentParser(
        description="Download files or directories from S3 with streaming support. "
        "AWS credentials must be provided via AWS_ACCESS_KEY_ID and AWS_SECRET_ACCESS_KEY environment variables."
    )
    parser.add_argument("bucket", help="S3 bucket name")
    parser.add_argument("s3_path", help="S3 path (key or prefix)")
    parser.add_argument("local_path", help="Local destination path")
    parser.add_argument("--endpoint", help="S3 endpoint URL (optional)", default=None)

    args = parser.parse_args()

    s3_client = create_s3_client(args.endpoint)

    if is_s3_file(s3_client, args.bucket, args.s3_path):
        print(f"Detected file: {args.s3_path}")
        success = download_file(s3_client, args.bucket, args.s3_path, args.local_path)
        sys.exit(0 if success else 1)
    else:
        print(f"Detected directory/prefix: {args.s3_path}")
        success = download_directory(
            s3_client, args.bucket, args.s3_path, args.local_path
        )
        sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
