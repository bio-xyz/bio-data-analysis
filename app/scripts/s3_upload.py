#!/usr/bin/env python3
"""
S3 Upload Script - Uploads files or directories to S3 with streaming support.
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


def upload_file(s3_client, local_path, bucket, s3_key):
    """Upload a single file to S3 with streaming."""
    try:
        file_size = os.path.getsize(local_path)
        print(f"Uploading {local_path} to s3://{bucket}/{s3_key} ({file_size} bytes)")

        with open(local_path, "rb") as file_data:
            s3_client.upload_fileobj(file_data, bucket, s3_key)

        print(f"✓ Uploaded {s3_key}")
        return True
    except ClientError as e:
        print(f"✗ Error uploading {local_path}: {e}", file=sys.stderr)
        return False
    except Exception as e:
        print(f"✗ Unexpected error uploading {local_path}: {e}", file=sys.stderr)
        return False


def upload_directory(s3_client, local_dir, bucket, s3_prefix):
    """Recursively upload a directory to S3."""
    local_path = Path(local_dir)

    if not local_path.is_dir():
        print(f"Error: {local_dir} is not a directory", file=sys.stderr)
        return False

    success_count = 0
    error_count = 0

    for root, dirs, files in os.walk(local_dir):
        for filename in files:
            local_file = os.path.join(root, filename)

            relative_path = os.path.relpath(local_file, local_dir)
            s3_key = os.path.join(s3_prefix, relative_path).replace("\\", "/")

            if upload_file(s3_client, local_file, bucket, s3_key):
                success_count += 1
            else:
                error_count += 1

    print(f"\nUpload complete: {success_count} succeeded, {error_count} failed")
    return error_count == 0


def main():
    parser = argparse.ArgumentParser(
        description="Upload files or directories to S3 with streaming support. "
        "AWS credentials must be provided via AWS_ACCESS_KEY_ID and AWS_SECRET_ACCESS_KEY environment variables."
    )
    parser.add_argument("local_path", help="Local file or directory path to upload")
    parser.add_argument("bucket", help="S3 bucket name")
    parser.add_argument("s3_path", help="S3 path (key prefix)")
    parser.add_argument("--endpoint", help="S3 endpoint URL (optional)", default=None)

    args = parser.parse_args()

    if not os.path.exists(args.local_path):
        print(f"Error: Local path '{args.local_path}' does not exist", file=sys.stderr)
        sys.exit(1)

    s3_client = create_s3_client(args.endpoint)

    local_path = Path(args.local_path)

    if local_path.is_file():
        success = upload_file(s3_client, args.local_path, args.bucket, args.s3_path)
        sys.exit(0 if success else 1)
    elif local_path.is_dir():
        success = upload_directory(
            s3_client, args.local_path, args.bucket, args.s3_path
        )
        sys.exit(0 if success else 1)
    else:
        print(
            f"Error: '{args.local_path}' is neither a file nor directory",
            file=sys.stderr,
        )
        sys.exit(1)


if __name__ == "__main__":
    main()
