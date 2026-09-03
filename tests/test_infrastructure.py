"""The three infrastructure containers must be reachable from the host before
we write clients against them.

Unlike the rest of the suite these tests need `docker compose up -d rabbitmq
redis minio minio-init`. They read the settings directly: .env holds
host-visible values, so no rewriting of hostnames is needed here.
"""

import boto3
import botocore.exceptions
import pytest
import redis as redis_lib
from kombu import Connection

from app.config import get_settings


def _s3():
    settings = get_settings()
    return boto3.client(
        "s3",
        endpoint_url=settings.minio_public_url,
        aws_access_key_id=settings.minio_access_key,
        aws_secret_access_key=settings.minio_secret_key,
        region_name="us-east-1",
    )


def test_redis_answers_ping():
    client = redis_lib.from_url(get_settings().redis_url)

    assert client.ping() is True


def test_rabbitmq_accepts_a_connection():
    """Also proves the credentials are wired: RabbitMQ restricts the default
    `guest` user to loopback, and a published port does not count as one."""
    with Connection(get_settings().rabbitmq_url) as conn:
        conn.connect()

        assert conn.connected


def test_minio_has_both_buckets():
    settings = get_settings()
    names = {b["Name"] for b in _s3().list_buckets()["Buckets"]}

    assert {settings.minio_bucket_raw, settings.minio_bucket_staging} <= names


def test_staging_expires_after_seven_days_and_raw_never_does():
    """staging/ holds the per-stage checkpoints; raw/ is kept forever so a
    document can always be re-ingested. Nothing but this rule enforces it.

    Asserting on the whole list, not just membership: `mc ilm rule add` mints
    a fresh rule ID per run, so a non-idempotent init would stack duplicates
    here rather than fail.
    """
    settings = get_settings()
    rules = _s3().get_bucket_lifecycle_configuration(Bucket=settings.minio_bucket_staging)["Rules"]

    assert [r["Expiration"]["Days"] for r in rules] == [7]

    with pytest.raises(botocore.exceptions.ClientError, match="NoSuchLifecycleConfiguration"):
        _s3().get_bucket_lifecycle_configuration(Bucket=settings.minio_bucket_raw)
