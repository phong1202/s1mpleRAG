"""These tests talk to the real MinIO from `docker compose up -d minio`.

There is no mocked S3 here on purpose: the things worth getting wrong --
presigned URL hosts, which errors mean "absent", lifecycle behaviour -- are
exactly the things a mock would answer according to our own assumptions.
"""

import hashlib
import uuid
from urllib.parse import urlparse

import botocore.exceptions
import httpx
import pytest

from app.config import get_settings
from shared.storage import ObjectStore, get_public_store, get_store


@pytest.fixture
def store():
    return get_public_store()


@pytest.fixture
def key(store):
    """A unique staging key, removed afterwards. Without the cleanup every run
    leaves objects that only the 7-day lifecycle would eventually collect."""
    name = f"staging/test-{uuid.uuid4()}/probe.bin"
    yield name
    store.delete(name)


def test_put_then_get_round_trips(store, key):
    store.put(key, b"hello")

    assert store.get(key) == b"hello"


def test_exists_reports_presence(store, key):
    assert store.exists(key) is False

    store.put(key, b"x")

    assert store.exists(key) is True


def test_exists_raises_instead_of_reporting_absence_on_a_real_failure(key):
    """`except ClientError: return False` would answer "not there" for bad
    credentials or a missing bucket. A stage checking its own checkpoint would
    then silently redo finished work rather than surface the failure."""
    settings = get_settings()
    broken = ObjectStore(
        endpoint=settings.minio_public_url,
        access_key="wrong",
        secret_key="alsowrong",
    )

    with pytest.raises(botocore.exceptions.ClientError):
        broken.exists(key)


def test_sha256_matches_the_content(store, key):
    payload = b"the quick brown fox" * 10_000
    store.put(key, payload)

    assert store.sha256(key) == hashlib.sha256(payload).hexdigest()


def test_put_json_round_trips(store, key):
    store.put_json(key, {"page_count": 42, "title": "Báo cáo quý 3"})

    assert store.get_json(key) == {"page_count": 42, "title": "Báo cáo quý 3"}


def test_delete_makes_an_object_absent(store, key):
    store.put(key, b"x")

    store.delete(key)

    assert store.exists(key) is False


def test_a_key_outside_the_two_buckets_is_rejected(store):
    with pytest.raises(ValueError, match="bucket"):
        store.put("wat/thing.bin", b"x")


def test_presigned_put_carries_the_host_the_store_was_built_with(key):
    """The signed URL is handed to a browser, so it has to name a host the
    browser can resolve; one signed with the internal service name is useless
    outside the compose network. Both endpoints are localhost on this machine,
    so the two stores are built with explicit, differing endpoints -- otherwise
    the test would pass without exercising anything.
    """
    internal = ObjectStore(endpoint="http://minio:9000")
    public = ObjectStore(endpoint="https://files.example.test")

    assert urlparse(internal.presigned_put(key)).netloc == "minio:9000"
    assert urlparse(public.presigned_put(key)).netloc == "files.example.test"
    assert "X-Amz-Signature" in public.presigned_put(key)


def test_the_factories_are_cached_and_use_the_two_endpoints():
    settings = get_settings()

    assert get_store() is get_store()
    assert get_public_store() is get_public_store()
    assert get_store().endpoint == settings.minio_endpoint
    assert get_public_store().endpoint == settings.minio_public_url


def test_a_presigned_url_actually_accepts_an_upload(store, key):
    """Asserting on the URL's shape only proves it looks signed. Putting bytes
    through it is what catches a wrong signature version, a wrong addressing
    style, or a clock the signature disagrees with."""
    url = store.presigned_put(key)

    response = httpx.put(url, content=b"through the signed url")

    assert response.status_code == 200
    assert store.get(key) == b"through the signed url"
