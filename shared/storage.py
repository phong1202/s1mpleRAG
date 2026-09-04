"""Object storage client shared by the API and the worker.

boto3 rather than the MinIO SDK: MinIO speaks the S3 API, so moving to real
S3 later is a change of endpoint rather than a rewrite of this file.
"""

import hashlib
import json
from functools import lru_cache

import boto3
from botocore.config import Config
from botocore.exceptions import ClientError

from app.config import get_settings

# Streaming chunk for hashing. Anything is better than reading whole objects,
# and 64 KiB keeps the syscall count low without holding much.
_CHUNK_SIZE = 1 << 16

# What S3 returns for "no such object". head_object answers 404 where
# get_object answers NoSuchKey, so both are treated as absence.
_ABSENT = {"404", "NoSuchKey", "NotFound"}


class ObjectStore:
    def __init__(
        self,
        endpoint: str | None = None,
        access_key: str | None = None,
        secret_key: str | None = None,
    ) -> None:
        settings = get_settings()
        self.endpoint = endpoint or settings.minio_endpoint
        self._bucket_raw = settings.minio_bucket_raw
        self._bucket_staging = settings.minio_bucket_staging
        self._client = boto3.client(
            "s3",
            endpoint_url=self.endpoint,
            aws_access_key_id=access_key or settings.minio_access_key,
            aws_secret_access_key=secret_key or settings.minio_secret_key,
            region_name=settings.minio_region,
            # Both are explicit because the defaults are wrong here.
            # Without s3v4 botocore signs presigned URLs with SigV2, which
            # cannot bind extra headers -- ruling out having MinIO verify an
            # upload's checksum. Path addressing because "bucket.minio:9000"
            # does not resolve inside the compose network.
            config=Config(
                signature_version="s3v4",
                s3={"addressing_style": "path"},
            ),
        )

    def _split(self, key: str) -> tuple[str, str]:
        """'raw/abc.pdf' -> ('raw', 'abc.pdf'). The bucket travels inside the
        key so every layer above only has to carry one string."""
        bucket, _, rest = key.partition("/")
        if bucket not in (self._bucket_raw, self._bucket_staging):
            raise ValueError(f"unknown bucket in key: {key!r}")
        return bucket, rest

    def put(self, key: str, data: bytes) -> None:
        bucket, name = self._split(key)
        self._client.put_object(Bucket=bucket, Key=name, Body=data)

    def get(self, key: str) -> bytes:
        bucket, name = self._split(key)
        return self._client.get_object(Bucket=bucket, Key=name)["Body"].read()

    def delete(self, key: str) -> None:
        """Deleting an absent key is not an error in S3, so this is idempotent."""
        bucket, name = self._split(key)
        self._client.delete_object(Bucket=bucket, Key=name)

    def exists(self, key: str) -> bool:
        """Only a genuine "not found" answers False. Bad credentials, a missing
        bucket or a network failure raise: a stage asking whether its own
        checkpoint is there would otherwise redo finished work on any error.
        """
        bucket, name = self._split(key)
        try:
            self._client.head_object(Bucket=bucket, Key=name)
        except ClientError as error:
            if error.response["Error"]["Code"] in _ABSENT:
                return False
            raise
        return True

    def sha256(self, key: str) -> str:
        """Hash the stored bytes without holding them.

        boto3 hands back a StreamingBody; reading it whole would pull up to
        MAX_FILE_SIZE_MB into memory per call, and this runs on every upload.
        """
        bucket, name = self._split(key)
        body = self._client.get_object(Bucket=bucket, Key=name)["Body"]
        digest = hashlib.sha256()
        for chunk in body.iter_chunks(_CHUNK_SIZE):
            digest.update(chunk)
        return digest.hexdigest()

    def put_json(self, key: str, obj: dict) -> None:
        self.put(key, json.dumps(obj, ensure_ascii=False).encode())

    def get_json(self, key: str) -> dict:
        return json.loads(self.get(key))

    def presigned_put(self, key: str, expires_in: int = 3600) -> str:
        bucket, name = self._split(key)
        return self._client.generate_presigned_url(
            "put_object",
            Params={"Bucket": bucket, "Key": name},
            ExpiresIn=expires_in,
        )


@lru_cache
def get_store() -> ObjectStore:
    """The internal store, for everything running inside the compose network."""
    return ObjectStore()


@lru_cache
def get_public_store() -> ObjectStore:
    """Signs URLs for clients outside the network. A presigned URL is bound to
    one host, so the host it is signed with is the host it must be fetched
    from."""
    return ObjectStore(endpoint=get_settings().minio_public_url)
