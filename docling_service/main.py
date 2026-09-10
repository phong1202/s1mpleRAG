"""Docling wrapped in its own container.

The model loads ONCE at process startup, not per request. concurrency=1 is
enforced with a lock, not just intended: FastAPI runs a sync `def` route in
a thread pool, so without the lock, two /parse requests arriving together
would both call the same DocumentConverter concurrently -- and there is no
guarantee that is safe for a model object never designed to be reentrant.
The service pulls the object from MinIO itself -- no pushing a multi-MB
payload over HTTP.
"""

import math
import os
import tempfile
import threading
from contextlib import asynccontextmanager

import boto3
from fastapi import FastAPI
from pydantic import BaseModel

_converter = None
_convert_lock = threading.Lock()

# Streaming chunk for pulling the object from MinIO -- the same convention
# as shared/storage.py's sha256(), for the same reason: reading the whole
# body into a Python bytes object holds up to MAX_FILE_SIZE_MB in RAM per
# request, and this runs on every parse call.
_CHUNK_SIZE = 1 << 16


def _minio_endpoint() -> str:
    """Assembled from parts, the same convention app/config.py uses -- not a
    bare MINIO_ENDPOINT, which is that project's escape hatch for pointing
    at real S3 and is unset by default."""
    scheme = os.environ.get("MINIO_SCHEME", "http")
    host = os.environ["MINIO_HOST"]
    port = os.environ["MINIO_PORT"]
    return f"{scheme}://{host}:{port}"


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _converter
    from docling.document_converter import DocumentConverter

    _converter = DocumentConverter()
    yield
    _converter = None


app = FastAPI(lifespan=lifespan)


class ParseRequest(BaseModel):
    object_key: str
    pages: list[int]


@app.get("/health")
def health():
    return {"status": "ok", "models_loaded": _converter is not None}


def _page_confidence(result, page_no: int) -> float:
    """Docling scores each page it actually processed -- parse, layout,
    table and OCR sub-scores rolled into one `mean_score`. `pages` is a
    defaultdict, so a page never processed reads back as a fresh, all-NaN
    entry rather than a KeyError; NaN is also possible for a page that
    genuinely has nothing to score (e.g. blank). Either way this returns
    0.0 rather than leaking a NaN into the response, since the schema
    promises 0.0 <= confidence <= 1.0."""
    score = result.confidence.pages[page_no].mean_score
    return 0.0 if math.isnan(score) else score


@app.post("/parse")
def parse(request: ParseRequest):
    client = boto3.client(
        "s3",
        endpoint_url=_minio_endpoint(),
        aws_access_key_id=os.environ["MINIO_ACCESS_KEY"],
        aws_secret_access_key=os.environ["MINIO_SECRET_KEY"],
    )
    bucket, _, name = request.object_key.partition("/")
    body = client.get_object(Bucket=bucket, Key=name)["Body"]

    with tempfile.NamedTemporaryFile(suffix=".pdf") as fh:
        for chunk in body.iter_chunks(_CHUNK_SIZE):
            fh.write(chunk)
        fh.flush()
        with _convert_lock:
            result = _converter.convert(fh.name)

    # Per page, not the whole document repeated: each page gets its own
    # markdown slice and its own confidence score, both of which Docling
    # already computes -- returning the same whole-document text and a
    # fixed 0.9 for every page would silently duplicate content and
    # discard a real quality signal S1 needs to decide whether a page's
    # result is trustworthy.
    return {
        "pages": [
            {
                "page": p,
                "markdown": result.document.export_to_markdown(page_no=p),
                "confidence": _page_confidence(result, p),
            }
            for p in request.pages
        ]
    }
