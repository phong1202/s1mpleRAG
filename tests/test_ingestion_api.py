"""Hash-first upload: the client supplies sha256 before an upload URL is
issued, not after. That lets the object key be derived from the hash
(raw/{sha256}.pdf) and the checksum bound into the presigned URL's
signature, so MinIO itself rejects mismatched bytes on upload -- register()
below never has to read an object back to verify it.
"""

import base64
import hashlib
import uuid

import httpx
import pytest

pytestmark = pytest.mark.asyncio


def _b64(sha256_hex: str) -> str:
    return base64.b64encode(bytes.fromhex(sha256_hex)).decode()


async def test_upload_url_returns_a_presigned_put(client):
    digest = hashlib.sha256(b"whatever the client is about to upload").hexdigest()

    response = await client.post(
        "/documents/upload-url", json={"filename": "a.pdf", "sha256": digest}
    )

    assert response.status_code == 200
    data = response.json()["data"]
    assert data["object_key"] == f"raw/{digest}.pdf"
    assert "X-Amz-Signature" in data["upload_url"]


async def test_upload_url_rejects_an_already_ingested_hash(client, uploaded_pdf):
    """Dedup runs before the upload, not only at register: a client that
    already has this file ingested should not have to spend bandwidth
    uploading it again just to be told so."""
    await client.post("/documents", json=uploaded_pdf)

    response = await client.post(
        "/documents/upload-url",
        json={"filename": uploaded_pdf["filename"], "sha256": uploaded_pdf["sha256"]},
    )

    assert response.status_code == 409
    assert response.json()["message"] == "Document already ingested"


async def test_the_full_upload_and_register_flow_against_real_minio(client):
    """The two endpoints have to compose: a URL from upload-url has to
    actually accept the exact bytes it was signed for, through a real PUT,
    before register can see the object at all."""
    payload = b"a whole fake pdf worth of bytes"
    digest = hashlib.sha256(payload).hexdigest()

    url_response = await client.post(
        "/documents/upload-url", json={"filename": "report.pdf", "sha256": digest}
    )
    target = url_response.json()["data"]

    put_response = httpx.put(
        target["upload_url"],
        content=payload,
        headers={"x-amz-checksum-sha256": _b64(digest)},
    )
    assert put_response.status_code == 200

    register_response = await client.post(
        "/documents",
        json={
            "object_key": target["object_key"],
            "filename": "report.pdf",
            "sha256": digest,
            "size_bytes": len(payload),
        },
    )

    assert register_response.status_code == 202
    assert register_response.json()["data"]["status"] == "QUEUED"


async def test_register_returns_202_and_a_queued_document(client, uploaded_pdf):
    response = await client.post("/documents", json=uploaded_pdf)

    assert response.status_code == 202
    assert response.json()["data"]["status"] == "QUEUED"
    assert uuid.UUID(response.json()["data"]["document_id"])


async def test_registering_the_same_hash_twice_returns_409(client, uploaded_pdf):
    await client.post("/documents", json=uploaded_pdf)

    second = await client.post("/documents", json=uploaded_pdf)

    assert second.status_code == 409
    assert second.json()["message"] == "Document already ingested"


async def test_a_wrong_client_supplied_hash_is_rejected(client, uploaded_pdf):
    """object_key is raw/{sha256}.pdf by construction, so a mismatch here
    means the two fields did not come from the same upload-url call --
    caught by comparing strings, no object read required."""
    payload = {**uploaded_pdf, "sha256": "f" * 64}

    response = await client.post("/documents", json=payload)

    assert response.status_code == 400
    assert "does not match" in response.json()["message"]


async def test_an_object_that_was_never_uploaded_is_rejected(client):
    payload = {
        "object_key": "raw/" + "9" * 64 + ".pdf",
        "filename": "ghost.pdf",
        "sha256": "9" * 64,
        "size_bytes": 10,
    }

    response = await client.post("/documents", json=payload)

    assert response.status_code == 404
