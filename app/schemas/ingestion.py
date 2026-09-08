from pydantic import BaseModel, ConfigDict, Field

_SHA256_HEX = r"^[0-9a-f]{64}$"


class UploadUrlRequest(BaseModel):
    filename: str = Field(min_length=1, max_length=255)
    # Required up front, not just at register time: the object key is
    # derived from it (raw/{sha256}.pdf) and it is bound into the presigned
    # URL's signature so MinIO itself rejects mismatched bytes on upload.
    sha256: str = Field(pattern=_SHA256_HEX)


class UploadTarget(BaseModel):
    upload_url: str
    object_key: str
    expires_in: int


class DocumentRegister(BaseModel):
    object_key: str = Field(min_length=1)
    filename: str = Field(min_length=1, max_length=255)
    sha256: str = Field(pattern=_SHA256_HEX)
    size_bytes: int = Field(gt=0)


class DocumentAccepted(BaseModel):
    document_id: str
    status: str


class DocumentStatus(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: str
    filename: str
    status: str
    stage: str | None
    attempts: int
    failed_stage: str | None
    last_error: str | None
