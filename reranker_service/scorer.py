"""Loads the cross-encoder named by the environment. Imported only by the
container's entry point: torch and the model stay out of everything else.

RERANKER_MODEL               a Hugging Face cross-encoder
RERANKER_BACKEND             onnx (default) or torch
RERANKER_ONNX_FILE           the ONNX file inside the repo, for a quantised one
RERANKER_TRUST_REMOTE_CODE   true for models that ship their own code (gte)
RERANKER_MAX_LENGTH          tokens per (query, document) pair, default 512
RERANKER_THREADS             torch threads, default 8 (ONNX Runtime uses every
                             physical core on its own)
RERANKER_MAX_WAITING         requests queued behind the one being scored before
                             new ones are refused with 503, default 2
RERANKER_MODEL_DIR           where the image keeps the model's files

The files are fetched into RERANKER_MODEL_DIR while the image is built and
loaded from there. Loading by repo id instead would ask the Hub for the repo's
file list on every start, and fail on a host without network.
"""

import os
from pathlib import Path


def settings() -> dict:
    return {
        "model": os.environ.get("RERANKER_MODEL", "cross-encoder/mmarco-mMiniLMv2-L12-H384-v1"),
        "backend": os.environ.get("RERANKER_BACKEND", "onnx"),
        "onnx_file": os.environ.get("RERANKER_ONNX_FILE", "onnx/model_qint8_avx512_vnni.onnx"),
        "trust_remote_code": os.environ.get("RERANKER_TRUST_REMOTE_CODE", "").lower()
        in ("1", "true", "yes"),
        "max_length": int(os.environ.get("RERANKER_MAX_LENGTH", "512")),
        "threads": int(os.environ.get("RERANKER_THREADS", "8")),
        "model_dir": os.environ.get("RERANKER_MODEL_DIR", "/opt/model"),
        "max_waiting": int(os.environ.get("RERANKER_MAX_WAITING", "2")),
    }


def fetch(config: dict) -> None:
    """Downloads what loading needs, and nothing else: a repo that ships ONNX
    files for six instruction sets would otherwise put all six in the image."""
    from huggingface_hub import snapshot_download

    prebuilt = config["backend"] == "onnx" and bool(config["onnx_file"])
    ignore = ["onnx/*", "openvino/*", "*.onnx", "*.onnx_data", "*.h5", "*.msgpack", "*.ot"]
    if prebuilt:
        ignore += ["*.safetensors", "*.bin"]  # the ONNX file carries the weights
    snapshot_download(config["model"], local_dir=config["model_dir"], ignore_patterns=ignore)
    if prebuilt:
        snapshot_download(
            config["model"], local_dir=config["model_dir"], allow_patterns=[config["onnx_file"]]
        )
        if not (Path(config["model_dir"]) / config["onnx_file"]).exists():
            # Without this the build fails later, in the load, with an error
            # that names neither the file nor the setting.
            raise SystemExit(
                f"{config['model']} has no {config['onnx_file']}. Set RERANKER_LOCAL_ONNX_FILE "
                "to an ONNX file the repo ships, or RERANKER_LOCAL_BACKEND=torch."
            )


def load(config: dict):
    import torch
    from sentence_transformers import CrossEncoder

    torch.set_num_threads(config["threads"])
    kwargs = {}
    if config["backend"] == "onnx" and config["onnx_file"]:
        kwargs["model_kwargs"] = {"file_name": config["onnx_file"]}
    local = Path(config["model_dir"])
    source = str(local) if (local / "config.json").exists() else config["model"]
    model = CrossEncoder(
        source,
        backend=config["backend"],
        max_length=config["max_length"],
        trust_remote_code=config["trust_remote_code"],
        device="cpu",
        **kwargs,
    )

    def score(query: str, documents: list[str]) -> list[float]:
        # Raw scores: their order is all the caller uses, and an activation
        # only rescales it.
        return [float(s) for s in model.predict([(query, d) for d in documents], batch_size=16)]

    return score
