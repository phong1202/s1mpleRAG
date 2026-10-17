# The CPU reranker. Built from the repository root:
#   docker compose build reranker
FROM python:3.12-slim

ENV PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    HF_HOME=/opt/hf

RUN pip install --index-url https://download.pytorch.org/whl/cpu torch==2.14.1
COPY reranker_service/requirements.txt /tmp/requirements.txt
RUN pip install -r /tmp/requirements.txt

ARG RERANKER_MODEL=cross-encoder/mmarco-mMiniLMv2-L12-H384-v1
ARG RERANKER_BACKEND=onnx
ARG RERANKER_ONNX_FILE=onnx/model_qint8_avx512_vnni.onnx
ARG RERANKER_TRUST_REMOTE_CODE=false
ENV RERANKER_MODEL=$RERANKER_MODEL \
    RERANKER_BACKEND=$RERANKER_BACKEND \
    RERANKER_ONNX_FILE=$RERANKER_ONNX_FILE \
    RERANKER_TRUST_REMOTE_CODE=$RERANKER_TRUST_REMOTE_CODE

WORKDIR /srv
COPY reranker_service ./reranker_service

# The model's files are fetched while the image is built: a container that
# starts needs no network, and a restart downloads nothing. With the onnx
# backend the repo must ship the ONNX file named by RERANKER_ONNX_FILE;
# sentence-transformers can export one, but only in memory, at every start.
# Loaded once more with the network shut off, so an image that would need
# the Hub at start fails here instead of on a host without network.
RUN python -c "from reranker_service import scorer; scorer.fetch(scorer.settings())" \
 && HF_HUB_OFFLINE=1 python -c "from reranker_service import scorer; scorer.load(scorer.settings())"
ENV HF_HUB_OFFLINE=1

RUN useradd --create-home reranker && mkdir -p /opt/hf && chown -R reranker /opt/hf /opt/model
USER reranker

EXPOSE 8200
CMD ["uvicorn", "reranker_service.main:app", "--host", "0.0.0.0", "--port", "8200"]
