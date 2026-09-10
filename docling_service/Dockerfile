FROM python:3.12-slim
# opencv-python (pulled in by docling's table model) links against X11/GL
# libraries python:3.12-slim does not ship. Found by actually running the
# service: it imports fine, then fails on the first real /parse call with
# `ImportError: libxcb.so.1: cannot open shared object file`.
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 libglib2.0-0 libsm6 libxext6 libxrender1 libxcb1 \
    && rm -rf /var/lib/apt/lists/*
COPY --from=ghcr.io/astral-sh/uv:0.12.7 /uv /usr/local/bin/uv
WORKDIR /svc
ENV UV_PROJECT_ENVIRONMENT=/opt/venv UV_PYTHON_DOWNLOADS=never PATH="/opt/venv/bin:$PATH"
# Weights land here; compose mounts a volume at this path so they are
# downloaded once, not on every container start.
ENV HF_HOME=/models
COPY pyproject.toml ./
RUN uv sync --no-dev
COPY main.py ./
EXPOSE 8100
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8100", "--workers", "1"]
