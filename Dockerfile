FROM python:3.12-slim

# uv ships as a single static binary — copying it in is cheaper than
# installing it with pip, and pins the exact version used to build.
COPY --from=ghcr.io/astral-sh/uv:0.12.7 /uv /usr/local/bin/uv

WORKDIR /code

# The environment lives outside /code deliberately: docker-compose bind-mounts
# the source over /code, which would shadow a venv kept there and leave the
# container without its packages at runtime.
#
# UV_PYTHON_DOWNLOADS=never keeps uv on the interpreter this image already
# ships instead of downloading a second, managed one.
ENV UV_PROJECT_ENVIRONMENT=/opt/venv \
    UV_PYTHON_DOWNLOADS=never \
    UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    PATH="/opt/venv/bin:$PATH"

# Manifests first, so the dependency layer is cached across source changes.
COPY pyproject.toml uv.lock .python-version ./
RUN uv sync --frozen --no-dev

# The BPE table tiktoken needs is a 1.7 MB download it would otherwise fetch on
# first import -- at runtime, from a container that may have no route out. Bake
# it in instead, so importing the context builder never touches the network.
ENV TIKTOKEN_CACHE_DIR=/opt/tiktoken
RUN mkdir -p "$TIKTOKEN_CACHE_DIR" \
    && python -c "import tiktoken; tiktoken.get_encoding('cl100k_base')"

COPY . .

RUN useradd --create-home appuser && chown -R appuser:appuser /code /opt/venv /opt/tiktoken
USER appuser

EXPOSE 8000
CMD ["uvicorn", "app.app:app", "--host", "0.0.0.0", "--port", "8000"]
