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

COPY . .

RUN useradd --create-home appuser && chown -R appuser:appuser /code /opt/venv
USER appuser

EXPOSE 8000
CMD ["uvicorn", "app.app:app", "--host", "0.0.0.0", "--port", "8000"]
