FROM python:3.11-slim-bullseye

EXPOSE 8000

WORKDIR /app

ARG OCR_MODEL_TIER=small
ARG INSTALL_AI=false

ENV PYTHONUNBUFFERED=1 \
    OCR_MODEL_TIER=${OCR_MODEL_TIER} \
    OCR_DEVICE=cpu \
    OCR_MODEL_DIR=/app/.paddlex/official_models \
    PADDLE_PDX_CACHE_HOME=/app/.paddlex \
    PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK=True \
    PADDLE_PDX_MODEL_SOURCE=bos \
    PADDLE_PDX_ENABLE_MKLDNN_BYDEFAULT=0

COPY pyproject.toml uv.lock README.md /app/

RUN sed -i "s@http://deb.debian.org@http://mirrors.tuna.tsinghua.edu.cn@g" /etc/apt/sources.list && \
    apt-get update && \
    apt-get install -y --no-install-recommends \
        libgl1 \
        libgomp1 \
        libglib2.0-0 \
        libsm6 \
        libxrender1 \
        libxext6 && \
    if [ "${INSTALL_AI}" = "true" ]; then \
        apt-get install -y --no-install-recommends build-essential cmake; \
    fi && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

RUN python3 -m pip install -i https://pypi.tuna.tsinghua.edu.cn/simple --no-cache-dir --upgrade pip uv && \
    if [ "${INSTALL_AI}" = "true" ]; then \
        uv sync --frozen --no-dev --no-install-project --extra ai; \
    else \
        uv sync --frozen --no-dev --no-install-project; \
    fi

COPY . /app

RUN uv run --no-sync python -m scripts.prepare_models --with-doc-preprocessor

CMD ["uv", "run", "--no-sync", "uvicorn", "main:app", "--host", "0.0.0.0", "--workers", "1", "--log-config", "./log_conf.yaml"]
