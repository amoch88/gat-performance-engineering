# syntax=docker/dockerfile:1
FROM --platform=linux/amd64 python:3.11.11-slim-bookworm@sha256:081075da77b2b55c23c088251026fb69a7b2bf92471e491ff5fd75c192fd38e5
COPY --from=ghcr.io/astral-sh/uv:0.13.4@sha256:9f130421e930bfbdc563c00740466603f8723158e90297d07957c2294640262b /uv /usr/local/bin/uv
ENV DGLBACKEND=pytorch UV_PYTHON_DOWNLOADS=never UV_LINK_MODE=copy \
    OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
WORKDIR /app
# libgomp is needed by the official DGL CPU wheel. A build-only CA bundle
# can be supplied by managed environments without putting it in the image.
RUN --mount=type=secret,id=proxy_ca \
    if [ -f /run/secrets/proxy_ca ]; then \
      apt-get -o Acquire::https::CaInfo=/run/secrets/proxy_ca update; \
    else apt-get update; fi \
    && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*
COPY pyproject.toml uv.lock ./
RUN --mount=type=secret,id=proxy_ca \
    if [ -f /run/secrets/proxy_ca ]; then export SSL_CERT_FILE=/run/secrets/proxy_ca; fi; \
    uv sync --frozen --no-dev
COPY benchmarks ./benchmarks
ARG GIT_COMMIT=unknown
ENV BENCHMARK_GIT_COMMIT=$GIT_COMMIT BENCHMARK_IMAGE=dgl-performance-model:cpu
ENTRYPOINT ["uv", "run", "--frozen", "--no-sync", "python", "benchmarks/gat.py"]
