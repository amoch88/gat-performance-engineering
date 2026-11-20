# syntax=docker/dockerfile:1
# Build the unchanged root CPU image first. This extension adds only a CPU compiler.
FROM dgl-performance-model:cpu
RUN --mount=type=secret,id=proxy_ca \
    if [ -f /run/secrets/proxy_ca ]; then apt-get -o Acquire::https::CaInfo=/run/secrets/proxy_ca update; \
    else apt-get update; fi \
    && apt-get install -y --no-install-recommends g++ libgomp1 \
    && rm -rf /var/lib/apt/lists/*
COPY benchmarks/tuning_axes /app/benchmarks/tuning_axes
RUN uv run --frozen --no-sync python benchmarks/tuning_axes/native.py
ENTRYPOINT ["uv", "run", "--frozen", "--no-sync", "python", "benchmarks/tuning_axes/run.py"]
