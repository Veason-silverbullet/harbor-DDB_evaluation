FROM node:22-bookworm-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 python3-venv git curl ca-certificates procps \
    && rm -rf /var/lib/apt/lists/*
RUN python3 -m venv /opt/ddb-venv
ENV PATH="/opt/ddb-venv/bin:${PATH}" \
    PLAYWRIGHT_BROWSERS_PATH=/opt/playwright \
    PYTHONUNBUFFERED=1
COPY requirements.txt /opt/ddb-requirements.txt
RUN pip install --no-cache-dir -r /opt/ddb-requirements.txt \
    && playwright install --with-deps chromium \
    && rm -rf /var/lib/apt/lists/*
WORKDIR /app
RUN git init && git config --global --add safe.directory /app
CMD ["sleep", "infinity"]
