FROM ghcr.io/astral-sh/uv:python3.12-bookworm-slim

RUN apt-get update && \
    apt-get install -y --no-install-recommends curl ca-certificates git && \
    rm -rf /var/lib/apt/lists/*

RUN git clone --depth 1 https://github.com/NousResearch/hermes-agent.git /tmp/hermes-agent && \
    cd /tmp/hermes-agent && \
    uv pip install --system --no-cache -e ".[all]" && \
    rm -rf /tmp/hermes-agent/.git

COPY requirements.txt /app/requirements.txt
RUN uv pip install --system --no-cache -r /app/requirements.txt

COPY server.py /app/server.py
COPY start.sh /app/start.sh
RUN chmod +x /app/start.sh

ENV HOME=/data
ENV HERMES_HOME=/data/.hermes
ENV API_SERVER_PORT=8642
ENV API_SERVER_HOST=127.0.0.1

CMD ["/app/start.sh"]
