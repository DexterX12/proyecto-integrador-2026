# Dashboard Streamlit en Fargate. Los datos NO van en la imagen: entrypoint.sh los baja de S3 al arrancar
# (solo la versión de modelo publicada, sin pesos ni embeddings).
# Construir desde la raíz del repo:  docker build --platform linux/amd64 -f deploy/dashboard/Dockerfile -t fashrec-dashboard .
FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PYTHONPATH=/app \
    FASHREC_DATA_ROOT=/data \
    STREAMLIT_SERVER_HEADLESS=true \
    STREAMLIT_BROWSER_GATHER_USAGE_STATS=false

RUN apt-get update \
 && apt-get install -y --no-install-recommends libgomp1 curl unzip ca-certificates \
 && curl -fsSL https://awscli.amazonaws.com/awscli-exe-linux-x86_64.zip -o /tmp/awscli.zip \
 && unzip -q /tmp/awscli.zip -d /tmp && /tmp/aws/install \
 && rm -rf /tmp/aws /tmp/awscli.zip /var/lib/apt/lists/*

COPY deploy/dashboard/requirements.txt /tmp/requirements.txt
RUN pip install -r /tmp/requirements.txt

WORKDIR /app
COPY src ./src
COPY app ./app
COPY configs ./configs
COPY deploy/dashboard/entrypoint.sh /usr/local/bin/entrypoint.sh
RUN sed -i 's/\r$//' /usr/local/bin/entrypoint.sh && chmod +x /usr/local/bin/entrypoint.sh && mkdir -p /data

EXPOSE 8501
HEALTHCHECK --interval=30s --timeout=5s --start-period=180s CMD curl -fsS http://localhost:8501/_stcore/health || exit 1
ENTRYPOINT ["/usr/local/bin/entrypoint.sh"]
