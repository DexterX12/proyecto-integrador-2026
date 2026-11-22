# Imagen única del pipeline en Fargate: la misma para features, train, evaluate, recommend, publish, export y load.
# Qué etapa corre lo decide la máquina de estados (variable STAGE + Command); entrypoint.sh sincroniza S3 <-> /data.
# Construir desde la raíz del repo:  docker build --platform linux/amd64 -f deploy/pipeline/Dockerfile -t fashrec-pipeline .
FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PYTHONPATH=/app \
    FASHREC_DATA_ROOT=/data

# libgomp: OpenMP (faiss, torch). AWS CLI v2: sincronización S3 <-> disco local en entrypoint.sh.
RUN apt-get update \
 && apt-get install -y --no-install-recommends libgomp1 curl unzip ca-certificates \
 && curl -fsSL https://awscli.amazonaws.com/awscli-exe-linux-x86_64.zip -o /tmp/awscli.zip \
 && unzip -q /tmp/awscli.zip -d /tmp && /tmp/aws/install \
 && rm -rf /tmp/aws /tmp/awscli.zip /var/lib/apt/lists/*

RUN pip install torch==2.15.0 --index-url https://download.pytorch.org/whl/cpu
COPY deploy/pipeline/requirements.txt /tmp/requirements.txt
RUN pip install -r /tmp/requirements.txt

WORKDIR /app
COPY src ./src
COPY jobs ./jobs
COPY configs ./configs
COPY deploy/pipeline/entrypoint.sh /usr/local/bin/entrypoint.sh
RUN sed -i 's/\r$//' /usr/local/bin/entrypoint.sh && chmod +x /usr/local/bin/entrypoint.sh && mkdir -p /data

ENTRYPOINT ["/usr/local/bin/entrypoint.sh"]
CMD ["python", "-m", "jobs.features_job"]
