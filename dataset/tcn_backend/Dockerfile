# syntax=docker/dockerfile:1
# Label Studio ML backend serving the Muay-ThAI TCN bundles as pre-annotations.
#
# Build context is the REPOSITORY ROOT (see dataset/compose.yml and the root
# .dockerignore): the image copies models/ so `import models.action_detection`
# works with PYTHONPATH=/app. Weights are NOT baked in; compose bind-mounts
# models/action_detection/TCN/weights and models/yolo/weights read-only.
#
# Base image: CUDA 12.8 torch, required by the Blackwell GPU on the dev machine
# and matching the conda env (torch 2.11.0+cu128). Torch lives in Ubuntu's
# system Python (PEP 668), hence PIP_BREAK_SYSTEM_PACKAGES below.
FROM pytorch/pytorch:2.11.0-cuda12.8-cudnn9-runtime
ARG TEST_ENV

WORKDIR /app

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PORT=9090 \
    WORKERS=1 \
    THREADS=4 \
    PYTHONPATH=/app \
    PIP_BREAK_SYSTEM_PACKAGES=1

RUN apt-get update \
    && apt-get install -y --no-install-recommends git ffmpeg libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

COPY dataset/tcn_backend/requirements-base.txt \
     dataset/tcn_backend/requirements.txt \
     dataset/tcn_backend/requirements-test.txt ./
RUN pip install --no-cache-dir -r requirements-base.txt -r requirements.txt \
    && if [ "$TEST_ENV" = "true" ]; then pip install --no-cache-dir -r requirements-test.txt; fi

# Our code: the models package and the backend itself.
COPY models/ /app/models/
COPY dataset/tcn_backend/ /app/

EXPOSE 9090

# WORKERS=1 keeps the loaded models and the job queue in one process;
# --preload loads them at boot instead of on the first request.
CMD gunicorn --preload --bind :$PORT --workers $WORKERS --threads $THREADS --timeout 0 _wsgi:app
