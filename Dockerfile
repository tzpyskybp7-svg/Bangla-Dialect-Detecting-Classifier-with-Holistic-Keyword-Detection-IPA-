FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

# system deps (audio)
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    libsndfile1 \
    build-essential \
 && rm -rf /var/lib/apt/lists/*

# copy only what we need
COPY dialect_package/requirements.txt /app/requirements.txt
RUN pip install --upgrade pip && pip install -r /app/requirements.txt

# app code + artifacts
COPY dialect_package/src /app/src
COPY artifacts /app/artifacts

# make src importable
ENV PYTHONPATH=/app/src \
    MODEL_DIR=/app/artifacts/runs/latest \
    CHUNK_SELECT=first \
    HOST=0.0.0.0 \
    PORT=8000

EXPOSE 8000

# RUN UVICORN DIRECTLY (no run_app.py)
CMD ["python", "-m", "uvicorn", "dialect_predict:build_app", "--factory", "--host", "0.0.0.0", "--port", "8000", "--log-level", "info"]
