FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

# OS libs (sklearn/numpy wheels depend on these)
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 libstdc++6 ca-certificates curl \
 && rm -rf /var/lib/apt/lists/*

# Install Python deps
COPY requirements.txt .
RUN python -m pip install --upgrade pip \
 && pip install --no-cache-dir -r requirements.txt

# Copy only API source (thanks to .dockerignore, data/models/src won't be copied)
COPY api ./api

# Render/other PaaS will inject $PORT
EXPOSE 8000
CMD sh -c 'uvicorn api.api:app --host 0.0.0.0 --port ${PORT:-8000}'
