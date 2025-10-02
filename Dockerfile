# ----- Base Python image (stable, with wheels for sklearn/numpy) -----
FROM python:3.11-slim

# Python & pip sane defaults
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

# System libs needed by scikit-learn wheels
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 ca-certificates curl && \
    rm -rf /var/lib/apt/lists/*

# Install Python deps
COPY requirements.txt .
RUN pip install --upgrade pip && pip install -r requirements.txt

# Copy source
COPY api ./api
COPY models ./models

# (Optional) if you keep a small model_meta.json only; big model downloads at runtime
# Ensure port is open (Railway sets $PORT at runtime)
EXPOSE 8000

# Launch app; use Railway $PORT if present, else 8000
CMD ["sh","-c","uvicorn api.api:app --host 0.0.0.0 --port ${PORT:-8000}"]
