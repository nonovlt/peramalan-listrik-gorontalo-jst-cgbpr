# Multi-stage Dockerfile untuk Aplikasi Web SIKEL-Gorontalo
FROM python:3.11-slim as builder

WORKDIR /app

# Install build dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir --user -r requirements.txt

# Final Production Stage
FROM python:3.11-slim

WORKDIR /app

# Salin dependensi dari builder
COPY --from=builder /root/.local /root/.local
ENV PATH=/root/.local/bin:$PATH
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

# Salin source code proyek
COPY app/ ./app/
COPY templates/ ./templates/
COPY static/ ./static/
COPY data/ ./data/
COPY run.py .
COPY requirements.txt .

# Buat direktori runtime
RUN mkdir -p uploads reports data

# Port eksposur
EXPOSE 8000

# Healthcheck
HEALTHCHECK --interval=30s --timeout=5s --start-period=5s --retries=3 \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/api/status')" || exit 1

# Command peluncur
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--proxy-headers", "--forwarded-allow-ips", "*"]
