# ---------------------------------------------------------------------------
# Investment Dashboard – Flask API
# ---------------------------------------------------------------------------
# Build:  docker build -t investment-api .
# Run:    docker run -p 9000:9000 --env-file .env investment-api
# ---------------------------------------------------------------------------

FROM python:3.11-slim

# Security: run as non-root
RUN useradd --create-home --shell /bin/bash appuser

WORKDIR /app

# Install dependencies first (layer-cached when code changes)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Set ownership
RUN chown -R appuser:appuser /app

USER appuser

# Expose Flask API port
EXPOSE 9000

# Health check (uses the /health endpoint)
HEALTHCHECK --interval=30s --timeout=10s --start-period=30s --retries=3 \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:9000/health')" || exit 1

ENV PYTHONUNBUFFERED=1
ENV API_PORT=9000

CMD ["python", "flask_api.py"]
