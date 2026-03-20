# Investment Dashboard – Flask API
# Build: docker build -t investment-api .
# Run:   docker run -p 9000:9000 investment-api

FROM python:3.11-slim

WORKDIR /app

# Install Python dependencies first (cached layer)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application source
COPY . .

# Expose the API port (overridable via API_PORT env var)
EXPOSE 9000

# Run the Flask API
CMD ["python", "flask_api.py"]
