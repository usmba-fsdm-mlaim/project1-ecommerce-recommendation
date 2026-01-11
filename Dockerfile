
# Multi-stage build for production
FROM python:3.10-slim as builder

WORKDIR /app

# Install build dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc g++ && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
# Increase pip network timeout and retry to avoid transient ReadTimeout errors
ENV PIP_DEFAULT_TIMEOUT=120
RUN for i in 1 2 3; do \
      pip --disable-pip-version-check install --no-cache-dir --user -r requirements.txt --timeout 120 && break || \
      (echo "pip install failed, retrying ($i)" && sleep 5); \
    done

# Runtime stage
FROM python:3.10-slim

WORKDIR /app

# Install runtime dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl && rm -rf /var/lib/apt/lists/*

# Copy Python packages from builder
COPY --from=builder /root/.local /root/.local
ENV PATH=/root/.local/bin:$PATH

# Copy source code
COPY src/ ./src/
COPY web/ ./web/

# Create directories
RUN mkdir -p models data logs

# Environment variables
ENV MODEL_PATH=/app/models/recommendation_model.pkl
ENV DATA_PATH=/app/data/cleaned_data.csv
ENV PYTHONUNBUFFERED=1

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

CMD ["uvicorn", "src.app:app", "--host", "0.0.0.0", "--port", "8000"]