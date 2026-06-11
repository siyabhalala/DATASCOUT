FROM python:3.11-slim

WORKDIR /app

# System deps
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential curl \
    && rm -rf /var/lib/apt/lists/*

# Install Python deps first (layer caching)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy source
COPY . .

# Install as package
RUN pip install --no-cache-dir -e .

# Non-root user
RUN useradd -m datascout && chown -R datascout:datascout /app
USER datascout

EXPOSE 8000

ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

HEALTHCHECK --interval=30s --timeout=10s --start-period=10s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

CMD ["uvicorn", "datascout.api.main:app", "--host", "0.0.0.0", "--port", "8000"]