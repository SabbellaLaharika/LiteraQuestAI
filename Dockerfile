# Multi-Agent Academic Research Assistant Dockerfile
FROM python:3.11-slim

# Prevent Python from writing .pyc files and enable unbuffered output
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

# Install system dependencies including curl for healthchecks
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements.txt /app/
RUN pip install --no-cache-dir -r requirements.txt

# Copy source code and files
COPY . /app/

# Ensure output directory exists with proper permissions
RUN mkdir -p /app/output

# Healthcheck for app container
HEALTHCHECK --interval=15s --timeout=5s --start-period=10s --retries=3 \
    CMD python -c "import sys; from src.config import settings; sys.exit(0)"

# Expose web dashboard port
EXPOSE 8000

# Default entrypoint for docker-compose up
CMD ["python", "main.py", "--serve"]
