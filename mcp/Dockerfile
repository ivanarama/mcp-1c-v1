FROM python:3.11-slim

WORKDIR /app

# Install system dependencies (gcc not needed for qdrant-client pure HTTP mode)
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better caching
COPY requirements.txt .

# Install Python dependencies with optimizations
ENV PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_PYTHON_VERSION_WARNING=1 \
    PIP_PREFER_BINARY=1
RUN pip install --no-cache-dir --prefer-binary -r requirements.txt

# Copy application code
COPY . .

# Expose port
EXPOSE 8001

# Health check
# HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
#     CMD curl -f http://localhost:8000/health || exit 1

# Run the application
CMD ["python", "mcp_server.py"]
