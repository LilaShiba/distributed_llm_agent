# Use full Debian-based Python image for ARM
FROM python:3.11-bullseye

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    libffi-dev \
    libssl-dev \
    wget \
    && rm -rf /var/lib/apt/lists/*

# Upgrade pip and wheel
RUN python -m pip install --upgrade pip wheel setuptools

# Copy requirements and install
COPY requirements.txt .
RUN pip install --no-cache-dir --default-timeout=200 -r requirements.txt

# Copy router code
COPY . .

# Expose router port
EXPOSE 5000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python -c "import requests; requests.get('http://localhost:5000/health', timeout=5)"

# Run with Gunicorn for better production performance
CMD ["gunicorn", "--bind", "0.0.0.0:5000", "--workers", "2", "--timeout", "60", "app:app"]
