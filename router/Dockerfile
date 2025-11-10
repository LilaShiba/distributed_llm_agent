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
EXPOSE 8000

# Flask environment
ENV FLASK_APP=app.py
ENV FLASK_RUN_HOST=0.0.0.0
ENV FLASK_RUN_PORT=8000

# Run Flask
CMD ["flask", "run"]
