# 1. Base image
FROM python:3.11-slim

WORKDIR /app

# 2. System dependencies
RUN apt-get update && apt-get install -y \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/*

# 3. Copy and install Python dependencies
COPY requirements.txt .
RUN pip install --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# 4. Copy application code
COPY . .

# 5. Environment variables
ENV PYTHONPATH=/app

# 6. Default command: run training
CMD ["python", "main.py"]