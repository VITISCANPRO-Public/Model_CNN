# ── Test runner image for CI ──
# Runs pytest on the Model-CNN codebase without training data or GPU.

FROM python:3.11-slim

WORKDIR /app

# System dependencies
RUN apt-get update && apt-get install -y \
    curl git && \
    rm -rf /var/lib/apt/lists/*

# Install test dependencies
COPY tests/requirements.txt .
RUN pip install --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Ensure Python can find the scripts/ package
ENV PYTHONPATH=/app

# Run tests
CMD ["python", "-m", "pytest", "tests/", "-v", "--tb=short"]