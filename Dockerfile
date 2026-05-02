# 1. Use a slightly heavier base to ensure ML compatibility
FROM python:3.11-slim

# 2. Set environment variables for Python performance
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app \
    MODEL_NAME="all-MiniLM-L6-v2"

WORKDIR /app

# 3. Install system dependencies
# Added libgomp1: Essential for PyTorch/SentenceTransformers to run OpenMP
RUN apt-get update && apt-get install -y \
    build-essential \
    libpq-dev \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

#. Install Python dependencies

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 5.  Pre-download model weights during the BUILD phase
# This moves the 15-30 second "bottleneck" from  users to the CI/CD pipeline.
RUN python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('${MODEL_NAME}')"

# 6. Copy application code
# 
COPY . .

# 7. Final preparation
RUN chmod +x startup.sh
EXPOSE 8000

# 8. Entry point
ENTRYPOINT ["./startup.sh"]
