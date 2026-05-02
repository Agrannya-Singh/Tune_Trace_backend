#!/bin/bash
set -e

echo " Starting TuneTrace Backend..."

# 1. Run database migrations
#echo " Running Alembic migrations..."
#alembic upgrade head ## TODO : alembic migrations to be added to CI 

# 2. Pre-cache ML model weights
# This ensures the SentenceTransformer is ready before the first request
echo " Initializing Semantic Engine (all-MiniLM-L6-v2)..."
python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('all-MiniLM-L6-v2')"

# 3. Start Gunicorn with Uvicorn workers
echo " Starting server on port 8000..."
exec gunicorn -w 4 -k uvicorn.workers.UvicornWorker main:app --bind 0.0.0.0:8000
