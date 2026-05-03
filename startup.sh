#!/bin/bash
set -e

echo " Starting TuneTrace Backend..."

# 1. Run database migrations
#echo " Running Alembic migrations..."
#alembic upgrade head ## TODO : alembic migrations to be added to CI 

# 2. Server Startup
echo " Starting server on port 8000..."
exec gunicorn -w 2 --timeout 90 -k uvicorn.workers.UvicornWorker main:app --bind 0.0.0.0:8000
