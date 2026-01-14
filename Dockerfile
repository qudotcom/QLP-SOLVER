# ==============================================================
# Quantum Elasticity Solver - Combined Dockerfile for Koyeb
# Serves both FastAPI backend and static frontend via Nginx
# ==============================================================

# Stage 1: Build Python dependencies
FROM python:3.11-slim as python-base

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# Copy and install Python requirements
COPY backend/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy backend code
COPY backend/main.py .

# Stage 2: Final image with Nginx + Python
FROM python:3.11-slim

# Install nginx, supervisor, and curl (for health checks)
RUN apt-get update && apt-get install -y \
    nginx \
    supervisor \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy Python packages from build stage
COPY --from=python-base /usr/local/lib/python3.11/site-packages /usr/local/lib/python3.11/site-packages
COPY --from=python-base /usr/local/bin/uvicorn /usr/local/bin/uvicorn

# Set up backend
WORKDIR /app
COPY backend/main.py .

# Set up frontend - copy static files to nginx
COPY frontend/index.html /var/www/html/
COPY frontend/styles.css /var/www/html/
COPY frontend/app.js /var/www/html/

# Copy nginx configuration
COPY nginx.koyeb.conf /etc/nginx/sites-available/default

# Copy supervisor configuration
COPY supervisord.conf /etc/supervisor/conf.d/supervisord.conf

# Expose port (Koyeb uses PORT env variable, default 8000)
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Start supervisor (manages both nginx and uvicorn)
CMD ["/usr/bin/supervisord", "-c", "/etc/supervisor/conf.d/supervisord.conf"]
