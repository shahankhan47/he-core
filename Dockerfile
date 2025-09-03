# Use official slim Python image
FROM python:3.10-slim

# Set working dir in the container
WORKDIR /app

# Install system-level dependencies (edit as needed)
RUN apt-get update && apt-get install -y \
    build-essential \
    libpq-dev \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install before copying rest (for Docker cache)
COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

# Copy only necessary source files into the image
COPY . .

# Create user and set permissions (security)
RUN addgroup --system appuser && adduser --system --ingroup appuser appuser
USER appuser

# If you want, expose your web app port (e.g. FastAPI default 8000)
EXPOSE 8000
#command to run your app (customize to your entrypoint)
CMD ["uvicorn", "app:combined_app", "--host", "0.0.0.0", "--port", "8000", "--workers", "4"]