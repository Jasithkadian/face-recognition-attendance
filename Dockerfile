# Use Python 3.11 slim image for containerized build
FROM python:3.11-slim

# Install minimal runtime dependencies required for OpenCV headless
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    && rm -rf /var/lib/apt/lists/*

# Set up non-root user (UID 1000) for security and Hugging Face / Render compatibility
RUN useradd -m -u 1000 user
USER user
ENV HOME=/home/user \
    PATH=/home/user/.local/bin:$PATH \
    DATA_DIR=/home/user/app/data \
    PORT=7860

WORKDIR /home/user/app

# Create writable data and models directories
RUN mkdir -p /home/user/app/data /home/user/app/models

# Copy requirements file first for fast Docker layer caching
COPY --chown=user:user requirements.txt .

# Install Python dependencies (pure wheels, instant install, zero compilation)
RUN pip install --no-cache-dir -r requirements.txt

# Copy source code into container
COPY --chown=user:user . .

# Ensure ONNX models are present in container image
RUN python download_models.py

EXPOSE 7860 10000

# Launch FastAPI app with Uvicorn, dynamically binding to $PORT (Render or Hugging Face)
CMD ["sh", "-c", "uvicorn app.main:app --host 0.0.0.0 --port ${PORT:-7860}"]
