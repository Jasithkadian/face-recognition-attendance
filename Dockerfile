# Use Python 3.11 slim image for containerized build
FROM python:3.11-slim

# Install runtime dependencies required for OpenCV headless and git (for pip install)
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    git \
    && rm -rf /var/lib/apt/lists/*

# Set up non-root user (UID 1000) for Hugging Face Spaces security requirements
RUN useradd -m -u 1000 user
USER user
ENV HOME=/home/user \
    PATH=/home/user/.local/bin:$PATH \
    DATA_DIR=/home/user/app/data \
    PORT=7860

WORKDIR /home/user/app

# Create writable data directory for SQLite database
RUN mkdir -p /home/user/app/data

# Copy requirements file first for layer caching
COPY --chown=user:user requirements.txt .

# Install Python dependencies using dlib-bin precompiled wheel
RUN pip install --no-cache-dir -r requirements.txt

# Copy source code into container
COPY --chown=user:user . .

EXPOSE 7860

# Launch FastAPI app with Uvicorn on 0.0.0.0:7860
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "7860"]
