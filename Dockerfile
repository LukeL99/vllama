FROM nvidia/cuda:13.0.2-base-ubuntu24.04

# Set an argument to prevent interactive prompts during package installation
ARG DEBIAN_FRONTEND=noninteractive

# Install system dependencies, ollama, create user, and cleanup in a single layer
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    ca-certificates \
    python3.12 \
    python3.12-venv \
    && curl -fsSL https://ollama.com/install.sh | sh \
    && (groupadd -r vllama 2>/dev/null || true) \
    && useradd -r -g vllama -d /opt/vllama -s /bin/bash vllama \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/*

# Set the working directory
WORKDIR /opt/vllama

# Copy only requirements first for better layer caching
COPY requirements.txt .

# Create venv and install dependencies in a single layer
RUN python3.12 -m venv venv312 \
    && ./venv312/bin/pip install --no-cache-dir --upgrade pip \
    && ./venv312/bin/pip install --no-cache-dir -r requirements.txt \
    && find ./venv312 -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true \
    && find ./venv312 -type f -name "*.pyc" -delete \
    && find ./venv312 -type f -name "*.pyo" -delete

# Copy application files
COPY --chown=vllama:vllama vllama.py .
COPY --chown=vllama:vllama multiuser.conf .

# Set ownership of venv
RUN chown -R vllama:vllama /opt/vllama

# Switch to the non-root user.
USER vllama

# The command to run when the container starts.
# It executes the main script using the Python interpreter from the virtual environment,
# which guarantees the correct Python version (3.12) is used.
CMD ["/opt/vllama/venv312/bin/python", "vllama.py"]
