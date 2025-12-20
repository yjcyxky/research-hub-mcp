# Custom cross-rs Dockerfile for aarch64-unknown-linux-gnu with Python 3.12
# This extends the official cross image and adds Python 3.12 support

ARG CROSS_BASE_IMAGE
FROM $CROSS_BASE_IMAGE

# Install prerequisites for adding PPA
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        software-properties-common \
        gnupg \
        curl \
        ca-certificates && \
    # Add deadsnakes PPA
    add-apt-repository -y ppa:deadsnakes/ppa && \
    # Enable arm64 architecture
    dpkg --add-architecture arm64 && \
    # Update all package lists (native and arm64)
    apt-get update && \
    # Install Python 3.12 (native) and libraries (arm64)
    apt-get install -y \
        python3.12 \
        python3.12-dev \
        python3.12-venv \
        libpython3.12-dev:arm64 \
        libssl-dev:arm64 \
        pkg-config && \
    # Cleanup
    rm -rf /var/lib/apt/lists/* && \
    # Set default python
    update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.12 1

# Set environment variables for PyO3 cross-compilation
ENV PYO3_CROSS=1
ENV PYO3_CROSS_PYTHON_VERSION=3.12
ENV PYO3_CROSS_LIB_DIR=/usr/lib/aarch64-linux-gnu


