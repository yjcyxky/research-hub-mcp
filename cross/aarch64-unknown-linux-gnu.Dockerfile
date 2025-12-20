# Custom cross-rs Dockerfile for aarch64-unknown-linux-gnu with Python 3.12
# This extends the official cross image and adds Python 3.12 support

ARG CROSS_BASE_IMAGE
FROM $CROSS_BASE_IMAGE

# Install prerequisites
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        software-properties-common \
        gnupg \
        curl \
        ca-certificates && \
    # Configure ports for arm64
    dpkg --add-architecture arm64 && \
    # Add deadsnakes PPA
    add-apt-repository -y ppa:deadsnakes/ppa && \
    # Fix sources for arm64: The default sources.list only has x86_64 repositories.
    # We need to add ports.ubuntu.com for arm64 packages.
    sed -i 's/deb http:\/\/archive.ubuntu.com\/ubuntu\//deb [arch=amd64] http:\/\/archive.ubuntu.com\/ubuntu\//g' /etc/apt/sources.list && \
    sed -i 's/deb http:\/\/security.ubuntu.com\/ubuntu\//deb [arch=amd64] http:\/\/security.ubuntu.com\/ubuntu\//g' /etc/apt/sources.list && \
    echo "deb [arch=arm64] http://ports.ubuntu.com/ubuntu-ports/ focal main restricted universe multiverse" >> /etc/apt/sources.list && \
    echo "deb [arch=arm64] http://ports.ubuntu.com/ubuntu-ports/ focal-updates main restricted universe multiverse" >> /etc/apt/sources.list && \
    echo "deb [arch=arm64] http://ports.ubuntu.com/ubuntu-ports/ focal-security main restricted universe multiverse" >> /etc/apt/sources.list && \
    # Update all package lists
    apt-get update && \
    # Install Python 3.12 (native) and libraries
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
