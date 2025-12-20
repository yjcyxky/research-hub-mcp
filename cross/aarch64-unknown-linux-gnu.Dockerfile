# Custom cross-rs Dockerfile for aarch64-unknown-linux-gnu with Python 3.12
# This extends the official cross image and adds Python 3.12 support

ARG CROSS_BASE_IMAGE
FROM $CROSS_BASE_IMAGE

# Install software-properties-common for add-apt-repository
RUN apt-get update && apt-get install -y software-properties-common

# Add deadsnakes PPA for Python 3.12
RUN add-apt-repository -y ppa:deadsnakes/ppa

# Install Python 3.12 and development headers
RUN dpkg --add-architecture arm64 && \
    apt-get update && \
    apt-get install -y \
        python3.12 \
        python3.12-dev \
        python3.12-venv \
        libssl-dev:arm64 \
        pkg-config \
    && rm -rf /var/lib/apt/lists/*

# Set Python 3.12 as the default python3
RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.12 1

# Set environment variables for PyO3 cross-compilation
ENV PYO3_CROSS=1
ENV PYO3_CROSS_PYTHON_VERSION=3.12
