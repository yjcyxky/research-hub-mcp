# Custom cross-rs Dockerfile for aarch64-unknown-linux-gnu with Python 3.12
# This extends the official cross image and adds Python 3.12 support

ARG CROSS_BASE_IMAGE
FROM $CROSS_BASE_IMAGE

# Install Python 3.12 from deadsnakes PPA
# Note: Must combine add-apt-repository and apt-get update in same RUN to work correctly
RUN apt-get update && \
    apt-get install -y software-properties-common && \
    add-apt-repository -y ppa:deadsnakes/ppa && \
    dpkg --add-architecture arm64 && \
    apt-get update && \
    apt-get install -y \
        python3.12 \
        python3.12-dev \
        libssl-dev:arm64 \
        pkg-config && \
    rm -rf /var/lib/apt/lists/* && \
    update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.12 1

# Set environment variables for PyO3 cross-compilation
ENV PYO3_CROSS=1
ENV PYO3_CROSS_PYTHON_VERSION=3.12

