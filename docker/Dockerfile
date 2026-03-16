FROM nvcr.io/nvidia/pytorch:25.04-py3

ENV DEBIAN_FRONTEND=noninteractive
# Unset NGC pip constraints (Megatron requirement)
ENV PIP_CONSTRAINT=
# Performance-related environment variables
ENV CUDA_DEVICE_MAX_CONNECTIONS=1
ENV TORCH_EXTENSIONS_DIR=/workspace/torch_extensions

WORKDIR /workspace

# System packages for both DeepSpeed and Megatron
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    build-essential \
    libssl-dev \
    libopenmpi-dev \
    wget \
    curl \
    ca-certificates \
    ninja-build \
    python3-dev \
    && rm -rf /var/lib/apt/lists/*

# Upgrade pip and install common Python build tools
RUN pip install --upgrade pip setuptools wheel
RUN pip install "setuptools<80.0.0,>=77.0.0" "packaging>=24.2" "pybind11"

# Clone Megatron-LM
ARG MEGATRON_TAG=core_v0.16.0
RUN git clone --recursive --branch ${MEGATRON_TAG} https://github.com/NVIDIA/Megatron-LM.git /opt/Megatron-LM \
 && cd /opt/Megatron-LM \
 && git submodule update --init --recursive \
 && pip install -e .

# Transformers stack for DeepSpeed
RUN pip install \
    transformers \
    accelerate \
    datasets \
    safetensors \
    sentencepiece \
    einops

# DeepSpeed build flags
ENV DS_BUILD_OPS=1
ENV DS_BUILD_TRANSFORMER=1
ENV DS_BUILD_TRANSFORMER_INFERENCE=1
ENV DS_BUILD_CCL_COMM=0

# Install DeepSpeed and MPI support
RUN pip install --no-cache-dir deepspeed --no-build-isolation
RUN pip install mpi4py

# Optional profiling tools
RUN apt-get update && apt-get install -y \
    nvidia-cuda-dev \
    && rm -rf /var/lib/apt/lists/*
