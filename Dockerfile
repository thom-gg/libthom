# Tools to build and run libthom, without the sources: they are mounted in the container (locally, or on Modal
# with tools/modal_shell.py) and compiled inside it. CUDA toolkit, CMake and Python with matplotlib.
# The nvidia/cuda images exist for linux/amd64 and linux/arm64: the image builds natively on Apple Silicon
# (nvcc doesn't need a GPU to compile), and Modal builds it for linux/amd64.
# The driver of the GPU machine must support the CUDA version (nvidia-smi shows the max one).
ARG CUDA_VERSION=13.3.0
ARG UBUNTU_VERSION=26.04

FROM nvidia/cuda:${CUDA_VERSION}-devel-ubuntu${UBUNTU_VERSION}

ARG DEBIAN_FRONTEND=noninteractive

# CMake from apt instead of the Kitware x86_64 tarball, so that it works on both architectures.
# matplotlib from apt too (for tools/plot.py), pip refuses to install into the system Python of recent Ubuntus
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        build-essential \
        cmake \
        ninja-build \
        git \
        python3 \
        python3-matplotlib \
    && rm -rf /var/lib/apt/lists/*

# The routines' .cpp files include CUDA headers but are compiled by the host compiler, not nvcc
ENV CPATH=/usr/local/cuda/include
WORKDIR /workspace

CMD ["/bin/bash"]

