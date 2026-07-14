FROM nvidia/cuda:13.3.0-devel-ubuntu26.04

ARG DEBIAN_FRONTEND=noninteractive
ARG CMAKE_VERSION=3.28.6


# Install SSH server and dependencies
RUN apt-get update && \
apt-get install -y openssh-server && \
apt-get clean && \
rm -rf /var/lib/apt/lists/*

# Configure SSH
RUN mkdir -p /var/run/sshd && \
sed -i 's/#PermitRootLogin prohibit-password/PermitRootLogin prohibit-password/' /etc/ssh/sshd_config && \
sed -i 's@session\s*required\s*pam_loginuid.so@session optional pam_loginuid.so@g' /etc/pam.d/sshd && \
ssh-keygen -A

# Create SSH directory for root
RUN mkdir -p /root/.ssh && \
chmod 700 /root/.ssh

# Install CMake
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        ca-certificates \
        curl \
        gnupg \
        
        software-properties-common \
    && add-apt-repository -y ppa:deadsnakes/ppa \
    && apt-get update \
    && apt-get install -y --no-install-recommends \
        build-essential \
        git \
        ninja-build \
        python3.11 \
        python3.11-dev \
        python3.11-venv \
    && curl -fsSL "https://github.com/Kitware/CMake/releases/download/v${CMAKE_VERSION}/cmake-${CMAKE_VERSION}-linux-x86_64.tar.gz" \
        | tar -xz --strip-components=1 -C /usr/local \
    && curl -fsSL https://bootstrap.pypa.io/get-pip.py -o /tmp/get-pip.py \
    && python3.11 /tmp/get-pip.py \
    && ln -sf /usr/bin/python3.11 /usr/local/bin/python3 \
    && ln -sf /usr/bin/python3.11 /usr/local/bin/python \
    && rm -rf /var/lib/apt/lists/* /tmp/get-pip.py


ENV CPATH=/usr/local/cuda-13.3/include:$CPATH
WORKDIR /workspace

CMD ["/bin/bash"]
