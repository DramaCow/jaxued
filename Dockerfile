FROM nvidia/cuda:12.5.0-devel-ubuntu22.04

ENV CUDA_PATH /usr/local/cuda
ENV CUDA_INCLUDE_PATH /usr/local/cuda/include
ENV CUDA_LIBRARY_PATH /usr/local/cuda/lib64

# Set timezone
ENV TZ=Europe/London DEBIAN_FRONTEND=noninteractive

# Add old libraries (Python 3.11) to Ubuntu 22.04
RUN apt update
RUN apt install -y software-properties-common && add-apt-repository ppa:deadsnakes/ppa
RUN apt install -y \
    git \
    python3.11 \
    python3-pip \
    python3.11-venv \
    python3-setuptools \
    python3-wheel

# Create local user
# https://jtreminio.com/blog/running-docker-containers-as-current-host-user/
ARG UID
ARG GID
RUN if [ ${UID:-0} -ne 0 ] && [ ${GID:-0} -ne 0 ]; then \
    groupadd -g ${GID} duser &&\
    useradd -l -u ${UID} -g duser duser &&\
    install -d -m 0755 -o duser -g duser /home/duser &&\
    chown --changes --silent --no-dereference --recursive ${UID}:${GID} /home/duser \
    ;fi

USER duser
WORKDIR /home/duser

# Install Python packages
ENV PATH="/home/duser/.local/bin:$PATH"
RUN python3.11 -m pip install --upgrade pip
RUN python3.11 -m pip install tensorrt
ARG REQS
RUN python3.11 -m pip install $REQS -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
RUN python3.11 -m pip install git+https://github.com/MichaelTMatthews/Craftax.git@main
RUN python3.11 -m pip install git+https://github.com/DramaCow/jaxued.git@main

WORKDIR /home/duser/uedfomo