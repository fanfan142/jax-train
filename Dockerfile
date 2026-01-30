ARG CUDA_VERSION=12.4.1
ARG MUJOCO_PLAYGROUND_REF=main

FROM nvidia/cuda:${CUDA_VERSION}-devel-ubuntu22.04 AS builder

ARG MUJOCO_PLAYGROUND_REF=main

ENV DEBIAN_FRONTEND=noninteractive \
  PIP_DISABLE_PIP_VERSION_CHECK=1 \
  PIP_NO_CACHE_DIR=1

RUN apt-get update \
  && apt-get install -y --no-install-recommends \
    software-properties-common \
    git \
    ca-certificates \
    build-essential \
    libgl1 \
    libegl1 \
    libglx-mesa0 \
    libosmesa6 \
    libglfw3 \
    libglew2.2 \
    libx11-6 \
    libxrandr2 \
    libxinerama1 \
    libxcursor1 \
    libxi6 \
  && add-apt-repository ppa:deadsnakes/ppa \
  && apt-get update \
  && apt-get install -y --no-install-recommends \
    python3.12 \
    python3.12-dev \
    python3.12-venv \
  && python3.12 -m ensurepip --upgrade \
  && python3.12 -m pip install --no-cache-dir --upgrade pip \
  && python3.12 -m venv /opt/venv \
  && rm -rf /var/lib/apt/lists/*

ENV PATH="/opt/venv/bin:${PATH}"

WORKDIR /build
COPY requirements.txt /build/requirements.txt

RUN pip install --no-cache-dir "jax[cuda12]" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html \
  && pip install --no-cache-dir -r /build/requirements.txt \
  && pip install --no-cache-dir "git+https://github.com/google-deepmind/mujoco_playground.git@${MUJOCO_PLAYGROUND_REF}"

FROM nvidia/cuda:${CUDA_VERSION}-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
  PIP_DISABLE_PIP_VERSION_CHECK=1 \
  PIP_NO_CACHE_DIR=1

RUN apt-get update \
  && apt-get install -y --no-install-recommends \
    ca-certificates \
    software-properties-common \
    git \
  && add-apt-repository ppa:deadsnakes/ppa \
  && apt-get update \
  && apt-get install -y --no-install-recommends \
    libgl1 \
    libegl1 \
    libglx-mesa0 \
    libosmesa6 \
    libglfw3 \
    libglew2.2 \
    libx11-6 \
    libxrandr2 \
    libxinerama1 \
    libxcursor1 \
    libxi6 \
    python3.12 \
    python3.12-venv \
  && update-alternatives --install /usr/bin/python python /usr/bin/python3.12 1 \
  && update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.12 1 \
  && python3.12 -m ensurepip --upgrade \
  && python3.12 -m pip install --no-cache-dir --upgrade pip \
  && apt-get purge -y --auto-remove software-properties-common \
  && rm -rf /var/lib/apt/lists/*

ENV PATH="/opt/venv/bin:${PATH}"

COPY --from=builder /opt/venv /opt/venv

WORKDIR /workspace

RUN git clone --depth 1 https://github.com/google-deepmind/mujoco_menagerie.git /opt/mujoco_menagerie

ENV XLA_FLAGS=--xla_gpu_cuda_data_dir=/usr/local/cuda \
  NVIDIA_VISIBLE_DEVICES=all \
  NVIDIA_DRIVER_CAPABILITIES=compute,utility,graphics \
  JAX_PLATFORM_NAME=gpu \
  MUJOCO_GL=egl \
  MUJOCO_MENAGERIE_PATH=/opt/mujoco_menagerie

CMD ["/bin/bash"]
