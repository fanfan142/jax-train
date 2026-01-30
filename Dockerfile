FROM nvidia/cuda:12.4.1-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update \
  && apt-get install -y --no-install-recommends \
    software-properties-common \
    git \
    curl \
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
    python3.12-distutils \
    python3.12-venv \
  && curl -sS https://bootstrap.pypa.io/get-pip.py | python3.12 \
  && update-alternatives --install /usr/bin/python python /usr/bin/python3.12 1 \
  && update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.12 1 \
  && python -m pip install --no-cache-dir --upgrade pip \
  && rm -rf /var/lib/apt/lists/*

WORKDIR /workspace

COPY requirements.txt /workspace/requirements.txt

RUN python -m pip install --no-cache-dir "jax[cuda12]" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html \
  && python -m pip install --no-cache-dir -r /workspace/requirements.txt \
  && python -m pip install --no-cache-dir git+https://github.com/deepmind/mujoco_playground.git

ENV XLA_FLAGS=--xla_gpu_cuda_data_dir=/usr/local/cuda \
  NVIDIA_VISIBLE_DEVICES=all \
  NVIDIA_DRIVER_CAPABILITIES=compute,utility \
  JAX_PLATFORM_NAME=gpu

CMD ["/bin/bash"]
