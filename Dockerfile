FROM nvidia/cuda:11.7.1-cudnn8-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV PIP_NO_CACHE_DIR=1

WORKDIR /workspace/PCLD

RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.10 \
    python3.10-dev \
    python3-pip \
    python3.10-venv \
    build-essential \
    pkg-config \
    libcairo2-dev \
    libgirepository1.0-dev \
    git \
    && rm -rf /var/lib/apt/lists/*

RUN python3.10 -m pip install --upgrade pip

COPY requirements-lock.txt .

RUN python3.10 -m pip install \
    --extra-index-url https://download.pytorch.org/whl/cu117 \
    -r requirements-lock.txt

COPY . .

CMD ["/bin/bash"]
