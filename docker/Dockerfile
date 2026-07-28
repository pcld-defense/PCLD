FROM nvidia/cuda:11.7.1-cudnn8-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV PIP_NO_CACHE_DIR=1

RUN apt-get update && apt-get install -y \
    python3.10 \
    python3.10-dev \
    python3-pip \
    git \
    curl \
    wget \
    build-essential \
    libcairo2-dev \
    pkg-config \
    libgl1 \
    libglib2.0-0 \
    openssh-server \
    && rm -rf /var/lib/apt/lists/*

RUN mkdir /var/run/sshd

RUN ln -sf /usr/bin/python3.10 /usr/local/bin/python \
    && ln -sf /usr/bin/pip3 /usr/local/bin/pip

WORKDIR /workspace/PCLD

COPY requirements.txt requirements-infra.txt ./

RUN python -m pip install --upgrade pip setuptools wheel

RUN python -m pip install \
    torch==1.13.1+cu117 \
    torchvision==0.14.1+cu117 \
    --extra-index-url https://download.pytorch.org/whl/cu117

RUN grep -vE '^(torch|torchvision)==' requirements.txt > /tmp/requirements-project.txt \
    && python -m pip install -r /tmp/requirements-project.txt \
    && python -m pip install -r requirements-infra.txt

COPY . .

EXPOSE 22/tcp

CMD ["/usr/sbin/sshd","-D"]
