FROM python:3.11-slim-bookworm

ENV DEBIAN_FRONTEND=noninteractive \
    PIP_NO_CACHE_DIR=1 \
    PYTHONUNBUFFERED=1 \
    HF_HOME=/cache/huggingface \
    TRANSFORMERS_CACHE=/cache/huggingface/transformers \
    HF_DATASETS_CACHE=/cache/huggingface/datasets

WORKDIR /workspace

RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    git-lfs \
    libglib2.0-0 \
    libgomp1 \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

COPY requirements/cluster.txt /tmp/requirements-cluster.txt
COPY requirements/sae-container-constraints.txt /tmp/sae-container-constraints.txt
COPY pyproject.toml README.md /workspace/
COPY src /workspace/src
COPY configs /workspace/configs

RUN python -m pip install --upgrade pip setuptools wheel && \
    python -m pip install --index-url https://download.pytorch.org/whl/cu118 torch==2.6.0 && \
    python -m pip install --constraint /tmp/sae-container-constraints.txt -r /tmp/requirements-cluster.txt && \
    python -m pip install --constraint /tmp/sae-container-constraints.txt ".[mech]" && \
    (python -m pip uninstall -y torchvision || true) && \
    python -c "import torch, transformers, sae_lens, aisafety; print('torch', torch.__version__, 'cuda', torch.version.cuda); assert torch.__version__.startswith('2.6.0'); assert torch.version.cuda == '11.8'; print('sae_lens ok')"

RUN mkdir -p /cache/huggingface /workspace/outputs
