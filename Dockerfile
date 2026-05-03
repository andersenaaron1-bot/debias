FROM pytorch/pytorch:2.4.1-cuda12.1-cudnn9-runtime

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
    && rm -rf /var/lib/apt/lists/*

COPY requirements/cluster.txt /tmp/requirements-cluster.txt
COPY requirements/torch-cu121-constraints.txt /tmp/torch-cu121-constraints.txt
COPY pyproject.toml README.md /workspace/
COPY src /workspace/src
COPY configs /workspace/configs

RUN python -m pip install --upgrade pip setuptools wheel && \
    python -m pip install --constraint /tmp/torch-cu121-constraints.txt -r /tmp/requirements-cluster.txt && \
    python -m pip install --constraint /tmp/torch-cu121-constraints.txt ".[mech]" && \
    (python -m pip uninstall -y torchvision || true) && \
    python -c "import torch, transformers, sae_lens, aisafety; print('torch', torch.__version__, 'cuda', torch.version.cuda); assert torch.__version__.startswith('2.4.1'); assert torch.version.cuda == '12.1'; print('sae_lens ok')"

RUN mkdir -p /cache/huggingface /workspace/outputs
