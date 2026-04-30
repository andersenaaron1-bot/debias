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
COPY pyproject.toml README.md /workspace/
COPY src /workspace/src
COPY configs /workspace/configs
COPY docs /workspace/docs

RUN python -m pip install --upgrade pip setuptools wheel && \
    python -m pip install -r /tmp/requirements-cluster.txt && \
    python -m pip install -e ".[mech]" && \
    python -m pip install --force-reinstall --no-deps \
      --index-url https://download.pytorch.org/whl/cu121 \
      torchvision==0.19.1 && \
    python -c "import torch, torchvision, transformers, sae_lens, aisafety; print('torch', torch.__version__); print('torchvision', torchvision.__version__); print('sae_lens ok')"

RUN mkdir -p /cache/huggingface /workspace/outputs

ENTRYPOINT ["python", "-m", "aisafety.scripts.run_experiment_config"]
