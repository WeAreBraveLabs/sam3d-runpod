# SAM 3D Objects RunPod Serverless
# Meta's SAM 3D for image-to-3D generation
# - Uses RunPod's built-in model caching
# - Requires 32GB+ VRAM (A100/H100 recommended)

FROM runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04

ENV PYTHONUNBUFFERED=1
ENV PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
ENV HF_HOME=/runpod-volume/huggingface-cache
ENV CUDA_HOME=/usr/local/cuda
ENV PATH="${CUDA_HOME}/bin:${PATH}"
ENV LD_LIBRARY_PATH="${CUDA_HOME}/lib64:${LD_LIBRARY_PATH}"
# For kaolin
ENV TORCH_CUDA_ARCH_LIST="8.0;9.0"
# Skip conda init in inference.py
ENV LIDRA_SKIP_INIT=true

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libjpeg-dev \
    libpng-dev \
    ninja-build \
    git \
    wget \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

# Upgrade pip and install build tools
RUN pip install --no-cache-dir --upgrade pip setuptools wheel ninja

# Base image already has PyTorch 2.4.0 with CUDA 12.4

# Install flash-attn from pre-built wheel (avoids 30+ min compilation)
RUN pip install --no-cache-dir flash-attn --no-build-isolation \
    -f https://github.com/Dao-AILab/flash-attention/releases/expanded_assets/v2.8.3

# Install pytorch3d from pre-built wheel
RUN pip install --no-cache-dir pytorch3d \
    -f https://dl.fbaipublicfiles.com/pytorch3d/packaging/wheels/py311_cu124_pyt240/download.html

# Install kaolin (required for 3D visualization and rendering)
RUN pip install --no-cache-dir kaolin==0.17.0 -f https://nvidia-kaolin.s3.us-east-2.amazonaws.com/torch-2.4.0_cu124.html

# Install gsplat (specific commit for SAM3D compatibility)
RUN pip install --no-cache-dir \
    "gsplat @ git+https://github.com/nerfstudio-project/gsplat.git@2323de5905d5e90e035f792fe65bad0fedd413e7"

# Install MoGe (depth estimation)
RUN pip install --no-cache-dir \
    "MoGe @ git+https://github.com/microsoft/MoGe.git@a8c37341bc0325ca99b9d57981cc3bb2bd3e255b"

# Install core Python dependencies from SAM3D requirements
RUN pip install --no-cache-dir \
    runpod \
    huggingface_hub \
    safetensors \
    hydra-core==1.3.2 \
    omegaconf \
    loguru \
    einops \
    einops-exts \
    easydict \
    trimesh \
    open3d==0.18.0 \
    opencv-python-headless \
    pillow \
    numpy \
    scipy \
    scikit-image \
    seaborn==0.13.2 \
    matplotlib \
    tqdm \
    transformers \
    accelerate \
    timm==0.9.16 \
    kornia \
    roma==1.5.1 \
    xformers \
    spconv-cu120==2.3.8 \
    point-cloud-utils \
    pymeshfix \
    pyrender \
    xatlas

# Clone SAM 3D Objects repository
RUN git clone https://github.com/facebookresearch/sam-3d-objects.git /app/sam-3d-objects

# Install sam3d_objects package
WORKDIR /app/sam-3d-objects
RUN pip install --no-cache-dir -e .

# Apply hydra patch (required for SAM3D)
RUN python patching/hydra

WORKDIR /app

# Set Python path
ENV PYTHONPATH="/app/sam-3d-objects:/app${PYTHONPATH:+:$PYTHONPATH}"

# Copy handler
COPY handler.py /app/handler.py

CMD ["python3", "-u", "handler.py"]
