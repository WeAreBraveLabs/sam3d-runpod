"""
SAM 3D Objects RunPod Serverless Handler
API compatible with mockupWebsite frontend

Usage:
    POST /run with JSON body:
    {
        "input": {
            "image": "base64...",
            "mask": "base64..." (optional),
            "resolution": 512,
            "texture_size": 1024,
            "output_format": "glb",
            "seed": 42
        }
    }
"""

import runpod
import os
import sys
import base64
import tempfile
import time
import torch
import requests
import numpy as np
from io import BytesIO
from PIL import Image

# Add SAM3D to path
SAM3D_PATH = os.environ.get("SAM3D_PATH", "/app/sam-3d-objects")
sys.path.insert(0, SAM3D_PATH)

# RunPod model cache configuration
CACHE_DIR = "/runpod-volume/huggingface-cache/hub"
HF_MODEL_ID = "facebook/sam-3d-objects"

# Global pipeline - initialized once at worker startup
pipeline = None


def find_cached_model_path(model_name: str) -> str | None:
    """Find model path in RunPod's cache directory."""
    cache_name = model_name.replace("/", "--")
    snapshots_dir = os.path.join(CACHE_DIR, f"models--{cache_name}", "snapshots")

    if os.path.exists(snapshots_dir):
        snapshots = os.listdir(snapshots_dir)
        if snapshots:
            return os.path.join(snapshots_dir, snapshots[0])
    return None


def download_checkpoints():
    """Download SAM3D checkpoints from HuggingFace."""
    from huggingface_hub import snapshot_download

    cached_path = find_cached_model_path(HF_MODEL_ID)
    if cached_path:
        print(f"Found cached model at: {cached_path}")
        return cached_path

    print(f"Downloading model from HuggingFace: {HF_MODEL_ID}")
    local_dir = os.path.join(SAM3D_PATH, "checkpoints", "hf")

    # Download to checkpoints directory
    snapshot_download(
        repo_id=HF_MODEL_ID,
        local_dir=local_dir,
        local_dir_use_symlinks=False,
    )

    # Move checkpoints to expected location
    ckpt_src = os.path.join(local_dir, "checkpoints")
    if os.path.exists(ckpt_src):
        import shutil
        for item in os.listdir(ckpt_src):
            src = os.path.join(ckpt_src, item)
            dst = os.path.join(local_dir, item)
            if os.path.exists(dst):
                if os.path.isdir(dst):
                    shutil.rmtree(dst)
                else:
                    os.remove(dst)
            shutil.move(src, dst)
        os.rmdir(ckpt_src)

    return local_dir


def load_model():
    """Load SAM3D model into GPU memory."""
    global pipeline

    if pipeline is not None:
        return pipeline

    print("Loading SAM 3D Objects model...")
    start_time = time.time()

    # Download checkpoints if needed
    checkpoints_dir = download_checkpoints()

    # Import inference after setting up paths
    sys.path.append(os.path.join(SAM3D_PATH, "notebook"))
    from inference import Inference

    # Load pipeline
    config_path = os.path.join(checkpoints_dir, "pipeline.yaml")
    if not os.path.exists(config_path):
        # Try alternative path
        config_path = os.path.join(SAM3D_PATH, "checkpoints", "hf", "pipeline.yaml")

    print(f"Loading pipeline from: {config_path}")
    pipeline = Inference(config_path, compile=False)

    elapsed = time.time() - start_time
    print(f"Model loaded in {elapsed:.2f}s")

    return pipeline


def decode_base64_image(data: str) -> np.ndarray:
    """Decode base64 image string to numpy array."""
    # Remove data URL prefix if present
    if "," in data:
        data = data.split(",", 1)[1]
    image_data = base64.b64decode(data)
    image = Image.open(BytesIO(image_data))
    return np.array(image)


def download_image(url: str) -> np.ndarray:
    """Download image from URL."""
    response = requests.get(url, timeout=60)
    response.raise_for_status()
    image = Image.open(BytesIO(response.content))
    return np.array(image)


def decode_base64_mask(data: str) -> np.ndarray:
    """Decode base64 mask string to numpy array."""
    if "," in data:
        data = data.split(",", 1)[1]
    mask_data = base64.b64decode(data)
    mask = Image.open(BytesIO(mask_data))
    mask_array = np.array(mask)
    # Convert to binary mask
    if mask_array.ndim == 3:
        mask_array = mask_array[..., -1]  # Use alpha or last channel
    return (mask_array > 0).astype(np.uint8) * 255


def render_thumbnail(glb_object, size: tuple = (256, 256)) -> str:
    """Render a thumbnail from the GLB object."""
    try:
        # Try to render from gaussian splat if available
        # For now, return empty string - thumbnail generation requires additional setup
        return ""
    except Exception as e:
        print(f"Thumbnail generation failed: {e}")
        return ""


def handler(job):
    """
    RunPod serverless handler for SAM 3D Objects.
    API compatible with mockupWebsite frontend.

    Input parameters:
    - image: str - Base64 encoded image or URL (required)
    - mask: str - Base64 encoded mask (optional, embedded in alpha if not provided)
    - resolution: int - Output resolution (default: 512)
    - texture_size: int - Texture resolution: 512, 1024, 2048 (default: 1024)
    - output_format: str - "glb" or "obj" (default: "glb")
    - seed: int - Random seed (default: 42)

    Returns (matching SAM3DGenerateResponse):
    - model: str - Base64 encoded GLB/OBJ
    - thumbnail: str - Base64 encoded PNG thumbnail
    - metadata: object - Generation metadata
    """
    job_input = job["input"]
    start_time = time.time()

    # Get image - support both 'image' and 'input_image'
    image_data = job_input.get("image") or job_input.get("input_image")
    if not image_data:
        return {"error": "image is required"}

    # Load model (cached after first call)
    runpod.serverless.progress_update(job, "Loading model...")
    pipe = load_model()

    # Load image
    runpod.serverless.progress_update(job, "Processing image...")
    try:
        if isinstance(image_data, str) and image_data.startswith(("http://", "https://")):
            image = download_image(image_data)
        else:
            image = decode_base64_image(image_data)
    except Exception as e:
        return {"error": f"Failed to load image: {str(e)}"}

    # Load mask if provided
    mask = None
    mask_data = job_input.get("mask")
    if mask_data:
        try:
            mask = decode_base64_mask(mask_data)
        except Exception as e:
            print(f"Warning: Failed to load mask: {e}")

    # Ensure image has alpha channel if no mask provided
    if mask is None and image.shape[-1] == 3:
        # Add full alpha channel
        alpha = np.ones((*image.shape[:2], 1), dtype=np.uint8) * 255
        image = np.concatenate([image, alpha], axis=-1)

    print(f"Image shape: {image.shape}")
    if mask is not None:
        print(f"Mask shape: {mask.shape}")

    # Extract parameters with defaults
    texture_size = job_input.get("texture_size", 1024)
    output_format = job_input.get("output_format", "glb")
    seed = job_input.get("seed", 42)

    # Run inference
    runpod.serverless.progress_update(job, "Generating 3D model...")
    print(f"Running inference: texture_size={texture_size}, seed={seed}")

    try:
        # SAM3D inference
        output = pipe._pipeline.run(
            image,
            mask,
            seed=seed,
            stage1_only=False,
            with_mesh_postprocess=True,
            with_texture_baking=True,
            use_vertex_color=False,
        )
    except Exception as e:
        return {"error": f"Inference failed: {str(e)}"}

    inference_time = time.time() - start_time
    print(f"Inference completed in {inference_time:.2f}s")

    # Export mesh
    runpod.serverless.progress_update(job, "Exporting 3D model...")
    try:
        glb = output.get("glb")
        if glb is None:
            return {"error": "Model generation failed - no GLB output"}

        # Export to temp file
        with tempfile.NamedTemporaryFile(suffix=f".{output_format}", delete=False) as f:
            model_path = f.name

        glb.export(model_path)

        with open(model_path, "rb") as f:
            model_base64 = base64.b64encode(f.read()).decode("utf-8")

        os.unlink(model_path)

        # Get mesh info
        mesh = output.get("mesh")
        triangle_count = 0
        if mesh is not None and len(mesh) > 0:
            try:
                triangle_count = mesh[0].faces.shape[0]
            except:
                pass

        # Generate thumbnail
        runpod.serverless.progress_update(job, "Generating thumbnail...")
        thumbnail_base64 = ""
        try:
            # Use input image as thumbnail placeholder
            thumb_img = Image.fromarray(image[..., :3] if image.shape[-1] == 4 else image)
            thumb_img.thumbnail((256, 256))
            thumb_buffer = BytesIO()
            thumb_img.save(thumb_buffer, format="PNG")
            thumbnail_base64 = base64.b64encode(thumb_buffer.getvalue()).decode("utf-8")
        except Exception as e:
            print(f"Thumbnail generation failed: {e}")

    except Exception as e:
        return {"error": f"Export failed: {str(e)}"}

    total_time = time.time() - start_time
    model_size_mb = len(model_base64) * 3 / 4 / 1024 / 1024
    print(f"Export completed. Model size: ~{model_size_mb:.2f}MB, Total time: {total_time:.2f}s")

    # Return in format expected by website
    return {
        "model": model_base64,
        "thumbnail": thumbnail_base64,
        "metadata": {
            "triangle_count": triangle_count,
            "texture_resolution": f"{texture_size}x{texture_size}",
            "generation_time_ms": int(total_time * 1000),
        },
    }


# Initialize model at worker startup (RunPod best practice)
print("=" * 50)
print("SAM 3D Objects RunPod Worker Starting...")
print("=" * 50)

try:
    load_model()
    print("Model loaded successfully!")
except Exception as e:
    print(f"Warning: Model pre-load failed: {e}")
    print("Model will be loaded on first request")

# Start RunPod serverless worker
runpod.serverless.start({"handler": handler})
