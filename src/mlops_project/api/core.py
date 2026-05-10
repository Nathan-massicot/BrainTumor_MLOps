"""Core inference logic for FastAPI."""

import io
import json
import logging
import time
from base64 import b64decode
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from mlops_project.models.factory import load_checkpoint

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[3]
MODELS_ROOT = PROJECT_ROOT / "models"
DATA_ROOT = PROJECT_ROOT / "data"
PROCESSED_STATS_PATH = DATA_ROOT / "processed" / "norm_stats.json"

IMAGE_SIZE = 256


def get_normalization_stats() -> tuple[np.ndarray, np.ndarray]:
    """Load per-channel mean and std from training data."""
    if not PROCESSED_STATS_PATH.exists():
        raise FileNotFoundError(
            f"Missing {PROCESSED_STATS_PATH}. "
            "Run `python -m mlops_project.data.prepare` first."
        )
    payload = json.loads(PROCESSED_STATS_PATH.read_text())
    mean = np.array(payload["mean"], dtype=np.float32).reshape(3, 1, 1)
    std = np.clip(np.array(payload["std"], dtype=np.float32).reshape(3, 1, 1), 1e-6, None)
    return mean, std


def get_available_checkpoints() -> list[str]:
    """Return list of available model checkpoint filenames."""
    if not MODELS_ROOT.exists():
        return []
    return sorted([p.name for p in MODELS_ROOT.glob("*.pt") if p.is_file()])


def validate_checkpoint(checkpoint_name: str) -> Path:
    """Validate checkpoint exists and return its path."""
    checkpoint_path = MODELS_ROOT / checkpoint_name
    if not checkpoint_path.exists():
        available = get_available_checkpoints()
        raise FileNotFoundError(
            f"Checkpoint '{checkpoint_name}' not found. "
            f"Available: {available}"
        )
    return checkpoint_path


def load_model_cached(checkpoint_path: Path) -> tuple:
    """Load model from checkpoint (can be cached at application level)."""
    model, ckpt = load_checkpoint(str(checkpoint_path), device="cpu", eval_mode=True)
    return model, ckpt


def preprocess_image(image_data: bytes, mean: np.ndarray, std: np.ndarray) -> torch.Tensor:
    """
    Preprocess image bytes to model input tensor.
    
    Args:
        image_data: Raw image bytes
        mean: Per-channel normalization mean (shape 3, 1, 1)
        std: Per-channel normalization std (shape 3, 1, 1)
    
    Returns:
        Batch tensor (shape 1, 3, 256, 256)
    """
    # Load image
    image = Image.open(io.BytesIO(image_data))
    
    # Convert to RGB and resize
    rgb = image.convert("RGB").resize((IMAGE_SIZE, IMAGE_SIZE), Image.BILINEAR)
    array = np.array(rgb, dtype=np.float32) / 255.0
    
    # CHW format
    chw = np.transpose(array, (2, 0, 1))
    
    # Normalize
    normalized = (chw - mean) / std
    
    # To tensor (with fallback for torch<->numpy compatibility issues)
    try:
        return torch.from_numpy(normalized).float().unsqueeze(0)
    except RuntimeError:
        return torch.tensor(normalized.tolist(), dtype=torch.float32).unsqueeze(0)


def run_inference(
    image_bytes: bytes,
    checkpoint_path: Path,
    threshold: float = 0.5,
) -> dict:
    """
    Run inference on image with specified checkpoint.
    
    Args:
        image_bytes: Raw image bytes
        checkpoint_path: Path to checkpoint file
        threshold: Classification threshold
    
    Returns:
        Dict with label, confidence, risk_score, model_name, latency_ms
    """
    start = time.perf_counter()
    
    # Load normalization stats and model
    mean, std = get_normalization_stats()
    model, ckpt = load_model_cached(checkpoint_path)
    
    # Preprocess image
    x = preprocess_image(image_bytes, mean, std)
    
    # Forward pass
    with torch.no_grad():
        score = float(torch.sigmoid(model(x)).item())
    
    latency_ms = (time.perf_counter() - start) * 1000.0
    
    # Classify
    label = "tumor" if score >= threshold else "no_tumor"
    confidence = score if label == "tumor" else 1.0 - score
    
    return {
        "label": label,
        "confidence": confidence,
        "risk_score": score,
        "model_name": str(ckpt.get("model_name", checkpoint_path.stem)),
        "latency_ms": latency_ms,
    }
