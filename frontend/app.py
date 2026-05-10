"""Interactive Streamlit frontend with real local model inference.

This app runs predictions directly from local checkpoints in `models/*.pt`.
"""

from __future__ import annotations

import hashlib
import json
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image
import streamlit as st
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from mlops_project.models.factory import load_checkpoint


DATA_ROOT = PROJECT_ROOT / "data" / "raw" / "kaggle_3m"
MODELS_ROOT = PROJECT_ROOT / "models"
PROCESSED_STATS_PATH = PROJECT_ROOT / "data" / "processed" / "norm_stats.json"
SAMPLE_LIMIT = 6
IMAGE_SIZE = 256


@dataclass(frozen=True)
class PredictionResult:
    """Container for one model prediction."""

    label: str
    confidence: float
    risk_score: float
    model_name: str
    latency_ms: float


def find_sample_images(limit: int = SAMPLE_LIMIT) -> list[Path]:
    """Return a small list of representative TIFF images from the raw dataset."""
    if not DATA_ROOT.exists():
        return []

    samples: list[Path] = []
    for patient_dir in sorted(path for path in DATA_ROOT.iterdir() if path.is_dir()):
        candidate = next(
            (
                image_path
                for image_path in sorted(patient_dir.glob("*.tif"))
                if not image_path.name.endswith("_mask.tif")
            ),
            None,
        )
        if candidate is not None:
            samples.append(candidate)
        if len(samples) >= limit:
            break

    return samples


def available_checkpoints() -> list[Path]:
    """Return sorted local checkpoints available for inference."""
    if not MODELS_ROOT.exists():
        return []
    return sorted(path for path in MODELS_ROOT.glob("*.pt") if path.is_file())


def load_image(path: Path) -> Image.Image:
    """Load an image from disk for previewing in Streamlit."""
    return Image.open(path)


def load_bytes_from_path(path: Path) -> bytes:
    """Read raw bytes from a local file."""
    return path.read_bytes()


def mask_path_for(image_path: Path) -> Path:
    """Return the corresponding mask path for a dataset image."""
    return image_path.with_name(f"{image_path.stem}_mask.tif")


def to_grayscale(image: Image.Image) -> Image.Image:
    """Convert an image to grayscale."""
    return image.convert("L")


@st.cache_data
def load_normalization_stats(path_str: str) -> tuple[np.ndarray, np.ndarray]:
    """Load per-channel mean/std used during training-time preprocessing."""
    path = Path(path_str)
    if not path.exists():
        raise FileNotFoundError(
            "Missing data/processed/norm_stats.json. "
            "Run `python -m mlops_project.data.prepare` first."
        )

    payload = json.loads(path.read_text())
    mean = np.array(payload["mean"], dtype=np.float32).reshape(3, 1, 1)
    std = np.array(payload["std"], dtype=np.float32).reshape(3, 1, 1)
    std = np.clip(std, 1e-6, None)
    return mean, std


@st.cache_resource
def load_model_bundle(checkpoint_path_str: str):
    """Load and cache a model checkpoint for repeated Streamlit runs."""
    model, ckpt = load_checkpoint(checkpoint_path_str, device="cpu", eval_mode=True)
    return model, ckpt


def preprocess_for_model(image: Image.Image, mean: np.ndarray, std: np.ndarray) -> torch.Tensor:
    """Convert a PIL image to the normalized tensor expected by trained models."""
    rgb = image.convert("RGB").resize((IMAGE_SIZE, IMAGE_SIZE), Image.BILINEAR)
    array = np.array(rgb, dtype=np.float32) / 255.0
    chw = np.transpose(array, (2, 0, 1))
    normalized = (chw - mean) / std
    try:
        tensor = torch.from_numpy(normalized).float().unsqueeze(0)
    except RuntimeError:
        # Fallback for environments where torch<->numpy binary compatibility is broken.
        tensor = torch.tensor(normalized.tolist(), dtype=torch.float32).unsqueeze(0)
    return tensor


def run_local_predictor(image: Image.Image, checkpoint_path: Path, threshold: float) -> PredictionResult:
    """Run one forward pass from a local checkpoint and return UI-friendly values."""
    start = time.perf_counter()
    mean, std = load_normalization_stats(str(PROCESSED_STATS_PATH))
    model, ckpt = load_model_bundle(str(checkpoint_path))
    x = preprocess_for_model(image, mean, std)

    with torch.no_grad():
        logits = model(x)
        score = float(torch.sigmoid(logits).item())

    latency_ms = (time.perf_counter() - start) * 1000.0
    label = "tumor" if score >= threshold else "no_tumor"
    confidence = score if label == "tumor" else 1.0 - score

    return PredictionResult(
        label=label,
        confidence=confidence,
        risk_score=score,
        model_name=str(ckpt.get("model_name", checkpoint_path.stem)),
        latency_ms=latency_ms,
    )


def main() -> None:
    """Render the learning-oriented frontend."""
    st.set_page_config(page_title="MLOps Brain Tumor Demo", page_icon="🧠", layout="wide")

    if "selected_sample" not in st.session_state:
        st.session_state.selected_sample = None

    st.title("MLOps Brain Tumor Frontend Demo")
    st.caption("Real local inference from checkpoints in models/*.pt")

    st.info(
        "Research/educational interface only. "
        "Predictions are not clinically validated and must not be used for diagnosis."
    )

    checkpoints = available_checkpoints()

    with st.sidebar:
        st.header("Inference Controls")
        threshold = st.slider("Decision threshold", min_value=0.05, max_value=0.95, value=0.50)

        if not checkpoints:
            st.error("No checkpoints found in models/. Add at least one .pt model file.")
            selected_checkpoint = None
        else:
            checkpoint_names = [path.name for path in checkpoints]
            selected_name = st.selectbox("Checkpoint", checkpoint_names, index=0)
            selected_checkpoint = MODELS_ROOT / selected_name

        st.divider()
        st.markdown("### How to Use")
        st.markdown("1. Select a checkpoint.")
        st.markdown("2. Upload an MRI image or choose an example.")
        st.markdown("3. Run prediction and inspect confidence + latency.")

    left_col, right_col = st.columns([1.2, 1.0])

    with left_col:
        input_mode = st.radio("Input source", ["Upload", "Examples"], horizontal=True)

        selected_example: Path | None = None
        file_bytes: bytes | None = None
        display_name = ""
        preview_image: Image.Image | None = None
        source_image: Image.Image | None = None
        mask_image: Image.Image | None = None

        if input_mode == "Upload":
            uploaded_file = st.file_uploader(
                "Upload MRI image", type=["png", "jpg", "jpeg", "tif", "tiff"], accept_multiple_files=False
            )

            if uploaded_file is None:
                st.warning("Upload an image or switch to Examples to start the demo.")
            else:
                file_bytes = uploaded_file.getvalue()
                display_name = uploaded_file.name
                source_image = Image.open(uploaded_file).convert("RGB")
                preview_image = to_grayscale(source_image)

        else:
            sample_images = find_sample_images()
            if not sample_images:
                st.error("No sample images were found in data/raw/kaggle_3m.")
            else:
                st.markdown("### Click a dataset example")
                columns = st.columns(3)
                for index, sample_path in enumerate(sample_images):
                    with columns[index % 3]:
                        sample_img = to_grayscale(load_image(sample_path))
                        st.image(sample_img, caption=sample_path.parent.name, use_container_width=True)
                        if st.button(
                            f"Use example {index + 1}",
                            key=f"sample_{sample_path.as_posix()}",
                            type="primary" if st.session_state.selected_sample == str(sample_path) else "secondary",
                        ):
                            st.session_state.selected_sample = str(sample_path)
                            st.rerun()

                if st.session_state.selected_sample:
                    selected_example = Path(st.session_state.selected_sample)
                    file_bytes = load_bytes_from_path(selected_example)
                    display_name = selected_example.name
                    source_image = load_image(selected_example).convert("RGB")
                    preview_image = to_grayscale(source_image)
                    possible_mask = mask_path_for(selected_example)
                    if possible_mask.exists():
                        mask_image = load_image(possible_mask).convert("RGB")

        if file_bytes is None or source_image is None:
            return

        st.subheader("Selected Input")
        st.write(f"File name: {display_name}")
        st.write(f"File size: {len(file_bytes)} bytes")
        st.write(f"SHA-256 (short): {hashlib.sha256(file_bytes).hexdigest()[:16]}")

        if preview_image is not None:
            st.image(preview_image, caption="Input preview (grayscale)", use_container_width=True)

        if mask_image is not None:
            st.image(mask_image, caption="Reference mask from raw data", use_container_width=True)

        predict_clicked = input_mode == "Upload" and st.button("Run prediction", type="primary")

    with right_col:
        st.subheader("Prediction")
        should_predict = predict_clicked or (input_mode == "Examples" and file_bytes is not None)

        if selected_checkpoint is None:
            st.caption("Add a checkpoint in models/ to enable local inference.")
            return

        if should_predict and file_bytes is not None and source_image is not None:
            try:
                result = run_local_predictor(
                    image=source_image,
                    checkpoint_path=selected_checkpoint,
                    threshold=float(threshold),
                )
            except FileNotFoundError as exc:
                st.error(str(exc))
                return
            except Exception as exc:
                st.error(f"Prediction failed: {exc}")
                return

            st.metric("Predicted class", result.label.upper())
            st.metric("Confidence", f"{result.confidence * 100:.1f}%")
            st.metric("Model", result.model_name)
            st.metric("Latency", f"{result.latency_ms:.1f} ms")
            st.progress(result.risk_score, text=f"Risk score: {result.risk_score:.2f}")

            if result.label == "tumor":
                st.warning(f"Model output: **TUMOR DETECTED** (confidence: {result.confidence*100:.1f}%)")
            else:
                st.info(f"Model output: **NO TUMOR** (confidence: {result.confidence*100:.1f}%)")

            if input_mode == "Examples" and selected_example is not None:
                st.caption(f"Auto-run from raw example: {selected_example.relative_to(DATA_ROOT.parent)}")

            with st.expander("Explainability Status"):
                if result.model_name == "baseline":
                    st.info(
                        "Spatial focus is not available for the baseline model. "
                        "This model uses global image statistics (mean/std per channel), "
                        "so it does not attend to specific regions."
                    )
                else:
                    st.info(
                        "Spatial focus maps (Grad-CAM) are not shown yet. "
                        "For now, trust the prediction score and confidence metric only. "
                        "Future work: implement Grad-CAM or occlusion sensitivity from model internals."
                    )

                st.markdown(
                    "Current app behavior: we show real prediction scores (class, confidence, latency). "
                    "No synthetic or computed localization maps are shown."
                )

            with st.expander("What happened behind the scenes?"):
                st.markdown(
                    "- Selected a local checkpoint from models/*.pt.\n"
                    "- Loaded training normalization stats from data/processed/norm_stats.json.\n"
                    "- Resized image to 256x256, normalized channels, and ran one forward pass.\n"
                    "- Applied sigmoid to get tumour probability and compared with threshold.\n"
                    "- Rendered prediction metrics without localization maps."
                )

        else:
            if input_mode == "Examples":
                st.caption("Choose one of the raw dataset examples to see the output.")
            else:
                st.caption("Click 'Run prediction' to generate output.")

    st.divider()
    st.subheader("Next Integration Step")
    st.code(
        """
# Optional API mode (future):
# POST /predict
# payload = uploaded image
# response = {label, confidence, latency_ms, model_version}
        """.strip(),
        language="python",
    )


if __name__ == "__main__":
    main()
