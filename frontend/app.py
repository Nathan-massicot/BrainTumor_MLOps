"""Interactive Streamlit frontend with a mock predictor.

This app is designed for issue #33 so you can build and learn in parallel
while the real model/API is still being implemented.
"""

from __future__ import annotations

import hashlib
import random
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image
import streamlit as st


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = PROJECT_ROOT / "data" / "raw" / "kaggle_3m"
SAMPLE_LIMIT = 6


@dataclass(frozen=True)
class MockPrediction:
    """Container for a fake prediction result."""

    label: str
    confidence: float
    risk_score: float


def _stable_random_value(image_bytes: bytes, seed: int) -> float:
    """Return a deterministic pseudo-random value in [0, 1]."""
    digest = hashlib.sha256(image_bytes).hexdigest()
    digest_seed = int(digest[:8], 16) + seed
    rng = random.Random(digest_seed)
    return rng.random()


def run_mock_predictor(
    image_bytes: bytes,
    threshold: float,
    tumor_bias: float,
    seed: int,
) -> MockPrediction:
    """Generate a fake model output for UI and integration testing.

    Args:
        image_bytes: Raw uploaded file bytes.
        threshold: Decision boundary between no-tumor and tumor.
        tumor_bias: Positive values increase tumor probability.
        seed: Manual seed for reproducibility demos.

    Returns:
        MockPrediction: Predicted class and confidence-like score.
    """
    base_score = _stable_random_value(image_bytes=image_bytes, seed=seed)
    shifted_score = min(1.0, max(0.0, base_score + tumor_bias))

    label = "tumor" if shifted_score >= threshold else "no_tumor"
    confidence = shifted_score if label == "tumor" else 1.0 - shifted_score

    return MockPrediction(
        label=label,
        confidence=confidence,
        risk_score=shifted_score,
    )


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


def generate_mock_attention(
    image: Image.Image, prediction: MockPrediction, seed: int = 42
) -> np.ndarray:
    """Generate a mock attention heatmap showing where the model 'focuses'.
    
    Args:
        image: The input MRI image.
        prediction: The model prediction result.
        seed: Seed for reproducible heatmap generation.
    
    Returns:
        A numpy array (H, W) with values in [0, 1] representing attention intensity.
    """
    rng = random.Random(seed + int(prediction.risk_score * 1000))
    width, height = image.size

    center_x = rng.uniform(width * 0.3, width * 0.7)
    center_y = rng.uniform(height * 0.3, height * 0.7)
    sigma = rng.uniform(min(height, width) * 0.15, min(height, width) * 0.3)

    x = np.linspace(0, width - 1, width)
    y = np.linspace(0, height - 1, height)
    xx, yy = np.meshgrid(x, y)

    heatmap = np.exp(-((xx - center_x) ** 2 + (yy - center_y) ** 2) / (2 * sigma ** 2))
    heatmap = heatmap / (heatmap.max() + 1e-8)
    heatmap = heatmap * prediction.confidence

    return heatmap


def main() -> None:
    """Render the learning-oriented frontend."""
    st.set_page_config(page_title="MLOps Brain Tumor Demo", page_icon="🧠", layout="wide")

    if "selected_sample" not in st.session_state:
        st.session_state.selected_sample = None

    st.title("MLOps Brain Tumor Frontend Demo")
    st.caption("Issue #33: interactive Streamlit UI with a mock model response")

    st.info(
        "This is a learning and integration scaffold. "
        "Predictions are mocked and are not medically valid."
    )

    with st.sidebar:
        st.header("Mock Controls")
        threshold = st.slider("Decision threshold", min_value=0.05, max_value=0.95, value=0.50)
        tumor_bias = st.slider(
            "Tumor bias",
            min_value=-0.40,
            max_value=0.40,
            value=0.00,
            help="Shift scores to simulate easier/harder positive detection.",
        )
        seed = st.number_input("Reproducibility seed", min_value=0, max_value=9999, value=42)

        st.divider()
        st.markdown("### How to Learn")
        st.markdown("1. Upload an MRI image.")
        st.markdown("2. Or choose a raw dataset example.")
        st.markdown("3. Change threshold and bias.")
        st.markdown("4. Click predict and inspect outputs.")
        st.markdown("5. Repeat with the same image to see deterministic behavior.")

    left_col, right_col = st.columns([1.2, 1.0])

    with left_col:
        input_mode = st.radio("Input source", ["Upload", "Examples"], horizontal=True)

        uploaded_file = None
        selected_example: Path | None = None
        file_bytes: bytes | None = None
        display_name = ""
        preview_image = None
        mask_image = None

        if input_mode == "Upload":
            uploaded_file = st.file_uploader(
                "Upload MRI image", type=["png", "jpg", "jpeg", "tif", "tiff"], accept_multiple_files=False
            )

            if uploaded_file is None:
                st.warning("Upload an image or switch to Examples to start the demo.")
            else:
                file_bytes = uploaded_file.getvalue()
                display_name = uploaded_file.name
                preview_image = to_grayscale(Image.open(uploaded_file))

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
                    preview_image = to_grayscale(load_image(selected_example))
                    possible_mask = mask_path_for(selected_example)
                    if possible_mask.exists():
                        mask_image = to_grayscale(load_image(possible_mask))

        if file_bytes is None:
            return

        st.subheader("Selected Input")
        st.write(f"File name: {display_name}")
        st.write(f"File size: {len(file_bytes)} bytes")
        st.write(f"SHA-256 (short): {hashlib.sha256(file_bytes).hexdigest()[:16]}")
        if preview_image is not None:
            st.image(preview_image, caption="Input preview (grayscale)", use_container_width=True)

        if mask_image is not None:
            st.image(mask_image, caption="Reference mask from raw data (grayscale)", use_container_width=True)

        predict_clicked = input_mode == "Upload" and st.button("Run mock prediction", type="primary")

    with right_col:
        st.subheader("Prediction")
        should_predict = predict_clicked or (input_mode == "Examples" and file_bytes is not None)

        if should_predict and file_bytes is not None:
            result = run_mock_predictor(
                image_bytes=file_bytes,
                threshold=float(threshold),
                tumor_bias=float(tumor_bias),
                seed=int(seed),
            )

            st.metric("Predicted class", result.label.upper())
            st.metric("Confidence", f"{result.confidence * 100:.1f}%")
            st.progress(result.risk_score, text=f"Risk score: {result.risk_score:.2f}")

            # Neutral black and white result display
            if result.label == "tumor":
                st.warning(f"⚠️ Model output: **TUMOR DETECTED** (confidence: {result.confidence*100:.1f}%)")
            else:
                st.info(f"✓ Model output: **NO TUMOR** (confidence: {result.confidence*100:.1f}%)")

            if input_mode == "Examples" and selected_example is not None:
                st.caption(f"Auto-run from raw example: {selected_example.relative_to(DATA_ROOT.parent)}")

            # Add attention visualization
            with st.expander("📍 Model Focus Map (Attention Visualization)"):
                st.markdown(
                    f"This visualization shows where the model focuses to make its prediction. "
                    f"**Confidence: {result.confidence*100:.1f}%**"
                )

                if preview_image is not None:
                    attention = generate_mock_attention(
                        preview_image, result, seed=int(seed)
                    )

                    img_array = np.array(preview_image).astype(float) / 255.0
                    attention_normalized = attention / (attention.max() + 1e-6)
                    overlay = img_array * (1 - attention_normalized * 0.4) + attention_normalized * 0.6
                    overlay = (overlay * 255).astype(np.uint8)

                    col1, col2 = st.columns(2)
                    with col1:
                        st.image(preview_image, caption="Original Image (B&W)", use_container_width=True)
                    with col2:
                        st.image(overlay, caption="Model Focus Map", use_container_width=True)
                    
                    st.markdown(
                        f"The focus map highlights the regions contributing to the **{result.label.upper()}** prediction "
                        f"with {result.confidence*100:.1f}% confidence."
                    )

            with st.expander("What happened behind the scenes?"):
                st.markdown(
                    "- We hashed the uploaded bytes for deterministic scoring.\n"
                    "- We applied a user-controlled bias to simulate model behavior.\n"
                    "- We compared score vs threshold to choose a class.\n"
                    "- We generated a mock attention map to visualize model focus.\n"
                    "- This is a mock pipeline to unblock UI development."
                )

        else:
            if input_mode == "Examples":
                st.caption("Choose one of the raw dataset examples to see the output.")
            else:
                st.caption("Click 'Run mock prediction' to generate output.")

    st.divider()
    st.subheader("Next Integration Step")
    st.code(
        """
# Replace run_mock_predictor(...) with real API call:
# POST /predict
# payload = uploaded image
# response = {label, confidence, latency_ms, model_version}
        """.strip(),
        language="python",
    )


if __name__ == "__main__":
    main()
# 