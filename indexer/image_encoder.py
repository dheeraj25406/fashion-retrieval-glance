"""
image_encoder.py

Third stage of the fashion retrieval indexing pipeline. Recursively loads
every image under `data/raw/`, encodes it into a normalized OpenCLIP
embedding, and saves:
    - data/processed/image_embeddings.npy  (float32 array, shape [N, D])
    - data/processed/image_paths.csv       (image_path column, row-aligned
                                             with the embeddings array)

This module only generates and saves image embeddings. FAISS indexing and
retrieval logic live in separate modules.

Usage:
    python3 -m indexer.image_encoder
"""

from __future__ import annotations

import csv
import logging
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import open_clip
import torch
from PIL import Image, UnidentifiedImageError
from tqdm import tqdm

MODEL_NAME = "ViT-B-32"
PRETRAINED_TAG = "laion2b_s34b_b79k"
RAW_DATA_DIR = Path("data/raw")
EMBEDDINGS_OUTPUT_PATH = Path("data/processed/image_embeddings.npy")
IMAGE_PATHS_OUTPUT_PATH = Path("data/processed/image_paths.csv")
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png"}

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)


def get_device() -> torch.device:
    """Return the Apple MPS device if available, otherwise fall back to CPU."""
    if torch.backends.mps.is_available():
        logger.info("MPS backend available - using Apple GPU.")
        device = torch.device("mps")
    else:
        logger.info("MPS backend not available - using CPU.")
        device = torch.device("cpu")
    logger.info("Using device: %s", device)
    return device


def load_model(device: torch.device) -> Tuple[torch.nn.Module, "open_clip.transform.PreprocessCfg"]:
    """Load the OpenCLIP model and its inference-time image preprocessing transform."""
    logger.info("Loading OpenCLIP model '%s' (pretrained='%s')...", MODEL_NAME, PRETRAINED_TAG)
    model, _, preprocess_val = open_clip.create_model_and_transforms(
        MODEL_NAME, pretrained=PRETRAINED_TAG, device=device
    )
    model.eval()
    logger.info("Model loaded successfully.")
    return model, preprocess_val


def find_image_paths(root_dir: Path) -> List[Path]:
    """Recursively collect all image file paths with supported extensions."""
    if not root_dir.exists():
        raise FileNotFoundError(f"Raw data directory not found: {root_dir}")

    image_paths = sorted(
        path
        for path in root_dir.rglob("*")
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    )
    logger.info("Found %d image(s) under '%s'.", len(image_paths), root_dir)
    return image_paths


def load_image(image_path: Path) -> Optional[Image.Image]:
    """Load and RGB-convert an image, returning None if the file is corrupted/unreadable."""
    try:
        with Image.open(image_path) as img:
            return img.convert("RGB")
    except (UnidentifiedImageError, OSError) as exc:
        tqdm.write(f"WARNING: Skipping corrupted image '{image_path}': {exc}")
        return None


def encode_image(
    image: Image.Image,
    model: torch.nn.Module,
    preprocess: "open_clip.transform.PreprocessCfg",
    device: torch.device,
) -> np.ndarray:
    """Encode a single PIL image into a unit-normalized OpenCLIP embedding vector."""
    image_tensor = preprocess(image).unsqueeze(0).to(device)
    with torch.inference_mode():
        embedding = model.encode_image(image_tensor)
        embedding = embedding / embedding.norm(dim=-1, keepdim=True)
    return embedding.squeeze(0).cpu().numpy().astype(np.float32)


def encode_all_images(
    image_paths: List[Path],
    model: torch.nn.Module,
    preprocess: "open_clip.transform.PreprocessCfg",
    device: torch.device,
) -> Tuple[np.ndarray, List[str]]:
    """
    Encode every image in image_paths, skipping corrupted files.

    Returns (embeddings, relative_paths), where embeddings[i] corresponds
    exactly to relative_paths[i] - this alignment is relied on downstream
    for indexing and retrieval, so both are built in lockstep here.
    """
    embeddings: List[np.ndarray] = []
    relative_paths: List[str] = []
    skipped_count = 0

    for image_path in tqdm(image_paths, desc="Encoding images", unit="img"):
        image = load_image(image_path)
        if image is None:
            skipped_count += 1
            continue

        try:
            embedding = encode_image(image, model, preprocess, device)
        except Exception as exc:  # noqa: BLE001 - keep the pipeline running on any failure
            tqdm.write(f"WARNING: Failed to encode '{image_path}': {exc}")
            skipped_count += 1
            continue

        embeddings.append(embedding)
        relative_paths.append(image_path.relative_to(RAW_DATA_DIR).as_posix())

    logger.info(
        "Encoded %d image(s), skipped %d corrupted/failed image(s).",
        len(embeddings),
        skipped_count,
    )
    return np.stack(embeddings, axis=0) if embeddings else np.empty((0, 0), dtype=np.float32), relative_paths


def save_embeddings(embeddings: np.ndarray, output_path: Path) -> None:
    """Save the embeddings array to disk as a .npy file, creating parent dirs as needed."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(output_path, embeddings)
    logger.info("Saved embeddings of shape %s to '%s'.", embeddings.shape, output_path)


def save_image_paths(relative_paths: List[str], output_path: Path) -> None:
    """Save the row-aligned image paths to a CSV file, creating parent dirs as needed."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(["image_path"])
        for relative_path in relative_paths:
            writer.writerow([relative_path])
    logger.info("Saved %d image path(s) to '%s'.", len(relative_paths), output_path)


def main() -> None:
    device = get_device()
    model, preprocess = load_model(device)

    image_paths = find_image_paths(RAW_DATA_DIR)
    if not image_paths:
        logger.warning("No images found - nothing to encode.")
        return

    embeddings, relative_paths = encode_all_images(image_paths, model, preprocess, device)
    if embeddings.size == 0:
        logger.warning("No embeddings were generated - all images were skipped.")
        return

    save_embeddings(embeddings, EMBEDDINGS_OUTPUT_PATH)
    save_image_paths(relative_paths, IMAGE_PATHS_OUTPUT_PATH)

    logger.info(
        "Done. %d/%d image(s) encoded successfully.",
        len(relative_paths),
        len(image_paths),
    )


if __name__ == "__main__":
    main()