"""
caption_generator.py

Generates natural-language captions for all images under `data/raw/` using
the Salesforce BLIP image-captioning model, and writes the results to
`data/processed/captions.csv`.

Usage:
    python3 -m indexer.caption_generator
"""

from __future__ import annotations

import csv
import logging
import time
from pathlib import Path
from typing import List, Optional, Set

import torch
from PIL import Image, UnidentifiedImageError
from tqdm import tqdm
from transformers import BlipForConditionalGeneration, BlipProcessor

MODEL_NAME = "Salesforce/blip-image-captioning-base"
RAW_DATA_DIR = Path("data/raw")
OUTPUT_CSV_PATH = Path("data/processed/captions.csv")
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png"}
CSV_FIELDNAMES = ["image_path", "caption"]
SAVE_EVERY_N_IMAGES = 100

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)


def get_device() -> torch.device:
    """Return the Apple MPS device if available, otherwise fall back to CPU."""
    if torch.backends.mps.is_available():
        logger.info("MPS backend available - using Apple GPU.")
        torch.set_float32_matmul_precision("high")
        device = torch.device("mps")
    else:
        logger.info("MPS backend not available - using CPU.")
        device = torch.device("cpu")
    logger.info("Using device: %s", device)
    return device


def load_model(device: torch.device) -> tuple[BlipProcessor, BlipForConditionalGeneration]:
    """Load the BLIP processor and captioning model onto the given device."""
    logger.info("Loading model '%s'...", MODEL_NAME)
    processor = BlipProcessor.from_pretrained(MODEL_NAME)
    model = BlipForConditionalGeneration.from_pretrained(MODEL_NAME).to(device)
    model.eval()
    logger.info("Model loaded successfully.")
    return processor, model


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


def generate_caption(
    image: Image.Image,
    processor: BlipProcessor,
    model: BlipForConditionalGeneration,
    device: torch.device,
) -> str:
    """Generate a single caption for the given PIL image."""
    inputs = processor(images=image, return_tensors="pt").to(device)
    with torch.inference_mode():
        output_ids = model.generate(
            **inputs,
            num_beams=5,
            max_new_tokens=30,
            early_stopping=True,
        )
    caption = processor.decode(output_ids[0], skip_special_tokens=True)
    return caption.strip()


def load_processed_image_names(output_path: Path) -> Set[str]:
    """Read image_path values already present in an existing captions CSV, if any."""
    if not output_path.exists():
        return set()

    processed: Set[str] = set()
    with output_path.open("r", newline="", encoding="utf-8") as csv_file:
        reader = csv.DictReader(csv_file)
        for row in reader:
            rel_path = row.get("image_path")
            if rel_path:
                processed.add(rel_path)

    if processed:
        logger.info(
            "Found existing '%s' with %d already-processed image(s) - will resume.",
            output_path,
            len(processed),
        )
    return processed


def append_rows_to_csv(rows: List[dict], output_path: Path, write_header: bool) -> None:
    """Append rows to the captions CSV, writing the header only if the file is new."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("a", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=CSV_FIELDNAMES)
        if write_header:
            writer.writeheader()
        writer.writerows(rows)


def process_images(
    image_paths: List[Path],
    processor: BlipProcessor,
    model: BlipForConditionalGeneration,
    device: torch.device,
    output_path: Path,
    already_processed: Set[str],
) -> tuple[int, int, int]:
    """
    Generate captions for image_paths, skipping any already in `already_processed`.

    Appends results to `output_path` every SAVE_EVERY_N_IMAGES images so progress
    survives interruptions. Returns (processed_count, skipped_count, failed_count).
    """
    file_exists = output_path.exists()
    pending_rows: List[dict] = []
    processed_count = 0
    skipped_count = 0
    failed_count = 0

    for image_path in tqdm(image_paths, desc="Captioning images", unit="img"):
        relative_path = image_path.relative_to(RAW_DATA_DIR).as_posix()

        if relative_path in already_processed:
            skipped_count += 1
            continue

        image = load_image(image_path)
        if image is None:
            failed_count += 1
            continue

        try:
            caption = generate_caption(image, processor, model, device)
        except Exception as exc:  # noqa: BLE001 - keep the pipeline running on any failure
            tqdm.write(f"WARNING: Failed to caption '{image_path}': {exc}")
            failed_count += 1
            continue

        pending_rows.append({"image_path": relative_path, "caption": caption})
        processed_count += 1

        if len(pending_rows) >= SAVE_EVERY_N_IMAGES:
            append_rows_to_csv(pending_rows, output_path, write_header=not file_exists)
            file_exists = True
            tqdm.write(f"Checkpoint: saved {len(pending_rows)} new caption(s).")
            pending_rows = []

    if pending_rows:
        append_rows_to_csv(pending_rows, output_path, write_header=not file_exists)
        tqdm.write(f"Final save: saved {len(pending_rows)} new caption(s).")

    return processed_count, skipped_count, failed_count


def main() -> None:
    start_time = time.monotonic()

    device = get_device()
    processor, model = load_model(device)

    image_paths = find_image_paths(RAW_DATA_DIR)
    # lets test with 10 images first
    # image_paths = image_paths[:10]
    if not image_paths:
        logger.warning("No images found - nothing to caption.")
        return

    already_processed = load_processed_image_names(OUTPUT_CSV_PATH)

    processed_count, skipped_count, failed_count = process_images(
        image_paths, processor, model, device, OUTPUT_CSV_PATH, already_processed
    )

    elapsed_seconds = time.monotonic() - start_time
    logger.info(
        "Summary: %d processed, %d skipped (already done), %d failed/corrupted, "
        "%d total image(s) found.",
        processed_count,
        skipped_count,
        failed_count,
        len(image_paths),
    )
    logger.info("Total execution time: %.2f seconds.", elapsed_seconds)


if __name__ == "__main__":
    main()