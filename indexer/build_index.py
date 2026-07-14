"""
build_index.py

Fourth stage of the fashion retrieval indexing pipeline. Loads the image
embeddings produced by image_encoder.py, L2-normalizes them, builds a
FAISS IndexFlatIP (cosine similarity via inner product on unit vectors),
and saves the index to disk.

This module's only responsibility is building and saving the FAISS index.
It does not load captions, attributes, or image paths, and does not build
metadata or perform retrieval.

Usage:
    python3 -m indexer.build_index
"""

from __future__ import annotations

import logging
from pathlib import Path

import faiss
import numpy as np

EMBEDDINGS_PATH = Path("data/processed/image_embeddings.npy")
INDEX_OUTPUT_PATH = Path("data/processed/faiss_index.bin")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)


def load_embeddings(embeddings_path: Path) -> np.ndarray:
    """Load the image embeddings array, validating existence, dtype, and non-emptiness."""
    if not embeddings_path.exists():
        raise FileNotFoundError(
            f"Embeddings file not found: {embeddings_path}. "
            "Run image_encoder.py first to generate it."
        )

    logger.info("Loading embeddings from '%s'...", embeddings_path)
    embeddings = np.load(embeddings_path)

    if embeddings.size == 0:
        raise ValueError(f"Embeddings array in '{embeddings_path}' is empty.")

    if embeddings.dtype != np.float32:
        raise ValueError(
            f"Expected embeddings dtype float32, got {embeddings.dtype}."
        )

    logger.info("Loaded embedding matrix with shape: %s", embeddings.shape)
    logger.info("Embedding dimension: %d", embeddings.shape[1])
    print(f"Embedding matrix shape: {embeddings.shape}")
    print(f"Embedding dimension: {embeddings.shape[1]}")

    return embeddings


def normalize_embeddings(embeddings: np.ndarray) -> np.ndarray:
    """L2-normalize each embedding row so inner product equals cosine similarity."""
    logger.info("L2-normalizing embeddings...")
    embeddings = embeddings.copy()
    embeddings /= np.linalg.norm(embeddings, axis=1, keepdims=True)
    logger.info("Normalization complete.")
    return embeddings


def build_faiss_index(embeddings: np.ndarray) -> faiss.IndexFlatIP:
    """Build a FAISS IndexFlatIP and add all embeddings to it."""
    embedding_dimension = embeddings.shape[1]
    logger.info("Building FAISS IndexFlatIP with dimension %d...", embedding_dimension)

    index = faiss.IndexFlatIP(embedding_dimension)
    index.add(embeddings)

    logger.info("Total indexed vectors: %d", index.ntotal)
    print(f"Total indexed vectors: {index.ntotal}")

    return index


def save_index(index: faiss.IndexFlatIP, output_path: Path) -> None:
    """Save the FAISS index to disk, creating parent directories as needed."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    faiss.write_index(index, str(output_path))
    logger.info("Saved FAISS index to '%s'.", output_path)


def main() -> None:
    embeddings = load_embeddings(EMBEDDINGS_PATH)
    normalized_embeddings = normalize_embeddings(embeddings)
    index = build_faiss_index(normalized_embeddings)
    save_index(index, INDEX_OUTPUT_PATH)
    logger.info("FAISS index build complete.")


if __name__ == "__main__":
    main()