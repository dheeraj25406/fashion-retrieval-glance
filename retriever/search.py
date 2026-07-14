"""Orchestration layer for the fashion image retrieval pipeline.

Connects every previously built module into a single end-to-end text-to-
image search: parses a natural-language query, encodes it with the same
OpenCLIP text encoder used during indexing, searches the FAISS index for
nearest-neighbour candidates, and reranks those candidates using structured
attribute matching.

This module does not rebuild the FAISS index, does not encode images, does
not load captions.csv, and does not perform attribute extraction - all of
that is handled by the indexer modules and by retriever/reranker.py.

Usage:
    python3 -m retriever.search
"""

from __future__ import annotations

import csv
import logging
import time
from pathlib import Path
from typing import Dict, List, Tuple

import faiss
import numpy as np
import open_clip
import torch

from retriever.query_parser import build_vocab_patterns, parse_query
from retriever.reranker import load_attributes, rerank_candidates

MODEL_NAME = "ViT-B-32"
PRETRAINED_TAG = "laion2b_s34b_b79k"
FAISS_INDEX_PATH = Path("data/processed/faiss_index.bin")
IMAGE_PATHS_CSV_PATH = Path("data/processed/image_paths.csv")
ATTRIBUTES_CSV_PATH = Path("data/processed/attributes.csv")
TOP_K = 20

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)


def get_device() -> torch.device:
    """Select the Apple MPS device if available, otherwise fall back to CPU.

    Returns:
        The torch device to run the OpenCLIP model on.
    """
    if torch.backends.mps.is_available():
        logger.info("MPS backend available - using Apple GPU.")
        device = torch.device("mps")
    else:
        logger.info("MPS backend not available - using CPU.")
        device = torch.device("cpu")
    logger.info("Using device: %s", device)
    return device


def load_clip_model(
    device: torch.device,
) -> Tuple[torch.nn.Module, "open_clip.tokenizer.HFTokenizer"]:
    """Load the same OpenCLIP model and tokenizer used during indexing.

    Args:
        device: The torch device to load the model onto.

    Returns:
        A tuple of (model, tokenizer). The tokenizer is required to encode
        query text into token tensors before calling ``model.encode_text``.
    """
    logger.info("Loading OpenCLIP model '%s' (pretrained='%s')...", MODEL_NAME, PRETRAINED_TAG)
    model, _, _ = open_clip.create_model_and_transforms(
        MODEL_NAME, pretrained=PRETRAINED_TAG, device=device
    )
    model.eval()
    tokenizer = open_clip.get_tokenizer(MODEL_NAME)
    logger.info("CLIP model and tokenizer loaded successfully.")
    return model, tokenizer


def load_faiss_index(index_path: Path) -> faiss.Index:
    """Load a prebuilt FAISS index from disk.

    Args:
        index_path: Path to the faiss_index.bin file produced by
            indexer/build_index.py.

    Returns:
        The loaded FAISS index.

    Raises:
        FileNotFoundError: If ``index_path`` does not exist.
    """
    if not index_path.exists():
        raise FileNotFoundError(
            f"FAISS index not found: {index_path}. Run indexer/build_index.py first."
        )

    logger.info("Loading FAISS index from '%s'...", index_path)
    index = faiss.read_index(str(index_path))
    logger.info("Loaded FAISS index with %d vector(s).", index.ntotal)
    return index


def load_image_paths(image_paths_csv_path: Path) -> List[str]:
    """Load image paths in the exact order they were indexed.

    The row order of this file matches the row order of image_embeddings.npy
    at index-build time, so list position ``i`` here corresponds exactly to
    FAISS internal index ``i``. This function must preserve that order.

    Args:
        image_paths_csv_path: Path to the image_paths.csv file produced by
            indexer/image_encoder.py.

    Returns:
        An ordered list of image paths, positionally aligned with the
        FAISS index.

    Raises:
        FileNotFoundError: If ``image_paths_csv_path`` does not exist.
    """
    if not image_paths_csv_path.exists():
        raise FileNotFoundError(
            f"Image paths file not found: {image_paths_csv_path}. "
            "Run indexer/image_encoder.py first."
        )

    with image_paths_csv_path.open("r", newline="", encoding="utf-8") as csv_file:
        reader = csv.DictReader(csv_file)
        image_paths = [row["image_path"] for row in reader]

    logger.info("Loaded %d image path(s) from '%s'.", len(image_paths), image_paths_csv_path)
    return image_paths


def encode_query(
    query_text: str,
    model: torch.nn.Module,
    tokenizer: "open_clip.tokenizer.HFTokenizer",
    device: torch.device,
) -> np.ndarray:
    """Encode and L2-normalize the original query text with the OpenCLIP text encoder.

    Args:
        query_text: The original, unparsed user query text (not the
            query_parser.py remaining_text).
        model: The loaded OpenCLIP model.
        tokenizer: The OpenCLIP tokenizer matching ``model``.
        device: The torch device the model is loaded on.

    Returns:
        A unit-normalized 1-D float32 embedding vector for the query.
    """
    tokens = tokenizer([query_text]).to(device)
    with torch.inference_mode():
        embedding = model.encode_text(tokens)
        embedding = embedding / embedding.norm(dim=-1, keepdim=True)
    return embedding.squeeze(0).cpu().numpy().astype(np.float32)


def search_index(
    query_embedding: np.ndarray,
    index: faiss.Index,
    image_paths: List[str],
    top_k: int,
) -> List[str]:
    """Search the FAISS index and convert the resulting indices into image paths.

    Args:
        query_embedding: A unit-normalized 1-D query embedding.
        index: The loaded FAISS IndexFlatIP.
        image_paths: The ordered image paths list from :func:`load_image_paths`,
            positionally aligned with the FAISS index.
        top_k: Number of nearest neighbours to retrieve.

    Returns:
        A list of image paths for the top_k nearest neighbours, ordered by
        descending cosine similarity.
    """
    query_matrix = np.expand_dims(query_embedding, axis=0)
    effective_k = min(top_k, index.ntotal)
    if effective_k < top_k:
        logger.warning(
            "Requested top_k=%d exceeds index size (%d) - returning %d instead.",
            top_k, index.ntotal, effective_k,
        )

   scores, indices = index.search(query_matrix, effective_k)

    candidate_paths: List[str] = []
    for position in indices[0]:
        if position == -1:
            continue  # FAISS pads with -1 when fewer than k results exist
        candidate_paths.append(image_paths[position])

    logger.info("FAISS search returned %d candidate(s).", len(candidate_paths))
    return candidate_paths, scores[0]


def print_results(reranked_results: List[Dict[str, object]]) -> None:
    """Print the final reranked results as a numbered list of image/score pairs.

    Args:
        reranked_results: The list returned by rerank_candidates(), sorted
            by descending attribute score.
    """
    for rank, result in enumerate(reranked_results, start=1):
        print(f"{rank}.")
        print(f"   image: {result['image_path']}")
        print(f"   score: {result['score']}")


def main() -> None:
    """Run the complete text-to-image retrieval pipeline end-to-end."""
    start_time = time.monotonic()

    query_text = input("Enter your query: ").strip()

    if not query_text:
        logger.error("Query cannot be empty.")
        return

    device = get_device()
    model, tokenizer = load_clip_model(device)
    index = load_faiss_index(FAISS_INDEX_PATH)
    image_paths = load_image_paths(IMAGE_PATHS_CSV_PATH)
    attributes_by_path = load_attributes(ATTRIBUTES_CSV_PATH)

    query_patterns = build_vocab_patterns()
    parsed_query = parse_query(query_text, query_patterns)

    query_embedding = encode_query(query_text, model, tokenizer, device)
    candidate_paths = search_index(query_embedding, index, image_paths, TOP_K)
    reranked_results = rerank_candidates(candidate_paths, parsed_query, attributes_by_path)

    elapsed_seconds = time.monotonic() - start_time

    print(f"\nParsed query: {parsed_query}")
    print(f"Number of FAISS candidates: {len(candidate_paths)}\n")
    print("Final reranked results:")
    print_results(reranked_results)
    print(f"\nTotal execution time: {elapsed_seconds:.2f} seconds")


if __name__ == "__main__":
    main()