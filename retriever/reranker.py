"""Reranks FAISS candidate images using structured attribute matching.

This module is the compositional-verification stage of the retrieval
pipeline: given a shortlist of candidate image paths (already produced by
a FAISS similarity search) and a parsed query (already produced by
retriever/query_parser.py), it scores each candidate by how many of its
structured attributes (from indexer/attribute_extractor.py's
attributes.csv) match the attributes requested in the query, and returns
the candidates sorted primarily by descending attribute score and secondarily
by descending FAISS similarity score.

This module does not perform CLIP embedding, FAISS search, query parsing,
image loading, or visualization - it only reranks.

Usage:
    python3 -m retriever.reranker
"""

from __future__ import annotations

import csv
import logging
from pathlib import Path
from typing import Dict, List

ATTRIBUTES_CSV_PATH = Path("data/processed/attributes.csv")

# Single-valued attribute fields in attributes.csv (one string per image).
SINGLE_VALUE_FIELDS: List[str] = [
    "upper_garment",
    "lower_garment",
    "outerwear",
    "footwear",
    "scene",
    "style",
]

# Multi-valued attribute fields in attributes.csv (comma-separated strings).
MULTI_VALUE_FIELDS: List[str] = [
    "colors",
    "accessories",
]

# Points awarded per matching attribute field.
ATTRIBUTE_WEIGHTS: Dict[str, int] = {
    "colors": 3,
    "upper_garment": 3,
    "lower_garment": 3,
    "outerwear": 3,
    "footwear": 3,
    "style": 2,
    "scene": 1,
    "accessories": 1,
}

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)


def load_attributes(attributes_csv_path: Path) -> Dict[str, Dict[str, str]]:
    """Load structured attributes into an in-memory lookup keyed by image path.

    Args:
        attributes_csv_path: Path to the attributes.csv file produced by
            indexer/attribute_extractor.py.

    Returns:
        A dictionary mapping each ``image_path`` to its full row of
        attribute fields (as a dict of column name to raw string value).

    Raises:
        FileNotFoundError: If ``attributes_csv_path`` does not exist.
    """
    if not attributes_csv_path.exists():
        raise FileNotFoundError(
            f"Attributes file not found: {attributes_csv_path}. "
            "Run indexer/attribute_extractor.py first to generate it."
        )

    attributes_by_path: Dict[str, Dict[str, str]] = {}
    with attributes_csv_path.open("r", newline="", encoding="utf-8") as csv_file:
        reader = csv.DictReader(csv_file)
        for row in reader:
            image_path = row.get("image_path", "")
            if not image_path:
                logger.warning("Skipping attributes row with missing image_path: %s", row)
                continue
            attributes_by_path[image_path] = row

    logger.info("Loaded attributes for %d image(s) from '%s'.", len(attributes_by_path), attributes_csv_path)
    return attributes_by_path


def _split_multi_value(raw_value: str) -> List[str]:
    """Split a comma-separated attribute string into lowercased, trimmed terms.

    Args:
        raw_value: A raw CSV cell value such as ``"red, blue"`` or ``""``.

    Returns:
        A list of lowercased terms, empty if ``raw_value`` is blank.
    """
    if not raw_value:
        return []
    return [term.strip().lower() for term in raw_value.split(",") if term.strip()]


def compute_attribute_score(parsed_query: Dict[str, object], attributes: Dict[str, str]) -> int:
    """Score how well a candidate's attributes match a parsed query.

    Each requested attribute field in ``parsed_query`` that has at least one
    overlapping value with the candidate's corresponding field in
    ``attributes`` contributes that field's fixed weight to the total score.
    Scoring is purely deterministic set-overlap matching - no partial
    credit, no fuzzy matching, no ML.

    Args:
        parsed_query: The structured query dictionary produced by
            retriever/query_parser.py, where every attribute field is a
            list of lowercased terms.
        attributes: A single candidate's attribute row from attributes.csv,
            where single-valued fields are plain strings and multi-valued
            fields (colors, accessories) are comma-separated strings.

    Returns:
        The total integer score for this candidate.
    """
    score = 0

    for field in SINGLE_VALUE_FIELDS:
        query_terms = parsed_query.get(field) or []
        candidate_value = (attributes.get(field) or "").strip().lower()
        if candidate_value and candidate_value in query_terms:
            score += ATTRIBUTE_WEIGHTS[field]

    for field in MULTI_VALUE_FIELDS:
        query_terms = set(parsed_query.get(field) or [])
        candidate_terms = set(_split_multi_value(attributes.get(field, "")))
        if query_terms & candidate_terms:
            score += ATTRIBUTE_WEIGHTS[field]

    return score


def rerank_candidates(
    candidate_results: List[Dict[str, object]],
    parsed_query: Dict[str, object],
    attributes_by_path: Dict[str, Dict[str, str]],
) -> List[Dict[str, object]]:
    """Rerank FAISS candidate results by structured attribute match score.

    Args:
        candidate_results: Ordered FAISS retrieval results. Each item
            contains ``image_path`` and ``similarity``.
        parsed_query: The structured query dictionary produced by
            retriever/query_parser.py.
        attributes_by_path: The in-memory attribute lookup produced by
            :func:`load_attributes`.

    Returns:
        A list of dictionaries, one per candidate, each containing
        ``image_path``, ``score``, ``similarity``, and ``attributes``,
        sorted by descending score, with ties broken by descending
        similarity. Candidates with no attribute row score 0 and are
        not dropped.
    """
    scored_candidates: List[Dict[str, object]] = []

    for candidate in candidate_results:
        image_path = candidate["image_path"]
        similarity = candidate["similarity"]
        attributes = attributes_by_path.get(image_path)
        if attributes is None:
            logger.warning(
                "No attributes found for candidate '%s' - scoring as 0.", image_path
            )
            attributes = {}

        score = compute_attribute_score(parsed_query, attributes)
        scored_candidates.append(
            {
                "image_path": image_path,
                "score": score,
                "similarity": similarity,
                "attributes": attributes,
            }
        )

    # sorted() is stable, so equal-(score, similarity) candidates preserve
    # their original relative order from candidate_results.
    scored_candidates.sort(
        key=lambda candidate: (candidate["score"], candidate["similarity"]),
        reverse=True,
    )

    logger.info(
        "Reranked %d candidate(s); top score = %s.",
        len(scored_candidates),
        scored_candidates[0]["score"] if scored_candidates else "N/A",
    )
    return scored_candidates


def main() -> None:
    """Demonstrate reranking using dummy candidates and a sample parsed query."""
    dummy_attributes_by_path: Dict[str, Dict[str, str]] = {
        "0001.jpg": {
            "upper_garment": "",
            "lower_garment": "",
            "outerwear": "jacket",
            "footwear": "",
            "colors": "red, black",
            "accessories": "",
            "scene": "",
            "style": "formal",
        },
        "0010.jpg": {
            "upper_garment": "shirt",
            "lower_garment": "jeans",
            "outerwear": "",
            "footwear": "sneakers",
            "colors": "blue, white",
            "accessories": "cap",
            "scene": "street",
            "style": "casual",
        },
        "0100.jpg": {
            "upper_garment": "",
            "lower_garment": "",
            "outerwear": "jacket",
            "footwear": "",
            "colors": "red",
            "accessories": "",
            "scene": "office",
            "style": "formal",
        },
    }

    candidate_results = [
        {
            "image_path": "0001.jpg",
            "similarity": 0.91,
        },
        {
            "image_path": "0010.jpg",
            "similarity": 0.88,
        },
        {
            "image_path": "0100.jpg",
            "similarity": 0.82,
        },
    ]

    parsed_query: Dict[str, object] = {
        "query": "red formal jacket",
        "remaining_text": "",
        "colors": ["red"],
        "upper_garment": [],
        "lower_garment": [],
        "outerwear": ["jacket"],
        "footwear": [],
        "accessories": [],
        "style": ["formal"],
        "scene": [],
    }

    reranked = rerank_candidates(
        candidate_results,
        parsed_query,
        dummy_attributes_by_path,
    )

    print(f"Query: {parsed_query['query']}\n")
    for result in reranked:
        print(
            f"  {result['image_path']} "
            f"(score={result['score']}, similarity={result['similarity']:.4f})"
        )


if __name__ == "__main__":
    main()