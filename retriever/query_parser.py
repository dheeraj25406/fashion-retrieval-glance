"""
query_parser.py

First module of the retrieval pipeline. Converts a user's natural-language
query into structured filters using the same controlled vocabularies as
indexer/attribute_extractor.py. Purely deterministic regex/token matching -
no ML, no LLMs, no external APIs.

This module does not perform retrieval, embedding, FAISS search, or ranking.

Usage:
    python3 -m retriever.query_parser
"""

from __future__ import annotations

import logging
import re
from typing import Dict, List

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)

# Controlled vocabularies - kept identical to indexer/attribute_extractor.py
# so query-side and index-side attributes are directly comparable.
VOCABULARIES: Dict[str, List[str]] = {
    "colors": [
        "black", "white", "red", "blue", "green", "yellow",
        "pink", "brown", "beige", "gray", "orange", "purple",
    ],
    "upper_garment": [
        "shirt", "t-shirt", "tshirt", "blouse", "sweater", "hoodie", "polo", "kurta",
    ],
    "lower_garment": [
        "jeans", "trousers", "pants", "shorts", "skirt", "leggings",
    ],
    "outerwear": [
        "jacket", "blazer", "coat", "cardigan",
    ],
    "footwear": [
        "shoes", "sneakers", "boots", "heels", "sandals",
    ],
    "accessories": [
        "hat", "cap", "tie", "belt", "handbag", "backpack", "sunglasses", "scarf",
    ],
    "style": [
        "formal", "casual", "sporty", "traditional", "party", "business",
    ],
    "scene": [
        "runway", "street", "office", "indoor", "outdoor", "park", "beach", "stage",
    ],
}


def build_vocab_patterns() -> Dict[str, re.Pattern]:
    """
    Compile one case-insensitive whole-word regex per vocabulary category.
    Terms are ordered longest-first within each pattern so overlapping terms
    (e.g. "t-shirt" vs a hypothetical "shirt") don't shadow one another.
    """
    patterns: Dict[str, re.Pattern] = {}
    for category, terms in VOCABULARIES.items():
        sorted_terms = sorted(terms, key=len, reverse=True)
        escaped_terms = [re.escape(term) for term in sorted_terms]
        pattern = r"\b(" + "|".join(escaped_terms) + r")\b"
        patterns[category] = re.compile(pattern, flags=re.IGNORECASE)
    return patterns


def find_matches(text: str, pattern: re.Pattern) -> List[str]:
    """Return distinct vocabulary terms matched in text, in order of first appearance."""
    matches = pattern.findall(text)
    seen: List[str] = []
    for match in matches:
        lowered = match.lower()
        if lowered not in seen:
            seen.append(lowered)
    return seen


def remove_matched_terms(text: str, patterns: Dict[str, re.Pattern]) -> str:
    """Strip all vocabulary-matched terms from text, collapsing extra whitespace."""
    remaining = text
    for pattern in patterns.values():
        remaining = pattern.sub("", remaining)
    remaining = re.sub(r"\s+", " ", remaining).strip()
    return remaining


def parse_query(query: str, patterns: Dict[str, re.Pattern]) -> Dict[str, object]:
    """Parse a natural-language query into structured filters plus leftover free text."""
    lowered_query = query.lower()

    matches_by_category: Dict[str, List[str]] = {
        category: find_matches(lowered_query, pattern)
        for category, pattern in patterns.items()
    }
    remaining_text = remove_matched_terms(lowered_query, patterns)

    result: Dict[str, object] = {"query": query, "remaining_text": remaining_text}
    result.update(matches_by_category)

    logger.info("Parsed query '%s' -> %s", query, result)
    return result


def main() -> None:
    patterns = build_vocab_patterns()

    example_queries = [
        "red formal jacket",
        "black sneakers",
        "blue jeans with white shirt",
        "casual green hoodie",
    ]

    for query in example_queries:
        parsed = parse_query(query, patterns)
        print(parsed)


if __name__ == "__main__":
    main()