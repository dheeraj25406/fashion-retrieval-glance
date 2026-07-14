"""
attribute_extractor.py

Second stage of the fashion retrieval indexing pipeline. Reads free-text
captions produced by caption_generator.py and deterministically parses them
into structured fashion metadata using controlled vocabularies (no LLM or
external API calls). Writes the result to `data/processed/attributes.csv`.

Usage:
    python3 -m indexer.attribute_extractor
"""

from __future__ import annotations

import csv
import logging
import re
from pathlib import Path
from typing import Dict, List

from tqdm import tqdm

INPUT_CSV_PATH = Path("data/processed/captions.csv")
OUTPUT_CSV_PATH = Path("data/processed/attributes.csv")

OUTPUT_FIELDNAMES = [
    "image_path",
    "caption",
    "upper_garment",
    "lower_garment",
    "outerwear",
    "footwear",
    "colors",
    "accessories",
    "scene",
    "style",
]

# Controlled vocabularies. Order within each list does not affect matching,
# but longer/more-specific terms are checked before shorter ones at match
# time so e.g. "t-shirt" isn't shadowed by a looser future addition.
COLOR_VOCAB: List[str] = [
    "black", "white", "red", "blue", "green", "yellow",
    "pink", "brown", "beige", "gray", "orange", "purple",
]

UPPER_GARMENT_VOCAB: List[str] = [
    "shirt", "tshirt", "t-shirt", "blouse", "sweater", "hoodie", "polo", "kurta",
]

LOWER_GARMENT_VOCAB: List[str] = [
    "jeans", "trousers", "pants", "shorts", "skirt", "leggings",
]

OUTERWEAR_VOCAB: List[str] = [
    "blazer", "jacket", "coat", "cardigan",
]

FOOTWEAR_VOCAB: List[str] = [
    "shoes", "sneakers", "boots", "heels", "sandals",
]

ACCESSORY_VOCAB: List[str] = [
    "hat", "cap", "tie", "belt", "handbag", "backpack", "sunglasses", "scarf",
]

SCENE_VOCAB: List[str] = [
    "runway", "street", "office", "indoor", "outdoor", "park", "beach", "stage",
]

STYLE_VOCAB: List[str] = [
    "formal", "casual", "sporty", "traditional", "party", "business",
]

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)


def build_vocab_pattern(vocab: List[str]) -> re.Pattern:
    """
    Compile a case-insensitive regex that matches any vocabulary term as a
    whole word/phrase. Longer terms are ordered first within alternation so
    that overlapping terms (e.g. "t-shirt" vs "shirt") are both matchable
    independently without one masking the other.
    """
    sorted_terms = sorted(vocab, key=len, reverse=True)
    escaped_terms = [re.escape(term) for term in sorted_terms]
    pattern = r"\b(" + "|".join(escaped_terms) + r")\b"
    return re.compile(pattern, flags=re.IGNORECASE)


COLOR_PATTERN = build_vocab_pattern(COLOR_VOCAB)
UPPER_GARMENT_PATTERN = build_vocab_pattern(UPPER_GARMENT_VOCAB)
LOWER_GARMENT_PATTERN = build_vocab_pattern(LOWER_GARMENT_VOCAB)
OUTERWEAR_PATTERN = build_vocab_pattern(OUTERWEAR_VOCAB)
FOOTWEAR_PATTERN = build_vocab_pattern(FOOTWEAR_VOCAB)
ACCESSORY_PATTERN = build_vocab_pattern(ACCESSORY_VOCAB)
SCENE_PATTERN = build_vocab_pattern(SCENE_VOCAB)
STYLE_PATTERN = build_vocab_pattern(STYLE_VOCAB)


def find_first_match(caption: str, pattern: re.Pattern) -> str:
    """Return the first vocabulary term matched in the caption, or '' if none."""
    match = pattern.search(caption)
    return match.group(1).lower() if match else ""


def find_all_matches(caption: str, pattern: re.Pattern) -> List[str]:
    """Return all distinct vocabulary terms matched in the caption, in order of appearance."""
    matches = pattern.findall(caption)
    seen: List[str] = []
    for match in matches:
        lowered = match.lower()
        if lowered not in seen:
            seen.append(lowered)
    return seen


def extract_attributes(caption: str) -> Dict[str, str]:
    """
    Deterministically parse a single caption into structured fashion metadata
    using controlled-vocabulary matching. Missing fields are empty strings;
    multi-value fields are comma-separated.
    """
    colors = find_all_matches(caption, COLOR_PATTERN)
    accessories = find_all_matches(caption, ACCESSORY_PATTERN)

    return {
        "upper_garment": find_first_match(caption, UPPER_GARMENT_PATTERN),
        "lower_garment": find_first_match(caption, LOWER_GARMENT_PATTERN),
        "outerwear": find_first_match(caption, OUTERWEAR_PATTERN),
        "footwear": find_first_match(caption, FOOTWEAR_PATTERN),
        "colors": ", ".join(colors),
        "accessories": ", ".join(accessories),
        "scene": find_first_match(caption, SCENE_PATTERN),
        "style": find_first_match(caption, STYLE_PATTERN),
    }


def read_captions(input_path: Path) -> List[Dict[str, str]]:
    """Read the captions CSV into a list of row dicts."""
    if not input_path.exists():
        raise FileNotFoundError(f"Captions file not found: {input_path}")

    with input_path.open("r", newline="", encoding="utf-8") as csv_file:
        reader = csv.DictReader(csv_file)
        rows = list(reader)

    logger.info("Read %d caption row(s) from '%s'.", len(rows), input_path)
    return rows


def process_rows(rows: List[Dict[str, str]]) -> tuple[List[Dict[str, str]], int]:
    """
    Extract structured attributes for each caption row, skipping any row
    that fails without stopping the overall run. Returns (output_rows, failed_count).
    """
    output_rows: List[Dict[str, str]] = []
    failed_count = 0

    for row in tqdm(rows, desc="Extracting attributes", unit="row"):
        image_path = row.get("image_path", "")
        caption = row.get("caption", "")

        try:
            attributes = extract_attributes(caption)
        except Exception as exc:  # noqa: BLE001 - keep the pipeline running on any failure
            tqdm.write(f"WARNING: Failed to extract attributes for '{image_path}': {exc}")
            failed_count += 1
            continue

        output_row = {"image_path": image_path, "caption": caption}
        output_row.update(attributes)
        output_rows.append(output_row)

    return output_rows, failed_count


def write_attributes_csv(rows: List[Dict[str, str]], output_path: Path) -> None:
    """Write structured attribute rows to a CSV file, creating parent dirs as needed."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=OUTPUT_FIELDNAMES)
        writer.writeheader()
        writer.writerows(rows)
    logger.info("Wrote %d attribute row(s) to '%s'.", len(rows), output_path)


def main() -> None:
    rows = read_captions(INPUT_CSV_PATH)
    if not rows:
        logger.warning("No caption rows found - nothing to extract.")
        return

    output_rows, failed_count = process_rows(rows)
    write_attributes_csv(output_rows, OUTPUT_CSV_PATH)

    logger.info(
        "Summary: %d processed, %d failed, %d total row(s).",
        len(output_rows),
        failed_count,
        len(rows),
    )


if __name__ == "__main__":
    main()