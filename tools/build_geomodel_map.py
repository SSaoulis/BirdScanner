#!/usr/bin/env python3
"""Offline build-time tool: map geomodel species labels onto classifier classes.

This script is **not** part of the BirdScanner runtime. It builds the crosswalk
between the geomodel's ~12k species labels (eBird/Clements naming) and the
classifier's ~700 class labels (IOC-style naming) so a geomodel prediction can be
projected onto the classifier's index space.

Matching is done by :func:`birdscanner.ml.geomodel.build_name_mapping`, which
compares normalised common names (case/accent/punctuation folded, British
``grey`` unified to ``gray``). That resolves the ~92% of classes that differ only
in spelling. The remaining classes are genuine cross-checklist synonyms
(e.g. classifier ``"Common blackbird"`` == geomodel ``"Eurasian Blackbird"``) that
string matching cannot bridge; this tool prints them as a table for a human to
curate. Curated pairs are added straight into the output JSON and are **preserved**
across re-runs (any entry whose classifier label is not an auto-match is treated as
a curated override).

Inputs
------
- ``assets/labels/BirdNET+_Geomodel_V3.0.3_Global_12K_Labels.txt`` — geomodel labels.
- ``assets/models/convnext_v2_tiny.onnx_class_to_idx.json`` — classifier classes
  (source of truth for "what species the classifier predicts").

Output
------
- ``assets/labels/geomodel_classifier_map.json`` — a flat, sorted
  ``{geomodel_common_name: classifier_label}`` object (auto-matches + curated pairs).

Usage
-----
    python tools/build_geomodel_map.py            # (re)build + print unmatched table
    python tools/build_geomodel_map.py --check    # print the table only, write nothing
"""

from __future__ import annotations

import argparse
import json
import os
import sys

# Make the repo root importable so ``birdscanner.ml.geomodel`` resolves when this
# script is run directly (``python tools/build_geomodel_map.py``).
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(_THIS_DIR)
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

# pylint: disable=wrong-import-position
from birdscanner.ml.geomodel import (  # noqa: E402
    build_name_mapping,
    load_labels,
    normalize_common_name,
)

GEOMODEL_LABELS_PATH = os.path.join(
    REPO_ROOT, "assets", "labels", "BirdNET+_Geomodel_V3.0.3_Global_12K_Labels.txt"
)
CLASS_TO_IDX_PATH = os.path.join(
    REPO_ROOT, "assets", "models", "convnext_v2_tiny.onnx_class_to_idx.json"
)
OUTPUT_PATH = os.path.join(
    REPO_ROOT, "assets", "labels", "geomodel_classifier_map.json"
)


def load_classifier_labels(path: str) -> list[str]:
    """Load the classifier's class labels from its ``class_to_idx`` JSON.

    Parameters
    - path: path to ``convnext_v2_tiny.onnx_class_to_idx.json`` (a ``{label: index}`` map).

    Returns
    - the class labels ordered by their index (the classifier's output order).
    """
    with open(path, encoding="utf-8") as f:
        class_to_idx: dict[str, int] = json.load(f)
    return [label for label, _ in sorted(class_to_idx.items(), key=lambda kv: kv[1])]


def load_existing_overrides(
    path: str, geomodel_labels: list[dict[str, str]]
) -> dict[str, str]:
    """Recover hand-curated pairs from a previously written map, to preserve them.

    An entry from the existing map is treated as a curated override when its
    classifier label is **not** one that normalisation would auto-match on its own, so
    re-running the build never clobbers hand-added synonyms.

    Parameters
    - path: the output map path from a previous run (may not exist).
    - geomodel_labels: the geomodel rows, used to recompute the auto-match set.

    Returns
    - ``{geomodel_common_name: classifier_label}`` for the curated entries only.
    """
    if not os.path.exists(path):
        return {}
    with open(path, encoding="utf-8") as f:
        existing: dict[str, str] = json.load(f)

    auto_geo_keys = {normalize_common_name(row["common"]) for row in geomodel_labels}
    overrides: dict[str, str] = {}
    for geo_common, classifier_label in existing.items():
        # Auto-matches map a geomodel name to a classifier label with the same
        # normalised key; anything else was hand-curated and must be kept.
        if normalize_common_name(geo_common) != normalize_common_name(classifier_label):
            overrides[geo_common] = classifier_label
        elif normalize_common_name(geo_common) not in auto_geo_keys:
            overrides[geo_common] = classifier_label
    return overrides


def print_unmatched_table(unmatched: list[str]) -> None:
    """Print the classifier classes with no geomodel match, for manual curation.

    Parameters
    - unmatched: sorted classifier labels still lacking a geomodel counterpart.
    """
    if not unmatched:
        print("\nAll classifier classes are mapped. Nothing to curate.")
        return
    width = max(len(label) for label in unmatched)
    print(f"\n{len(unmatched)} classifier classes need manual curation:\n")
    print(f"  {'classifier_label'.ljust(width)}    geomodel_common_name (fill in)")
    print(f"  {'-' * width}    {'-' * 30}")
    for label in unmatched:
        print(f"  {label.ljust(width)}    ")
    print(
        '\nAdd each as a "<geomodel_common_name>": "<classifier_label>" entry to\n'
        f"  {os.path.relpath(OUTPUT_PATH, REPO_ROOT)}\n"
        "then re-run this tool (curated entries are preserved)."
    )


def main() -> None:
    """Build the geomodel->classifier label map and report unmatched classes."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--check",
        action="store_true",
        help="print the unmatched table without writing the JSON",
    )
    args = parser.parse_args()

    geomodel_labels = load_labels(GEOMODEL_LABELS_PATH)
    classifier_labels = load_classifier_labels(CLASS_TO_IDX_PATH)
    overrides = load_existing_overrides(OUTPUT_PATH, geomodel_labels)

    mapping, unmatched = build_name_mapping(
        geomodel_labels, classifier_labels, overrides
    )

    print(f"geomodel labels:      {len(geomodel_labels)}")
    print(f"classifier classes:   {len(classifier_labels)}")
    print(
        f"mapped (auto+curated): {len(mapping)}  (curated overrides: {len(overrides)})"
    )
    print(f"unmatched classes:    {len(unmatched)}")

    if not args.check:
        with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
            json.dump(dict(sorted(mapping.items())), f, ensure_ascii=False, indent=2)
            f.write("\n")
        print(f"\nwrote {os.path.relpath(OUTPUT_PATH, REPO_ROOT)}")

    print_unmatched_table(unmatched)


if __name__ == "__main__":
    main()
