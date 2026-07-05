"""Geomodel (spatio-temporal species prior) inference and label crosswalk.

Loads the BirdNET geomodel, runs it for a location to get per-species occurrence
priors, and maps the geomodel's ~12k species labels onto the classifier's class
labels so the two can be combined (see :func:`build_name_mapping`).
"""

import re
import unicodedata

import onnxruntime as ort
import numpy as np

# 48 was picked as the authors of the original model used 48 weeks in training data, 4 per month.
NUM_WEEKS = 48

# A single geomodel label row: {"id": species_id, "scientific": name, "common": name}.
GeomodelLabel = dict[str, str]


def load_labels(path: str) -> list[GeomodelLabel]:
    """Load the tab-separated geomodel label file.

    Each non-blank line is ``species_id\\tscientific_name\\tcommon_name``. File order is
    preserved because it is the geomodel's output index order.

    Parameters
    - path: path to the ``*_Labels.txt`` file shipped with the geomodel.

    Returns
    - one dict per species with ``id``/``scientific``/``common`` keys, in index order.
    """
    labels: list[GeomodelLabel] = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.rstrip("\n")
            if not line:
                continue
            species_id, scientific, common = line.split("\t")
            labels.append(
                {
                    "id": species_id,
                    "scientific": scientific,
                    "common": common,
                }
            )
    return labels


def generate_grid_prediction(model_path: str, lat: float, lon: float) -> np.ndarray:
    """Run the geomodel over all weeks of the year for one location.

    Parameters
    - model_path: path to the geomodel ONNX file.
    - lat: latitude of the location, in degrees.
    - lon: longitude of the location, in degrees.

    Returns
    - array of shape ``(NUM_WEEKS, n_species)`` of sigmoid probabilities in ``[0, 1]``.
    """
    session = ort.InferenceSession(model_path)
    weeks = np.arange(1, NUM_WEEKS + 1, dtype=np.float32)

    # Shape (NUM_WEEKS, 3), columns = [lat, lon, week]
    inputs = np.stack(
        [
            np.full(NUM_WEEKS, lat, dtype=np.float32),
            np.full(NUM_WEEKS, lon, dtype=np.float32),
            weeks,
        ],
        axis=1,
    )

    # returns a tuple of (weeks, n_species) — sigmoid probabilities in [0, 1]

    pred = session.run(None, {"input": inputs})[0]

    return pred


def normalize_common_name(name: str) -> str:
    """Normalise a bird common name for cross-checklist matching.

    The geomodel (eBird/Clements naming, Title Case) and the classifier (IOC-style
    naming, sentence case, apostrophes/accents stripped) disagree on casing, punctuation
    and the British ``grey`` vs American ``gray`` spelling. This collapses all of those so
    that e.g. ``"Audouin's Gull"``, ``"Audouins gull"`` and ``"Grey Heron"`` /
    ``"Gray Heron"`` compare equal. It does **not** bridge genuine synonyms (e.g.
    ``"Common Blackbird"`` vs ``"Eurasian Blackbird"``) — those need curation.

    Parameters
    - name: a species common name from either label set.

    Returns
    - a lower-case, accent-free, alphanumeric-only key for equality comparison.
    """
    decomposed = unicodedata.normalize("NFKD", name)
    without_accents = "".join(ch for ch in decomposed if not unicodedata.combining(ch))
    unified_spelling = without_accents.lower().replace("grey", "gray")
    return re.sub(r"[^a-z0-9]", "", unified_spelling)


def build_name_mapping(
    geomodel_labels: list[GeomodelLabel],
    classifier_labels: list[str],
    overrides: dict[str, str] | None = None,
) -> tuple[dict[str, str], list[str]]:
    """Map each classifier class label to a geomodel common name.

    The result is keyed by the **classifier label** because the geomodel's per-species
    prior is projected *onto* the classifier's classes: for each classifier class a
    caller borrows the occurrence prior of its mapped geomodel species. This direction
    is **many-to-one** friendly — several classifier classes can share one geomodel
    species (e.g. after an eBird lump, both ``"Common redpoll"`` and ``"Arctic redpoll"``
    map to ``"Redpoll"``), which a geomodel-keyed dict could not represent.

    Each classifier class is matched to a geomodel row by comparing their
    ``normalize_common_name`` keys; ``overrides`` supplies hand-curated
    ``classifier_label -> geomodel_common_name`` pairs for the genuine synonyms and
    geospatial proxies that normalisation cannot bridge (added on top of, and taking
    precedence over, the auto-matches).

    Parameters
    - geomodel_labels: rows from :func:`load_labels` (the ~12k-species geomodel).
    - classifier_labels: the classifier's class labels (keys of its ``class_to_idx``).
    - overrides: optional ``classifier_label -> geomodel_common_name`` curated pairs.

    Returns
    - ``(mapping, unmatched)`` where ``mapping`` is ``classifier_label ->
      geomodel_common_name`` for every classifier class that has a counterpart, and
      ``unmatched`` is the sorted classifier labels still without one.
    """
    geo_by_key: dict[str, GeomodelLabel] = {}
    for row in geomodel_labels:
        # First occurrence wins, so the mapping is stable across re-runs.
        geo_by_key.setdefault(normalize_common_name(row["common"]), row)

    mapping: dict[str, str] = {}
    unmatched: list[str] = []
    for label in classifier_labels:
        geo_row = geo_by_key.get(normalize_common_name(label))
        if geo_row is None:
            unmatched.append(label)
        else:
            mapping[label] = geo_row["common"]

    # Curated pairs are already classifier_label -> geomodel_common_name.
    mapping.update(overrides or {})

    unmatched = sorted(label for label in unmatched if label not in mapping)
    return mapping, unmatched


def bayesian_update(prior: np.ndarray, likelihood: np.ndarray) -> np.ndarray:
    """
    Perform a Bayesian update given a prior and likelihood.

    Parameters
    - prior: numpy.ndarray of shape (n_species,) representing the prior probabilities.
    - likelihood: numpy.ndarray of shape (n_species,) representing the likelihoods.

    Returns
    - posterior: numpy.ndarray of shape (n_species,) with the updated posterior
      probabilities.
    """
    unnormalized_posterior = prior * likelihood
    posterior = unnormalized_posterior / np.sum(unnormalized_posterior)
    return posterior
