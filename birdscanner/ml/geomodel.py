"""Geomodel (spatio-temporal species prior) inference and label crosswalk.

Loads the BirdNET geomodel, runs it for a location to get per-species occurrence
priors, and maps the geomodel's ~12k species labels onto the classifier's class
labels so the two can be combined (see :func:`build_name_mapping`).
"""

import json
import re
import unicodedata
from datetime import datetime
from typing import NamedTuple

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


def load_name_mapping(path: str) -> dict[str, str]:
    """Load the curated ``{classifier_label: geomodel_common_name}`` crosswalk.

    The file is the flat, sorted JSON object produced by
    ``tools/build_geomodel_map.py`` (auto-matches + hand-curated synonyms/proxies).

    Parameters
    - path: path to ``geomodel_classifier_map.json``.

    Returns
    - the ``classifier_label -> geomodel_common_name`` mapping.
    """
    with open(path, encoding="utf-8") as f:
        mapping: dict[str, str] = json.load(f)
    return mapping


def project_to_classifier(
    predictions: np.ndarray,
    geomodel_labels: list[GeomodelLabel],
    mapping: dict[str, str],
) -> dict[str, list[float]]:
    """Project the geomodel's per-week priors onto the classifier's classes.

    The geomodel emits a ``(NUM_WEEKS, n_geomodel_species)`` array whose columns are
    in geomodel label order. For each ``classifier_label -> geomodel_common_name``
    pair in ``mapping`` this slices out that species' column, producing the 48-week
    occurrence prior for the classifier class. Because the mapping is many-to-one,
    several classifier classes can borrow the same geomodel column (e.g. after an
    eBird lump). Classifier classes whose mapped geomodel name is absent from the
    label set are dropped (their prior is simply unavailable).

    Parameters
    - predictions: geomodel output of shape ``(NUM_WEEKS, n_geomodel_species)``.
    - geomodel_labels: rows from :func:`load_labels`; index order = column order.
    - mapping: ``classifier_label -> geomodel_common_name`` (see
      :func:`load_name_mapping` / :func:`build_name_mapping`).

    Returns
    - ``{classifier_label: [prob_week_1, ..., prob_week_NUM_WEEKS]}`` for every
      classifier class whose geomodel species is present in ``predictions``.
    """
    # First occurrence wins on a duplicate common name, matching build_name_mapping.
    index_by_common: dict[str, int] = {}
    for idx, row in enumerate(geomodel_labels):
        index_by_common.setdefault(row["common"], idx)

    priors: dict[str, list[float]] = {}
    for classifier_label, geo_common in mapping.items():
        column = index_by_common.get(geo_common)
        if column is None:
            continue
        priors[classifier_label] = predictions[:, column].astype(float).tolist()
    return priors


def compute_classifier_priors(
    model_path: str,
    labels_path: str,
    mapping_path: str,
    lat: float,
    lon: float,
) -> dict[str, list[float]]:
    """Compute the classifier-aligned 48-week occurrence prior for a location.

    Ties together the geomodel inference and the label crosswalk: runs the geomodel
    over all weeks for ``(lat, lon)``, loads the geomodel labels and the curated
    mapping, and projects the result onto the classifier's classes. This is the only
    entry point here that loads the ONNX model / touches onnxruntime.

    Parameters
    - model_path: path to the geomodel ONNX file.
    - labels_path: path to the geomodel ``*_Labels.txt`` file.
    - mapping_path: path to ``geomodel_classifier_map.json``.
    - lat: latitude of the location, in degrees.
    - lon: longitude of the location, in degrees.

    Returns
    - ``{classifier_label: [48 floats in [0, 1]]}`` for every mappable classifier class.
    """
    predictions = generate_grid_prediction(model_path, lat, lon)
    geomodel_labels = load_labels(labels_path)
    mapping = load_name_mapping(mapping_path)
    return project_to_classifier(predictions, geomodel_labels, mapping)


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


# Floor applied to each geomodel occurrence prior before the Bayesian update. A
# species the geomodel deems absent (probability ~0) is thereby made *unlikely*
# after the update, not *impossible*: a hard 0 would zero out that class' posterior
# irrecoverably, so a genuine out-of-range vagrant could never be reported.
DEFAULT_PRIOR_FLOOR = 1e-4

# Prior assigned to classifier classes with no geomodel counterpart (e.g. the
# non-bird ``Unknown`` sentinel, or any class the crosswalk could not map). 1.0 is
# neutral — it leaves the classifier's probability for that class unchanged.
DEFAULT_UNMAPPED_PRIOR = 1.0

# Number of highest pre-normalised scores retained per detection for debugging.
DEFAULT_TOP_K = 10


def week_of_year(when: datetime) -> int:
    """Map a date to a geomodel week index in ``1..NUM_WEEKS``.

    The geomodel's temporal axis is 48 "weeks" — four per month — so this splits
    each month into quarters (days 1-7 -> 1, 8-14 -> 2, 15-21 -> 3, 22+ -> 4) and
    offsets by the month. It approximates the geomodel's own binning (the exact
    day-of-year boundaries are not published) but lands in the correct
    month-quarter, which is all the coarse spatio-temporal prior needs.

    Parameters
    - when: the date/datetime to bin.

    Returns
    - a week index in ``[1, NUM_WEEKS]``.
    """
    quarter = min(4, (when.day - 1) // 7 + 1)
    return (when.month - 1) * 4 + quarter


def build_prior_matrix(
    priors: dict[str, list[float]],
    idx_to_class: dict[int, str],
    *,
    floor: float = DEFAULT_PRIOR_FLOOR,
    unmapped_prior: float = DEFAULT_UNMAPPED_PRIOR,
) -> np.ndarray:
    """Align the per-species weekly priors to the classifier's output order.

    Produces a ``(NUM_WEEKS, n_classes)`` matrix whose column ``i`` is the 48-week
    occurrence prior for the classifier class at index ``i``. Each stored
    probability is floored at ``floor`` (absent species made unlikely, not
    impossible). Classes missing from ``priors`` — the ``Unknown`` sentinel, or any
    class the crosswalk could not map — get a constant ``unmapped_prior`` so the
    Bayesian update leaves them untouched. Rows whose weekly vector is not
    ``NUM_WEEKS`` long are ignored (defensive against a malformed store).

    Parameters
    - priors: ``{classifier_label: [NUM_WEEKS probabilities]}`` from the geo store.
    - idx_to_class: the classifier's ``{index: label}`` map (its output order).
    - floor: minimum prior applied to every stored probability.
    - unmapped_prior: prior for classes absent from ``priors``.

    Returns
    - a ``(NUM_WEEKS, n_classes)`` float64 array in classifier-index column order.
    """
    n_classes = len(idx_to_class)
    matrix = np.full((NUM_WEEKS, n_classes), unmapped_prior, dtype=np.float64)
    class_to_idx = {label: idx for idx, label in idx_to_class.items()}
    for label, weekly in priors.items():
        idx = class_to_idx.get(label)
        if idx is None or len(weekly) != NUM_WEEKS:
            continue
        matrix[:, idx] = np.maximum(np.asarray(weekly, dtype=np.float64), floor)
    return matrix


def geomodel_posterior(
    classifier_probs: np.ndarray, prior: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Combine a classifier distribution with a geomodel prior (Bayesian update).

    Computes ``p(y | x, c) = p(y | x) p(y | c) / sum_y' p(y' | x) p(y' | c)`` where
    ``x`` is the classifier softmax (``classifier_probs``) and ``c`` the geomodel
    occurrence prior (``prior``). Returns both the normalised posterior and the
    *pre-normalised* product ``p(y | x) p(y | c)`` — the latter is persisted per
    detection for later inspection.

    Parameters
    - classifier_probs: the classifier's softmax vector, shape ``(n_classes,)``.
    - prior: the geomodel prior vector for the week, shape ``(n_classes,)``.

    Returns
    - ``(posterior, unnormalized)``; ``posterior`` sums to 1. When the product sums
      to 0 (only reachable with a zero floor) the classifier probabilities are
      returned unchanged so a detection is never lost.
    """
    unnormalized = classifier_probs * prior
    total = float(unnormalized.sum())
    if total <= 0.0:
        return classifier_probs.astype(np.float64), unnormalized
    return unnormalized / total, unnormalized


class GeoAdjustment(NamedTuple):
    """Result of applying the geomodel prior to one classifier distribution.

    Attributes:
        species: The geomodel-corrected prediction (posterior argmax).
        confidence: The posterior probability of ``species``, in ``[0, 1]``.
        classifier_species: The classifier's own top class (pre-adjustment).
        classifier_confidence: The classifier's softmax probability for its top
            class, in ``[0, 1]``.
        top_scores: The highest pre-normalised ``(species, score)`` pairs
            (descending), for debugging what the update boosted/suppressed.
    """

    species: str
    confidence: float
    classifier_species: str
    classifier_confidence: float
    top_scores: list[tuple[str, float]]


class GeoPriorAdjuster:
    """Applies the stored geomodel prior to a classifier distribution at runtime.

    Built once at detector startup from the priors persisted in the DB and the
    classifier's ``idx_to_class`` map; it precomputes the aligned
    ``(NUM_WEEKS, n_classes)`` prior matrix so each detection is a cheap row slice
    plus multiply. :meth:`adjust` returns the geomodel-corrected prediction (the
    posterior argmax), the classifier's own top class, and the top-K pre-normalised
    scores for debugging.
    """

    def __init__(
        self,
        priors: dict[str, list[float]],
        idx_to_class: dict[int, str],
        *,
        floor: float = DEFAULT_PRIOR_FLOOR,
        unmapped_prior: float = DEFAULT_UNMAPPED_PRIOR,
        top_k: int = DEFAULT_TOP_K,
    ) -> None:
        """Precompute the classifier-aligned prior matrix.

        Args:
            priors: ``{classifier_label: [NUM_WEEKS probabilities]}`` from the store.
            idx_to_class: the classifier's ``{index: label}`` output map.
            floor: minimum prior applied to every stored probability.
            unmapped_prior: prior for classes absent from ``priors``.
            top_k: how many highest pre-normalised scores to retain per detection.
        """
        self._idx_to_class = idx_to_class
        self._matrix = build_prior_matrix(
            priors, idx_to_class, floor=floor, unmapped_prior=unmapped_prior
        )
        self._top_k = top_k

    def adjust(self, classifier_probs: np.ndarray, week: int) -> GeoAdjustment:
        """Apply the week's geomodel prior to a classifier softmax vector.

        Args:
            classifier_probs: the classifier's softmax vector, shape ``(n_classes,)``.
            week: the geomodel week index (``1..NUM_WEEKS``; see :func:`week_of_year`).

        Returns:
            The :class:`GeoAdjustment` with the corrected + original predictions.
        """
        probs = np.asarray(classifier_probs, dtype=np.float64).reshape(-1)
        prior = self._matrix[self._week_index(week)]
        posterior, unnormalized = geomodel_posterior(probs, prior)

        post_idx = int(np.argmax(posterior))
        clf_idx = int(np.argmax(probs))
        return GeoAdjustment(
            species=self._label(post_idx),
            confidence=float(posterior[post_idx]),
            classifier_species=self._label(clf_idx),
            classifier_confidence=float(probs[clf_idx]),
            top_scores=self._top_scores(unnormalized),
        )

    @staticmethod
    def _week_index(week: int) -> int:
        """Clamp a 1-based week to a valid 0-based matrix row."""
        return min(max(week, 1), NUM_WEEKS) - 1

    def _label(self, idx: int) -> str:
        """Return the class label for an index (a placeholder if out of range)."""
        return self._idx_to_class.get(idx, f"<unknown:{idx}>")

    def _top_scores(self, unnormalized: np.ndarray) -> list[tuple[str, float]]:
        """Return the top-K ``(label, score)`` pairs of the pre-normalised vector."""
        k = min(self._top_k, unnormalized.shape[0])
        top_idx = np.argsort(unnormalized)[::-1][:k]
        return [(self._label(int(i)), float(unnormalized[i])) for i in top_idx]
