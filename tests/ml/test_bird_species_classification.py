"""Species-classification accuracy on the real int8 ONNX model.

Each labelled test image (``tests/_test_images/``) is cropped to its bird box
with the same :func:`preprocess_roi` the live pipeline uses (crop to a padded
square), then classified by the real ConvNeXt int8 model via the standard
``setup_classifier`` preprocessing (resize to 384x384 + ImageNet normalise). The
prediction must equal the expected species recorded in the bounding-box manifest.

The ``real_classifier`` and ``bird_image_cases`` fixtures (see
``tests/ml/conftest.py``) skip the test when the ONNX model or the JPEG fixtures
are absent, matching the rest of the model-dependent suite.
"""

from birdscanner.ml.detection_utils import preprocess_roi

# Every fixture bird is a clear, tightly-boxed subject, so a correct prediction
# should be well clear of the pipeline's 0.4 save threshold.
_MIN_CONFIDENCE = 0.4


def test_labelled_images_classify_to_expected_species(
    real_classifier, bird_image_cases
):
    """Each labelled crop classifies to its expected species with high confidence."""
    for case in bird_image_cases:
        roi, _ = preprocess_roi(case.image, case.box)
        species, confidence = real_classifier.classify(roi)

        assert species == case.species, (
            f"{case.name}: expected {case.species!r}, got {species!r} "
            f"(confidence {confidence:.3f})"
        )
        assert 0.0 <= confidence <= 1.0
        assert confidence > _MIN_CONFIDENCE
