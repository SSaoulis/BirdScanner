import birder
from birder.inference.classification import infer_image

(net, model_info) = birder.load_pretrained_model("convnext_v2_tiny_eu-common", inference=True)
# Note: A 256x256 variant is available as "uniformer_s_eu-common256px"

# Get the image size the model was trained on
size = birder.get_size_from_signature(model_info.signature)

# Create an inference transform
transform = birder.classification_transform(size, model_info.rgb_stats)

image = "path/to/image.jpeg"  # or a PIL image, must be loaded in RGB format
# (out, _) = infer_image(net, image, transform)
# out is a NumPy array with shape of (1, 707), representing class probabilities.
