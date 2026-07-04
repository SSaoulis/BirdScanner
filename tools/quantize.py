"""Dev script: dynamically quantize the ConvNeXt ONNX model to int8."""

import onnx
from onnxruntime.quantization import QuantType, quantize_dynamic

PATH = "models/convnext_v2_tiny.onnx"
INFER_MODEL_PATH = "models/convnext_v2_tiny_infer.onnx"
OUT_MODEL = "models/convnext_v2_tiny_int8.onnx"

# print onnx version
print(onnx.__version__)
quantize_dynamic(
    model_input=INFER_MODEL_PATH,
    model_output=OUT_MODEL,
    weight_type=QuantType.QUInt8,
)


m = onnx.load(OUT_MODEL)
ops = sorted(set(n.op_type for n in m.graph.node))
print(ops)
