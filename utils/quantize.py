# from onnxruntime.quantization import quantize_dynamic, QuantType
import onnx
path = "models/convnext_v2_tiny.onnx"
infer_model_path = "models/convnext_v2_tiny_infer.onnx"
out_model = "models/convnext_v2_tiny_int8.onnx"
# onnx.load(path)
# onnx.checker.check_model(path)
from onnxruntime.quantization import quantize_dynamic, QuantType
#print onxx version
print(onnx.__version__)
quantize_dynamic(
    model_input=infer_model_path,
    model_output=out_model,
    weight_type=QuantType.QUInt8,
    # op_types_to_quantize=['Add', 'Cast', 'Constant', 'Div', 'DynamicQuantizeLinear', 'Erf', 'Flatten', 'GlobalAveragePool', 'LayerNormalization', 'MatMulInteger', 'Mul', 'ReduceL2', 'ReduceMean', 'Reshape', 'Transpose']
    # per_channel=True,
    # extra_options={
    #     "EnableSubgraph": True,
    #     "ForceQuantizeNoInputCheck": True,
    # }
)


m = onnx.load(out_model)
ops = sorted(set(n.op_type for n in m.graph.node))
print(ops)