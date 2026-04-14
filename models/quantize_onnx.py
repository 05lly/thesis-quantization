import onnx
from onnxruntime.quantization import quantize_dynamic, QuantType

print("加载模型...")
model = onnx.load("resnet18_fp32.onnx")

print("检查模型输入输出...")
print(f"输入: {model.graph.input[0].name}")
print(f"输出: {model.graph.output[0].name}")

print("量化 INT8...")
try:
    quantize_dynamic(
        "resnet18_fp32.onnx",
        "resnet18_int8.onnx",
        weight_type=QuantType.QInt8,
        per_channel=False
    )
    print("导出成功: resnet18_int8.onnx")
except Exception as e:
    print(f"量化失败: {e}")
    print("尝试使用 alternate 方法...")
    # 备用方法：只量化权重，不量化激活
    quantize_dynamic(
        "resnet18_fp32.onnx",
        "resnet18_int8.onnx",
        weight_type=QuantType.QInt8,
        op_types_to_quantize=['Conv', 'Gemm'],
        per_channel=False
    )
    print("导出成功: resnet18_int8.onnx (使用备用方法)")