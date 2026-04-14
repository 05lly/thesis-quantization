from onnxruntime.quantization import shape_inference
import onnx

# 加载模型
model = onnx.load("resnet18_fp32.onnx")
# 进行形状推断
inferred_model = shape_inference.infer_shapes(model)
# 保存修复后的模型
onnx.save(inferred_model, "resnet18_fp32_fixed.onnx")
print("预处理完成，已生成模型: resnet18_fp32_fixed.onnx")
