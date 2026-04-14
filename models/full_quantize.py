import onnx
from onnxruntime.quantization import quantize_dynamic, QuantType
from onnxruntime.quantization import shape_inference
import os

print("1. 加载原始模型...")
model = onnx.load("resnet18_fp32.onnx")

print("2. 进行形状推断...")
try:
    # 尝试形状推断
    inferred_model = shape_inference.infer_shapes(model)
    onnx.save(inferred_model, "resnet18_fp32_with_shape.onnx")
    print("   形状推断成功")
except Exception as e:
    print(f"   形状推断失败: {e}")
    print("   使用原始模型继续...")
    onnx.save(model, "resnet18_fp32_with_shape.onnx")

print("3. 进行动态量化...")
try:
    quantize_dynamic(
        "resnet18_fp32_with_shape.onnx",
        "resnet18_int8.onnx",
        weight_type=QuantType.QInt8
    )
    print("   量化成功!")
except Exception as e:
    print(f"   量化失败: {e}")
    
print("4. 检查生成的文件...")
if os.path.exists("resnet18_int8.onnx"):
    size = os.path.getsize("resnet18_int8.onnx") / 1024**2
    print(f"   INT8 模型已生成，大小: {size:.2f} MB")
else:
    print("   INT8 模型生成失败")