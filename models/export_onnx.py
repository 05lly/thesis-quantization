import torch
import torch.nn as nn
from torchvision import models

print("1. 加载模型...")
model = models.resnet18(pretrained=False)
model.fc = nn.Linear(512, 10)

print("2. 加载 FP32 权重...")
state_dict = torch.load("fp32_resnet18_best.pth", map_location='cpu')
model.load_state_dict(state_dict)
model.eval()

print("3. 导出 FP32 ONNX...")
dummy_input = torch.randn(1, 3, 224, 224)
torch.onnx.export(model, dummy_input, "resnet18_fp32.onnx",
                  input_names=["input"], output_names=["output"],
                  opset_version=11)

print("导出成功: resnet18_fp32.onnx")