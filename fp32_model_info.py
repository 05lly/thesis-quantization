import torch
import os
from torchvision import models
import torch.nn as nn

def count_parameters(model):
    return sum(p.numel() for p in model.parameters())

def file_size_mb(path):
    return os.path.getsize(path) / (1024 * 1024)

results = []

# =========================
# ResNet18
# =========================
resnet = models.resnet18(pretrained=False)
resnet.fc = nn.Linear(resnet.fc.in_features, 10)
resnet_params = count_parameters(resnet)

resnet_path = "models/resnet18_fp32.pth"
resnet_size = file_size_mb(resnet_path) if os.path.exists(resnet_path) else None

results.append(("ResNet18", resnet_params, resnet_size))

# =========================
# MobileNetV2
# =========================
mobilenet = models.mobilenet_v2(pretrained=False)
mobilenet.classifier[1] = nn.Linear(mobilenet.last_channel, 10)
mobilenet_params = count_parameters(mobilenet)

mobilenet_path = "models/mobilenetv2_fp32.pth"
mobilenet_size = file_size_mb(mobilenet_path) if os.path.exists(mobilenet_path) else None

results.append(("MobileNetV2", mobilenet_params, mobilenet_size))

# =========================
# VGG16
# =========================
vgg = models.vgg16(pretrained=False)
vgg.classifier[6] = nn.Linear(4096, 10)
vgg_params = count_parameters(vgg)

vgg_path = "vgg16_fp32_best.pth"
vgg_size = file_size_mb(vgg_path) if os.path.exists(vgg_path) else None

results.append(("VGG16", vgg_params, vgg_size))

# =========================
# 打印结果
# =========================
print("=" * 65)
print(f"{'模型':<15}{'参数数量':<18}{'FP32模型大小(MB)'}")
print("-" * 65)

for name, params, size in results:
    print(f"{name:<15}{params:<18,}{size:>10.2f} MB")

print("=" * 65)
