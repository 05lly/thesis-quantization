import torch
import torch.nn as nn
from torchvision import models
import os

# --- 配置路径与参数 ---
model_dir = "/root/autodl-tmp/my_backup"
num_classes = 10

def get_vgg16_structure():
    """重建自定义QuantizableVGG16结构"""
    class QuantizableVGG16(nn.Module):
        def __init__(self, num_classes=10):
            super(QuantizableVGG16, self).__init__()
            vgg = models.vgg16(weights=None)
            self.features = vgg.features
            self.avgpool = vgg.avgpool
            self.classifier = vgg.classifier
            self.classifier[6] = nn.Linear(self.classifier[6].in_features, num_classes)
            self.quant = torch.ao.quantization.QuantStub()
            self.dequant = torch.ao.quantization.DeQuantStub()

        def forward(self, x):
            x = self.quant(x); x = self.features(x); x = self.avgpool(x)
            x = torch.flatten(x, 1); x = self.classifier(x); x = self.dequant(x)
            return x
    return QuantizableVGG16(num_classes=num_classes)

def export_model(model_name, pth_filename, pt_filename, structure_func=None):
    print(f"\n>>> 正在处理: {model_name}")
    pth_path = os.path.join(model_dir, pth_filename)
    pt_path = os.path.join(model_dir, pt_filename)
    
    if not os.path.exists(pth_path):
        print(f"未找到权重文件 {pth_path}")
        return

    # 1. 结构重建
    try:
        if structure_func:
            model = structure_func()
        elif model_name == "ResNet18":
            # 训练时用了 models.quantization.resnet18
            model = models.quantization.resnet18(weights=None, quantize=False)
            model.fc = nn.Linear(model.fc.in_features, num_classes)
        elif model_name == "MobileNetV2":
            # 训练时用了标准 models.mobilenet_v2
            model = models.quantization.mobilenet_v2(weights=None, quantize=False)
            model.classifier[1] = nn.Linear(model.last_channel, num_classes)
        # 2. 加载权重
        checkpoint = torch.load(pth_path, map_location='cpu', weights_only=True)
        model.load_state_dict(checkpoint, strict=False)
        model.eval()
        # 3. TorchScript 追踪 (Tracing)
        dummy_input = torch.randn(1, 3, 224, 224)
        traced_model = torch.jit.trace(model, dummy_input)
        # 4. 保存部署文件
        torch.jit.save(traced_model, pt_path)
        print(f"成功生成部署文件: {pt_filename}")
        print(f"文件大小: {os.path.getsize(pt_path)/1024/1024:.2f} MB")
    except Exception as e:
        print(f"错误：转换 {model_name} 时发生异常: {e}")

if __name__ == "__main__":
    # 1. MobileNetV2
    export_model("MobileNetV2", "fp32_mobilenetv2_best.pth", "mobilenetv2_c10_fp32_deploy.pt")
    # 2. ResNet18
    export_model("ResNet18", "fp32_resnet18_best.pth", "resnet18_c10_fp32_deploy.pt")
    # 3. VGG16
    export_model("VGG16", "fp32_vgg16_best.pth", "vgg16_c10_fp32_deploy.pt", structure_func=get_vgg16_structure)
    print("\n所有CIFAR-10 FP32部署文件处理完毕")