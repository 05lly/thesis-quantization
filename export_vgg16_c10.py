import torch
import torch.nn as nn
from torchvision import models
import os
import time

# --- 1. 环境配置 ---
torch.backends.quantized.engine = 'qnnpack'
model_dir = "/root/autodl-tmp/my_backup" if os.path.exists("/root/autodl-tmp") else "models"

# 路径对齐
best_qat_path = os.path.join(model_dir, "vgg16_c10_qat_best.pth")
fp32_path = os.path.join(model_dir, "fp32_vgg16_best.pth")
deploy_path = os.path.join(model_dir, "vgg16_c10_int8_deploy.pt")

# --- 2. 模型结构 (必须一致) ---
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
        x = self.quant(x)
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        x = self.dequant(x)
        return x

    def fuse_model(self):
        for m in self.modules():
            if type(m) == nn.Sequential:
                for i in range(len(m)):
                    if i + 1 < len(m) and type(m[i]) == nn.Conv2d and type(m[i+1]) == nn.ReLU:
                        torch.ao.quantization.fuse_modules(m, [str(i), str(i+1)], inplace=True)
                    elif i + 1 < len(m) and type(m[i]) == nn.Linear and type(m[i+1]) == nn.ReLU:
                        torch.ao.quantization.fuse_modules(m, [str(i), str(i+1)], inplace=True)

# --- 3. 转换逻辑 ---
start_time = time.time()
print(f"[*] Starting Export for VGG16...")

export_model = QuantizableVGG16(num_classes=10)
export_model.eval()
export_model.fuse_model()

export_model.train() 
export_model.qconfig = torch.ao.quantization.get_default_qat_qconfig('qnnpack')
torch.ao.quantization.prepare_qat(export_model, inplace=True)

# 加载训练好的权重
if not os.path.exists(best_qat_path):
    print(f"CRITICAL ERROR: {best_qat_path} not found!")
    exit()
export_model.load_state_dict(torch.load(best_qat_path, map_location='cpu'))

# 转换
export_model.to('cpu').eval()
int8_model = torch.ao.quantization.convert(export_model, inplace=False)

# 追踪导出
example_input = torch.randn(1, 3, 224, 224)
traced_model = torch.jit.trace(int8_model, example_input)
torch.jit.save(traced_model, deploy_path)

# --- 4. 报表输出 (完全对齐) ---
def get_size_mb(path):
    return os.path.getsize(path) / (1024 * 1024) if os.path.exists(path) else 0

fp32_size = get_size_mb(fp32_path)
int8_size = get_size_mb(deploy_path)

print("\n" + "=" * 55)
print("QAT Process Finished.")
print(f"Deployment Model Saved: {deploy_path}")
print(f"FP32 Model Size: {fp32_size:.2f} MB")
print(f"INT8 Deploy Size: {int8_size:.2f} MB")
print(f"Compression Ratio: {fp32_size/int8_size:.2f}x")
print(f"Export Time: {(time.time()-start_time):.2f} seconds")
print("=" * 55)
print("Experiment Complete. Ready for Raspberry Pi 5.")