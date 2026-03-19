import torch
import torch.nn as nn
from torchvision import datasets, transforms, models
import os
from tqdm import tqdm

# --- 配置 ---
torch.backends.quantized.engine = 'qnnpack'
model_dir = "/root/autodl-tmp/my_backup"
ckpt_path = os.path.join(model_dir, "vgg16_qat_best.pth")

# --- 1. 定义完全一致的结构 ---
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

# --- 2. 准备物理转换 ---
model = QuantizableVGG16(num_classes=10)
model.train() # 必须先设为 train
model.fuse_model()
model.qconfig = torch.ao.quantization.get_default_qat_qconfig('qnnpack')
torch.ao.quantization.prepare_qat(model, inplace=True)

# 加载权重
model.load_state_dict(torch.load(ckpt_path, map_location='cpu'))

# 执行物理转换 (关键一步：Float -> Int8)
model.eval() 
model_int8 = torch.ao.quantization.convert(model, inplace=False)

# --- 3. 实测真 INT8 精度 ---
transform = transforms.Compose([
    transforms.Resize(224),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])
test_loader = torch.utils.data.DataLoader(
    datasets.CIFAR10('./data', train=False, transform=transform), batch_size=128)

correct = 0
with torch.no_grad():
    for inputs, labels in tqdm(test_loader, desc="VGG16 REAL INT8 Test"):
        outputs = model_int8(inputs) 
        _, pred = torch.max(outputs, 1)
        correct += (pred == labels).sum().item()

print(f"\n" + "="*45)
print(f"VGG16 REAL INT8 Final Accuracy: {100. * correct / 10000:.2f}%")
print("="*45)