import torch
import torch.nn as nn
from torchvision import datasets, transforms, models
import os
from tqdm import tqdm

# --- 配置 ---
torch.backends.quantized.engine = 'qnnpack'
model_dir = "/root/autodl-tmp/my_backup"
ckpt_path = os.path.join(model_dir, "mobilenetv2_qat_best.pth")

# --- 1. 定义结构 ---
model = models.quantization.mobilenet_v2(weights=None, quantize=False)
model.classifier[1] = nn.Linear(model.last_channel, 10)

# --- 2. 绕过断言的准备动作 ---
model.train() 
model.fuse_model(is_qat=True)
model.qconfig = torch.ao.quantization.get_default_qat_qconfig('qnnpack')
torch.ao.quantization.prepare_qat(model, inplace=True)

# --- 3. 加载权重 ---
if not os.path.exists(ckpt_path):
    print(f"❌ Error: {ckpt_path} not found.")
    exit()
model.load_state_dict(torch.load(ckpt_path, map_location='cpu'))

# --- 4. 执行物理转换 ---
model.eval() 
model_int8 = torch.ao.quantization.convert(model, inplace=False)

# --- 5. 实测 ---
transform = transforms.Compose([
    transforms.Resize(224),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])
test_loader = torch.utils.data.DataLoader(
    datasets.CIFAR10('./data', train=False, transform=transform), batch_size=128)

correct = 0
with torch.no_grad():
    for inputs, labels in tqdm(test_loader, desc="MobileNetV2 REAL INT8 Test"):
        outputs = model_int8(inputs)
        _, pred = torch.max(outputs, 1)
        correct += (pred == labels).sum().item()

print(f"\n" + "="*40)
print(f"MobileNetV2 REAL INT8 Accuracy: {100. * correct / 10000:.2f}%")
print("="*40)