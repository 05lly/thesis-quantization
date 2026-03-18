import torch
import torch.nn as nn
from torchvision import datasets, transforms, models
import os
from tqdm import tqdm

# --- 配置 ---
torch.backends.quantized.engine = 'qnnpack'
model_dir = "/root/autodl-tmp/my_backup"
ckpt_path = os.path.join(model_dir, "resnet18_qat_best.pth")

# --- 1. 定义与 QAT 训练时完全一致的结构 ---
model = models.quantization.resnet18(weights=None, quantize=False)
model.fc = nn.Linear(model.fc.in_features, 10)

# --- 2. 模拟训练时的准备动作 ---
model.eval()
model.fuse_model(is_qat=True)
model.qconfig = torch.ao.quantization.get_default_qat_qconfig('qnnpack')
torch.ao.quantization.prepare_qat(model, inplace=True)

# --- 3. 加载你辛苦跑出来的权重 ---
if not os.path.exists(ckpt_path):
    print(f"❌ 错误: 未找到权重文件 {ckpt_path}")
    exit()
model.load_state_dict(torch.load(ckpt_path, map_location='cpu'))

# --- 4. 【关键步】执行物理转换 (Real INT8) ---
# 这一步会将模型中的权重真正转为 int8 类型
model_int8 = torch.ao.quantization.convert(model, inplace=False)

# --- 5. 实测准确率 ---
transform = transforms.Compose([
    transforms.Resize(224),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])
test_loader = torch.utils.data.DataLoader(
    datasets.CIFAR10('./data', train=False, transform=transform), batch_size=128)

correct = 0
with torch.no_grad():
    for inputs, labels in tqdm(test_loader, desc="Testing Real ResNet18 INT8"):
        outputs = model_int8(inputs) # 在 CPU 上跑真实量化算子
        _, pred = torch.max(outputs, 1)
        correct += (pred == labels).sum().item()

print(f"\n" + "="*40)
print(f"ResNet18 REAL INT8 Accuracy: {100. * correct / 10000:.2f}%")
print("="*40)