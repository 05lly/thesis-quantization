# convert_int8_final_gpu.py
import torch
from torchvision import datasets, transforms
from qat_resnet18 import resnet18_qat  # 你的 QAT 模型定义文件

# ==============================
# 配置部分
# ==============================
qat_path = './models/resnet18_qat_best.pth'  # QAT训练好的模型路径
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# ==============================
# 实例化模型
# ==============================
model_qat = resnet18_qat(num_classes=10)  # CIFAR-10 是10类
model_qat.to(device)

# ==============================
# 加载训练好的 QAT 权重
# ==============================
print("Loading QAT checkpoint...")
checkpoint = torch.load(qat_path, map_location=device)
model_qat.load_state_dict(checkpoint)
model_qat.eval()  # 切换到评估模式

# ==============================
# 将 QAT 模型转换为 INT8
# ==============================
print("Converting model to INT8...")
# QAT 模型通常在训练后就带 fake-quant，直接转换
model_int8 = torch.quantization.convert(model_qat.eval(), inplace=False)
model_int8.to(device)
model_int8.eval()
print("INT8 conversion successful")

# ==============================
# 准备 CIFAR-10 测试数据
# ==============================
transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465),
                         (0.2470, 0.2435, 0.2616))
])

test_dataset = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=128, shuffle=False, num_workers=2)

# ==============================
# 测试 INT8 模型精度
# ==============================
print("Evaluating INT8 model...")
correct = 0
total = 0

with torch.no_grad():
    for images, labels in test_loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model_int8(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

accuracy = 100 * correct / total
print(f"INT8 model accuracy: {accuracy:.2f}%")