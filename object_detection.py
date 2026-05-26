import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
import os, time, datetime
from tqdm import tqdm
import numpy as np
import psutil
from torchao.quantization import quantize_, Int4WeightOnlyConfig, Int8DynamicActivationInt4WeightConfig
from torch.ao.quantization import prepare_qat, convert, get_default_qconfig_mapping
from torch.ao.quantization.fuse_modules import fuse_modules

# --- 1. 全局配置 ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
batch_size = 16
epochs = 10  # 减少训练轮数以加快进度
lr = 1e-3
model_dir = "models"
log_dir = "logs"
os.makedirs(model_dir, exist_ok=True)
os.makedirs(log_dir, exist_ok=True)

log_filename = os.path.join(log_dir, f"object_detection_quant_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.log")

def log_message(msg):
    t = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    full_msg = f"[{t}] {msg}"
    print(full_msg)
    with open(log_filename, "a", encoding="utf-8") as f:
        f.write(full_msg + "\n")

# --- 2. 数据准备（使用Pascal VOC或COCO数据集）---
# 这里使用一个简单的演示，实际使用时需要替换为完整的目标检测数据集
log_message("Preparing object detection dataset...")

# 简单的演示数据加载器
def get_dataloader():
    transform = transforms.Compose([
        transforms.Resize((640, 640)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    # 使用CIFAR10作为演示，实际应使用目标检测数据集
    train_dataset = datasets.CIFAR10('./data', train=True, download=True, transform=transform)
    test_dataset = datasets.CIFAR10('./data', train=False, download=True, transform=transform)
    
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
    
    return train_loader, test_loader

train_loader, test_loader = get_dataloader()

# --- 3. 目标检测模型定义 ---
# 简化的YOLOv5风格模型，适合演示和边缘设备部署
class SimpleYOLO(nn.Module):
    def __init__(self, num_classes=80):
        super(SimpleYOLO, self).__init__()
        self.num_classes = num_classes
        self.num_anchors = 3
        
        # 主干网络 - 轻量级设计
        self.backbone = nn.Sequential(
            # Conv1
            nn.Conv2d(3, 16, 3, 2, 1, bias=False),
            nn.BatchNorm2d(16),
            nn.LeakyReLU(0.1, inplace=True),
            # Conv2
            nn.Conv2d(16, 32, 3, 2, 1, bias=False),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.1, inplace=True),
            # Conv3
            nn.Conv2d(32, 64, 3, 2, 1, bias=False),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.1, inplace=True),
            # Conv4
            nn.Conv2d(64, 128, 3, 2, 1, bias=False),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.1, inplace=True),
            # Conv5
            nn.Conv2d(128, 256, 3, 2, 1, bias=False),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.1, inplace=True),
        )
        
        # 颈部网络
        self.neck = nn.Sequential(
            nn.Conv2d(256, 128, 1, 1, 0, bias=False),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(128, 256, 3, 1, 1, bias=False),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.1, inplace=True),
        )
        
        # 头部网络
        self.head = nn.Conv2d(256, self.num_anchors * (5 + self.num_classes), 1, 1, 0, bias=True)
    
    def forward(self, x):
        x = self.backbone(x)
        x = self.neck(x)
        x = self.head(x)
        # 输出格式: [batch, anchors*(5+classes), h, w]
        return x

# --- 4. 模型准备 ---
log_message("Initializing SimpleYOLO model...")
model = SimpleYOLO(num_classes=10)  # 使用CIFAR10的10个类别作为演示
model.to(device)

# --- 5. 训练目标检测模型 ---
log_message("Training object detection model...")

optimizer = optim.Adam(model.parameters(), lr=lr)
criterion = nn.MSELoss()  # 简化的损失函数，实际应使用YOLO损失

best_loss = float('inf')
for epoch in range(epochs):
    model.train()
    running_loss = 0.0
    
    for i, (inputs, labels) in enumerate(tqdm(train_loader, desc=f"Epoch [{epoch+1:02d}/{epochs}]", leave=False)):
        inputs = inputs.to(device)
        
        # 生成简化的目标检测标签（仅用于演示）
        batch_size, _, h, w = inputs.shape
        # 输出格式: [batch, anchors, h, w, 5+classes]
        target = torch.zeros(batch_size, 3, h//32, w//32, 15).to(device)
        
        optimizer.zero_grad()
        outputs = model(inputs)
        
        # 调整输出形状以匹配目标形状
        outputs = outputs.view(batch_size, 3, 15, h//32, w//32)
        outputs = outputs.permute(0, 1, 3, 4, 2)
        
        loss = criterion(outputs, target)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
    
    avg_loss = running_loss / len(train_loader)
    log_message(f"Epoch [{epoch+1:02d}/{epochs}] | Loss: {avg_loss:.4f}")
    
    if avg_loss < best_loss:
        best_loss = avg_loss
        torch.save(model.state_dict(), os.path.join(model_dir, "simple_yolo_best.pth"))
        log_message(f"New best model saved with loss: {best_loss:.4f}")

log_message(f"Object detection model training completed. Best loss: {best_loss:.4f}")

# --- 6. 量化目标检测模型 ---
log_message("\n=== Quantizing Object Detection Model ===")

# 加载最佳模型
model_best = SimpleYOLO(num_classes=10)
model_best.load_state_dict(torch.load(os.path.join(model_dir, "simple_yolo_best.pth"), map_location='cpu', weights_only=True))
model_best.to(device)
model_best.eval()

# --- 6.1 INT8量化 --- 
log_message("\n=== INT8 Quantization ===")
log_message("Preparing for INT8 quantization...")

# 创建模型副本
model_int8 = SimpleYOLO(num_classes=10)
model_int8.load_state_dict(torch.load(os.path.join(model_dir, "simple_yolo_best.pth"), map_location='cpu', weights_only=True))
model_int8.to('cpu')  # INT8量化通常在CPU上进行
model_int8.eval()

# 融合模块以提高量化效果
fuse_modules(model_int8, [
    ['backbone.0', 'backbone.1', 'backbone.2'],
    ['backbone.3', 'backbone.4', 'backbone.5'],
    ['backbone.6', 'backbone.7', 'backbone.8'],
    ['backbone.9', 'backbone.10', 'backbone.11'],
    ['backbone.12', 'backbone.13', 'backbone.14'],
    ['neck.0', 'neck.1', 'neck.2'],
    ['neck.3', 'neck.4', 'neck.5'],
], inplace=True)

# 设置INT8量化配置
qconfig_mapping = get_default_qconfig_mapping('qnnpack')  # ARM平台使用qnnpack

# 准备量化
model_int8_prepared = prepare_qat(model_int8, qconfig_mapping)

# 微调量化模型（快速微调1个epoch）
log_message("Fine-tuning INT8 quantized model...")
model_int8_prepared.train()
for inputs, labels in tqdm(train_loader, desc="INT8 Fine-tuning", leave=False):
    inputs = inputs.to('cpu')
    optimizer.zero_grad()
    outputs = model_int8_prepared(inputs)
    # 简化的损失计算
    loss = torch.mean(outputs)
    loss.backward()
    optimizer.step()

# 转换为INT8模型
model_int8 = convert(model_int8_prepared, inplace=False)
model_int8.eval()

# --- 6.2 INT4量化 --- 
log_message("\n=== INT4 Quantization ===")
log_message("Applying INT4 Weight-Only Quantization...")

# 创建模型副本
model_int4 = SimpleYOLO(num_classes=10)
model_int4.load_state_dict(torch.load(os.path.join(model_dir, "simple_yolo_best.pth"), map_location='cpu', weights_only=True))
model_int4.to(device)
model_int4.eval()

# 使用TorchAO进行INT4权重量化
quantize_config = Int4WeightOnlyConfig(group_size=32)
quantize_(model_int4, quantize_config)

# --- 7. 性能测试 ---
log_message("\n=== Comprehensive Performance Testing ===")

# 定义测试函数
def test_model_performance(model_name, model, input_size, device, test_loader=None):
    log_message(f"\nTesting {model_name} performance...")
    
    # 创建测试输入
    dummy_input = torch.randn(1, 3, input_size, input_size).to(device)
    
    # 内存监控函数
    def get_memory_usage():
        process = psutil.Process()
        return process.memory_info().rss / 1024 / 1024  # MB
    
    # 1. 内存占用测试
    initial_memory = get_memory_usage()
    
    # 预热
    with torch.no_grad():
        for _ in range(50):
            _ = model(dummy_input)
    
    # 2. 单张图片推理速度测试（帧率）
    with torch.no_grad():
        start_time = time.time()
        for _ in range(100):
            _ = model(dummy_input)
        inference_time = time.time() - start_time
    
    fps = 100 / inference_time
    
    # 3. 吞吐量测试（批量处理）
    batch_size = 4
    batch_input = torch.randn(batch_size, 3, input_size, input_size).to(device)
    
    with torch.no_grad():
        # 预热
        for _ in range(20):
            _ = model(batch_input)
        
        start_time = time.time()
        for _ in range(50):
            _ = model(batch_input)
        batch_inference_time = time.time() - start_time
    
    throughput = (50 * batch_size) / batch_inference_time
    
    # 4. 内存占用
    final_memory = get_memory_usage()
    memory_usage = final_memory - initial_memory
    
    # 5. 准确率测试（简化版）
    accuracy = "N/A"
    if test_loader is not None:
        correct = 0
        total = 0
        with torch.no_grad():
            for i, (inputs, labels) in enumerate(test_loader):
                if i >= 10:  # 只测试前10个batch以节省时间
                    break
                inputs = inputs.to(device)
                labels = labels.to(device)
                outputs = model(inputs)
                # 简化的准确率计算（仅作为演示）
                _, pred = outputs.max(1)
                correct += (pred == labels).sum().item()
                total += labels.size(0)
        accuracy = 100. * correct / total if total > 0 else "N/A"
    
    return {
        "fps": fps,
        "throughput": throughput,
        "memory_usage": memory_usage,
        "accuracy": accuracy
    }

# 测试所有模型
input_size = 640

# FP32模型测试
fp32_results = test_model_performance("FP32 Model", model_best, input_size, device, test_loader)

# INT8模型测试
int8_results = test_model_performance("INT8 Model", model_int8, input_size, 'cpu', test_loader)

# INT4模型测试
int4_results = test_model_performance("INT4 Weight-Only Model", model_int4, input_size, device, test_loader)

# --- 8. 结果总结 ---
log_message("\n" + "="*100)
log_message("=== Object Detection Model Quantization Results ===")
log_message("="*100)
log_message(f"{'Model Type':<30} {'FPS':<10} {'Throughput':<15} {'Memory (MB)':<15} {'Accuracy (%)':<15}")
log_message("-"*100)
log_message(f"{'FP32 Model':<30} {fp32_results['fps']:<10.2f} {fp32_results['throughput']:<15.2f} {fp32_results['memory_usage']:<15.2f} {fp32_results['accuracy']:<15}")
log_message(f"{'INT8 Model':<30} {int8_results['fps']:<10.2f} {int8_results['throughput']:<15.2f} {int8_results['memory_usage']:<15.2f} {int8_results['accuracy']:<15}")
log_message(f"{'INT4 Weight-Only Model':<30} {int4_results['fps']:<10.2f} {int4_results['throughput']:<15.2f} {int4_results['memory_usage']:<15.2f} {int4_results['accuracy']:<15}")

# 模型大小估算
fp32_size = sum(p.numel() for p in model_best.parameters()) * 4 / (1024 * 1024)  # MB
int8_size = sum(p.numel() for p in model_int8.parameters()) * 1 / (1024 * 1024)  # MB
int4_size = sum(p.numel() for p in model_int4.parameters()) * 0.5 / (1024 * 1024)  # MB

log_message("\n=== Model Size Comparison ===")
log_message(f"{'Model Type':<30} {'Size (MB)':<15} {'Compression Ratio':<20}")
log_message("-"*65)
log_message(f"{'FP32 Model':<30} {fp32_size:<15.2f} {'1.00x':<20}")
log_message(f"{'INT8 Model':<30} {int8_size:<15.2f} {fp32_size/int8_size:<20.2f}x")
log_message(f"{'INT4 Weight-Only Model':<30} {int4_size:<15.2f} {fp32_size/int4_size:<20.2f}x")

# 保存量化模型
torch.save(model_int8.state_dict(), os.path.join(model_dir, "simple_yolo_int8.pth"))
torch.save(model_int4.state_dict(), os.path.join(model_dir, "simple_yolo_int4.pth"))

log_message("\n=== Object Detection Module with Quantization Completed ===")
log_message("Quantized models saved to:")
log_message(f"- INT8 Model: {os.path.join(model_dir, 'simple_yolo_int8.pth')}")
log_message(f"- INT4 Model: {os.path.join(model_dir, 'simple_yolo_int4.pth')}")
log_message("\nTesting metrics collected:")
log_message("- Frame Rate (FPS): Inference speed for single image")
log_message("- Throughput: Number of images processed per second (batch size=4)")
log_message("- Memory Usage: RAM consumption during inference")
log_message("- Accuracy: Simplified classification accuracy (top-1)")
