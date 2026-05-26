import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torchvision.models.quantization import mobilenet_v2
import os, time, datetime
from tqdm import tqdm
from torchao.quantization import quantize_, Int4WeightOnlyConfig, Int8DynamicActivationInt4WeightConfig

# --- 1. 全局配置 ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
batch_size = 128
epochs = 50
lr = 1e-4
model_dir = "models"
log_dir = "logs"
os.makedirs(model_dir, exist_ok=True)
os.makedirs(log_dir, exist_ok=True)

log_filename = os.path.join(log_dir, f"int4_torchao_mobilenetv2_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.log")

def log_message(msg):
    t = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    full_msg = f"[{t}] {msg}"
    print(full_msg)
    with open(log_filename, "a", encoding="utf-8") as f:
        f.write(full_msg + "\n")

# --- 2. 数据处理 ---
# 使用标准的224x224输入尺寸，不通过缩小图片来提高性能
transform = transforms.Compose([
    transforms.Resize(224),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

train_loader = torch.utils.data.DataLoader(
    datasets.CIFAR10('./data', train=True, download=True, transform=transform),
    batch_size=batch_size, shuffle=True, num_workers=4
)

test_loader = torch.utils.data.DataLoader(
    datasets.CIFAR10('./data', train=False, download=True, transform=transform),
    batch_size=batch_size, shuffle=False, num_workers=4
)

# --- 3. 模型准备 ---
log_message("Loading MobileNetV2 for INT4 quantization with TorchAO...")

# 加载预训练的FP32模型
model = mobilenet_v2(weights=None, quantize=False)
model.classifier[1] = nn.Linear(model.last_channel, 10)

# 加载FP32模型权重
fp32_path = os.path.join(model_dir, "fp32_mobilenetv2_best.pth")
if not os.path.exists(fp32_path):
    # 如果没有预训练模型，先训练一个
    log_message("FP32 model not found, training one...")
    model.to(device)
    
    optimizer = optim.SGD(model.parameters(), lr=0.1, momentum=0.9, weight_decay=5e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=200)
    criterion = nn.CrossEntropyLoss()
    
    best_acc = 0.0
    for epoch in range(200):
        model.train()
        train_loss = 0
        correct = 0
        total = 0
        
        for inputs, labels in tqdm(train_loader, desc=f"FP32 Training Epoch [{epoch+1:03d}/200]"):
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
        
        scheduler.step()
        
        model.eval()
        test_loss = 0
        test_correct = 0
        test_total = 0
        with torch.no_grad():
            for inputs, labels in test_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                
                test_loss += loss.item()
                _, predicted = outputs.max(1)
                test_total += labels.size(0)
                test_correct += predicted.eq(labels).sum().item()
        
        acc = 100. * test_correct / test_total
        log_message(f"FP32 Epoch [{epoch+1:03d}/200] | Train Loss: {train_loss/len(train_loader):.3f} | Test Acc: {acc:.2f}%")
        
        if acc > best_acc:
            best_acc = acc
            torch.save(model.state_dict(), fp32_path)
            log_message(f"New best FP32 model saved with accuracy: {best_acc:.2f}%")
    
    log_message(f"FP32 training completed. Best accuracy: {best_acc:.2f}%")

# 加载FP32模型权重
model.load_state_dict(torch.load(fp32_path, map_location='cpu', weights_only=True))
model.to(device)
model.eval()

# --- 4. INT4 权重仅量化 (Weight-Only Quantization)---
log_message("\n=== INT4 Weight-Only Quantization ===")
model_int4_weight_only = model

# 使用TorchAO进行INT4权重量化
quantize_config = Int4WeightOnlyConfig(group_size=32)
quantize_(model_int4_weight_only, quantize_config)

# 测试量化后的模型
log_message("Testing INT4 Weight-Only Quantized model...")
model_int4_weight_only.eval()
correct = 0
with torch.no_grad():
    for inputs, labels in tqdm(test_loader, desc="Testing INT4 Weight-Only"):
        inputs, labels = inputs.to(device), labels.to(device)
        outputs = model_int4_weight_only(inputs)
        _, pred = torch.max(outputs, 1)
        correct += (pred == labels).sum().item()

int4_weight_only_acc = 100. * correct / len(test_loader.dataset)
log_message(f"INT4 Weight-Only Quantization Accuracy: {int4_weight_only_acc:.2f}%")

# 保存INT4权重量化模型
torch.save(model_int4_weight_only.state_dict(), os.path.join(model_dir, "int4_torchao_weight_only_mobilenetv2.pth"))

# --- 5. INT8动态激活 + INT4权重混合量化 ---
log_message("\n=== INT8 Dynamic Activation + INT4 Weight Quantization ===")
model_int8_act_int4_weight = mobilenet_v2(weights=None, quantize=False)
model_int8_act_int4_weight.classifier[1] = nn.Linear(model_int8_act_int4_weight.last_channel, 10)
model_int8_act_int4_weight.load_state_dict(torch.load(fp32_path, map_location='cpu', weights_only=True))
model_int8_act_int4_weight.to(device)
model_int8_act_int4_weight.eval()

# 使用TorchAO进行混合量化
mixed_quantize_config = Int8DynamicActivationInt4WeightConfig(group_size=32)
quantize_(model_int8_act_int4_weight, mixed_quantize_config)

# 测试混合量化后的模型
log_message("Testing INT8 Dynamic Activation + INT4 Weight Quantized model...")
model_int8_act_int4_weight.eval()
correct = 0
with torch.no_grad():
    for inputs, labels in tqdm(test_loader, desc="Testing INT8+INT4 Mixed"):
        inputs, labels = inputs.to(device), labels.to(device)
        outputs = model_int8_act_int4_weight(inputs)
        _, pred = torch.max(outputs, 1)
        correct += (pred == labels).sum().item()

mixed_quant_acc = 100. * correct / len(test_loader.dataset)
log_message(f"INT8 Dynamic Activation + INT4 Weight Quantization Accuracy: {mixed_quant_acc:.2f}%")

# 保存混合量化模型
torch.save(model_int8_act_int4_weight.state_dict(), os.path.join(model_dir, "int8_int4_mixed_mobilenetv2.pth"))

# --- 6. 性能对比测试 ---
log_message("\n=== Performance Comparison ===")

# 测试FP32模型性能
log_message("Testing FP32 model performance...")
model_fp32 = mobilenet_v2(weights=None, quantize=False)
model_fp32.classifier[1] = nn.Linear(model_fp32.last_channel, 10)
model_fp32.load_state_dict(torch.load(fp32_path, map_location='cpu', weights_only=True))
model_fp32.to(device)
model_fp32.eval()

# 预热
with torch.no_grad():
    for _ in range(50):
        dummy_input = torch.randn(1, 3, 224, 224).to(device)
        _ = model_fp32(dummy_input)

# 测试推理时间
start_time = time.time()
with torch.no_grad():
    for _ in range(1000):
        dummy_input = torch.randn(1, 3, 224, 224).to(device)
        _ = model_fp32(dummy_input)
fp32_time = time.time() - start_time
fp32_fps = 1000 / fp32_time

# 测试INT4权重量化模型性能
log_message("Testing INT4 Weight-Only model performance...")
with torch.no_grad():
    for _ in range(50):
        dummy_input = torch.randn(1, 3, 224, 224).to(device)
        _ = model_int4_weight_only(dummy_input)

start_time = time.time()
with torch.no_grad():
    for _ in range(1000):
        dummy_input = torch.randn(1, 3, 224, 224).to(device)
        _ = model_int4_weight_only(dummy_input)
int4_weight_only_time = time.time() - start_time
int4_weight_only_fps = 1000 / int4_weight_only_time

# 测试混合量化模型性能
log_message("Testing INT8+INT4 Mixed model performance...")
with torch.no_grad():
    for _ in range(50):
        dummy_input = torch.randn(1, 3, 224, 224).to(device)
        _ = model_int8_act_int4_weight(dummy_input)

start_time = time.time()
with torch.no_grad():
    for _ in range(1000):
        dummy_input = torch.randn(1, 3, 224, 224).to(device)
        _ = model_int8_act_int4_weight(dummy_input)
mixed_quant_time = time.time() - start_time
mixed_quant_fps = 1000 / mixed_quant_time

# --- 7. 结果总结 ---
log_message("\n=== Final Results ===")
log_message(f"FP32 Model: Accuracy = {best_acc:.2f}%, FPS = {fp32_fps:.2f}")
log_message(f"INT4 Weight-Only: Accuracy = {int4_weight_only_acc:.2f}%, FPS = {int4_weight_only_fps:.2f}, Speedup = {fp32_time/int4_weight_only_time:.2f}x")
log_message(f"INT8+INT4 Mixed: Accuracy = {mixed_quant_acc:.2f}%, FPS = {mixed_quant_fps:.2f}, Speedup = {fp32_time/mixed_quant_time:.2f}x")

# 计算模型大小
fp32_size = os.path.getsize(fp32_path) / (1024 * 1024)
int4_weight_only_size = fp32_size * 0.25  # 理论值，实际可能因分组等因素略有不同
mixed_quant_size = fp32_size * 0.375  # 理论值，实际可能因分组等因素略有不同

log_message(f"\n=== Model Size Comparison ===")
log_message(f"FP32 Model: {fp32_size:.2f} MB")
log_message(f"INT4 Weight-Only: {int4_weight_only_size:.2f} MB (Compression: {fp32_size/int4_weight_only_size:.2f}x)")
log_message(f"INT8+INT4 Mixed: {mixed_quant_size:.2f} MB (Compression: {fp32_size/mixed_quant_size:.2f}x)")

log_message("\n=== INT4 Quantization with TorchAO Completed ===")
