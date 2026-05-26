import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torchvision.models.quantization import mobilenet_v2
import os, time, datetime
from tqdm import tqdm
from torchao.quantization import quantize_, Int4WeightOnlyConfig

# 修复Windows环境下的多进程问题
if __name__ == '__main__':
    # --- 1. 全局配置 ---
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    batch_size = 128
    epochs = 10  # 减少训练轮数以加快进度
    lr = 1e-4
    model_dir = "models"
    log_dir = "logs"
    os.makedirs(model_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)

    log_filename = os.path.join(log_dir, f"int4_torchao_mobilenetv2_c10_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.log")
    def log_message(msg):
        t = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        full_msg = f"[{t}] {msg}"
        print(full_msg)
        with open(log_filename, "a", encoding="utf-8") as f: 
            f.write(full_msg + "\n")

    # --- 2. 数据处理  ---
    # 保持标准224x224输入尺寸，不通过缩小图片提高性能
    # Windows下设置num_workers=0以避免多进程问题
    transform = transforms.Compose([
        transforms.Resize(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    train_loader = torch.utils.data.DataLoader(
        datasets.CIFAR10('./data', train=True, download=True, transform=transform), 
        batch_size=batch_size, shuffle=True, num_workers=0
    )
    test_loader = torch.utils.data.DataLoader(
        datasets.CIFAR10('./data', train=False, download=True, transform=transform), 
        batch_size=batch_size, shuffle=False, num_workers=0
    )

    # --- 3. 模型准备 ---
    log_message("Loading MobileNetV2 for INT4 quantization with TorchAO...")

    # 加载模型
    model = mobilenet_v2(weights=None, quantize=False)
    model.classifier[1] = nn.Linear(model.last_channel, 10)

    # 加载或训练FP32模型
    fp32_path = os.path.join(model_dir, "fp32_mobilenetv2_c10_best.pth")
    if not os.path.exists(fp32_path):
        log_message("FP32 model not found, training a simple version...")
        model.to(device)
        
        optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9, weight_decay=5e-4)
        criterion = nn.CrossEntropyLoss()
        
        for epoch in range(5):  # 快速训练5个epoch
            model.train()
            for inputs, labels in tqdm(train_loader, desc=f"Training FP32 Epoch [{epoch+1}/5]", leave=False):
                inputs, labels = inputs.to(device), labels.to(device)
                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
        
        torch.save(model.state_dict(), fp32_path)
        log_message(f"Simple FP32 model saved to {fp32_path}")

    model.load_state_dict(torch.load(fp32_path, map_location='cpu', weights_only=True))
    model.to(device)
    model.eval()

    # --- 4. 测试FP32模型性能 ---
    log_message("\nTesting FP32 model performance...")

    # 测试精度
    correct = 0
    with torch.no_grad():
        for inputs, labels in tqdm(test_loader, desc="Testing FP32", leave=False):
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, pred = torch.max(outputs, 1)
            correct += (pred == labels).sum().item()

    fp32_acc = 100. * correct / len(test_loader.dataset)

    # 测试推理速度
    dummy_input = torch.randn(1, 3, 224, 224).to(device)
    with torch.no_grad():
        # 预热
        for _ in range(50):
            _ = model(dummy_input)
        
        # 测试
        start_time = time.time()
        for _ in range(100):
            _ = model(dummy_input)
        fp32_time = time.time() - start_time
        fp32_fps = 100 / fp32_time

    log_message(f"FP32 Model - Accuracy: {fp32_acc:.2f}%, FPS: {fp32_fps:.2f}")

    # --- 5. INT4 权重量化 (Weight-Only Quantization)---
    log_message("\n=== INT4 Weight-Only Quantization ===")

    # 创建模型副本
    model_int4_weight_only = mobilenet_v2(weights=None, quantize=False)
    model_int4_weight_only.classifier[1] = nn.Linear(model_int4_weight_only.last_channel, 10)
    model_int4_weight_only.load_state_dict(torch.load(fp32_path, map_location='cpu', weights_only=True))
    model_int4_weight_only.to(device)
    model_int4_weight_only.eval()

    # 使用TorchAO进行INT4权重量化
    quantize_config = Int4WeightOnlyConfig(group_size=32)
    quantize_(model_int4_weight_only, quantize_config)

    # 测试量化后的模型
    correct = 0
    with torch.no_grad():
        for inputs, labels in tqdm(test_loader, desc="Testing INT4 Weight-Only", leave=False):
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model_int4_weight_only(inputs)
            _, pred = torch.max(outputs, 1)
            correct += (pred == labels).sum().item()

    int4_weight_only_acc = 100. * correct / len(test_loader.dataset)

    # 测试推理速度
    with torch.no_grad():
        # 预热
        for _ in range(50):
            _ = model_int4_weight_only(dummy_input)
        
        # 测试
        start_time = time.time()
        for _ in range(100):
            _ = model_int4_weight_only(dummy_input)
        int4_weight_only_time = time.time() - start_time
        int4_weight_only_fps = 100 / int4_weight_only_time

    log_message(f"INT4 Weight-Only - Accuracy: {int4_weight_only_acc:.2f}%, FPS: {int4_weight_only_fps:.2f}, Speedup: {fp32_time/int4_weight_only_time:.2f}x")

    # 保存模型
    torch.save(model_int4_weight_only.state_dict(), os.path.join(model_dir, "mobilenetv2_int4_weight_only_c10.pth"))
    log_message(f"INT4 Weight-Only model saved")

    # --- 6. 总结 ---
    fp32_size = os.path.getsize(fp32_path) / (1024 * 1024)
    int4_size = fp32_size * 0.25  # 理论值

    log_message("=" * 80)
    log_message(f" MobileNetV2 INT4 Quantization Final Report (CIFAR-10) ")
    log_message("=" * 80)
    log_message(f" FP32 Accuracy     : {fp32_acc:.2f}%")
    log_message(f" FP32 FPS          : {fp32_fps:.2f}")
    log_message(f" FP32 Size         : {fp32_size:.2f} MB")
    log_message("-" * 80)
    log_message(f" INT4 Weight-Only Accuracy: {int4_weight_only_acc:.2f}%")
    log_message(f" INT4 Weight-Only FPS     : {int4_weight_only_fps:.2f}")
    log_message(f" INT4 Weight-Only Size    : {int4_size:.2f} MB (Compression: {fp32_size/int4_size:.2f}x)")
    log_message(f" INT4 Weight-Only Speedup : {fp32_time/int4_weight_only_time:.2f}x")
    log_message("=" * 80)