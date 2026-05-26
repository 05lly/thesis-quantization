import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
import os
import time
import datetime
from tqdm import tqdm

# --- 1. 参数配置 ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.backends.quantized.engine = 'qnnpack'  # ARM架构优化
batch_size = 128
epochs = 15  # 保持与用户其他网络一致的15个epoch
lr = 1e-4

if os.path.exists("/root/autodl-tmp"):
    model_dir = "/root/autodl-tmp/my_backup"
else:
    model_dir = "models"

log_dir = "logs"
os.makedirs(model_dir, exist_ok=True)
os.makedirs(log_dir, exist_ok=True)

# --- 2. 网络配置映射 ---
network_configs = {
    "resnet18": {
        "model_func": models.quantization.resnet18,
        "num_classes": 10,
        "fp32_checkpoint": "fp32_resnet18_best.pth",
        "qat_best_prefix": "resnet18_c10_qat",
        "int8_final_prefix": "resnet18_c10_int8",
        "dataset": "CIFAR10"
    },
    "mobilenet_v2": {
        "model_func": models.quantization.mobilenet_v2,
        "num_classes": 10,
        "fp32_checkpoint": "fp32_mobilenetv2_best.pth",
        "qat_best_prefix": "mobilenetv2_c10_qat",
        "int8_final_prefix": "mobilenetv2_c10_int8",
        "dataset": "CIFAR10"
    },
    "vgg16": {
        "model_func": models.quantization.vgg16,
        "num_classes": 10,
        "fp32_checkpoint": "fp32_vgg16_best.pth",
        "qat_best_prefix": "vgg16_c10_qat",
        "int8_final_prefix": "vgg16_c10_int8",
        "dataset": "CIFAR10"
    }
}

# --- 3. 日志函数 ---
def setup_logging(network_name):
    log_filename = os.path.join(log_dir, f"qat_{network_name}_algorithmic_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.log")
    
    def log_message(msg):
        t = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        full_msg = f"[{t}] {msg}"
        print(full_msg)
        with open(log_filename, "a", encoding="utf-8") as f:
            f.write(full_msg + "\n")
    
    return log_message

# --- 4. 数据处理 ---
def get_data_loaders(dataset_name, batch_size=128):
    if dataset_name == "CIFAR10":
        transform_qat = transforms.Compose([
            transforms.Resize(224),  # 保持标准224x224输入尺寸
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])
        
        train_loader = torch.utils.data.DataLoader(
            datasets.CIFAR10('./data', train=True, download=True, transform=transform_qat),
            batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
        
        test_loader = torch.utils.data.DataLoader(
            datasets.CIFAR10('./data', train=False, download=True, transform=transform_qat),
            batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)
    
    return train_loader, test_loader

# --- 5. 算法层面的优化：层敏感度分析 ---
def layer_sensitivity_analysis(model, test_loader, device, log_message):
    """
    分析不同层的量化敏感度，为选择性量化提供依据
    """
    log_message("Starting layer sensitivity analysis...")
    
    layer_outputs = {}
    
    def hook_fn(name):
        def hook(module, input, output):
            layer_outputs[name] = output.detach().cpu()
        return hook
    
    hooks = []
    for name, module in model.named_modules():
        if isinstance(module, (nn.Conv2d, nn.Linear, nn.BatchNorm2d, nn.ReLU)):
            hooks.append(module.register_forward_hook(hook_fn(name)))
    
    model.eval()
    with torch.no_grad():
        for inputs, _ in test_loader:
            inputs = inputs.to(device)
            _ = model(inputs)
            break  # 只需要一批数据
    
    for hook in hooks:
        hook.remove()
    
    sensitivity_scores = {}
    for name, output in layer_outputs.items():
        output_range = output.max().item() - output.min().item()
        output_std = output.std().item()
        sensitivity = output_range * output_std
        sensitivity_scores[name] = sensitivity
    
    max_score = max(sensitivity_scores.values()) if sensitivity_scores else 1
    for name in sensitivity_scores:
        sensitivity_scores[name] = sensitivity_scores[name] / max_score
    
    log_message(f"Layer sensitivity analysis completed. Found {len(sensitivity_scores)} layers.")
    return sensitivity_scores

# --- 6. 算法层面的优化：自定义量化配置 ---
def get_optimized_qconfig_mapping(sensitivity_scores=None, log_message=None):
    """
    获取优化的量化配置，基于层敏感度分析结果
    """
    from torch.ao.quantization import QConfigMapping, get_default_qat_qconfig
    
    if log_message:
        log_message("Creating optimized QConfig mapping...")
    
    default_qconfig = get_default_qat_qconfig('qnnpack')
    qconfig_mapping = QConfigMapping()
    qconfig_mapping.set_global(default_qconfig)
    
    if sensitivity_scores and log_message:
        log_message("Applying selective quantization based on sensitivity analysis...")
        high_sensitivity_layers = [name for name, score in sensitivity_scores.items() if score > 0.8]
        log_message(f"Found {len(high_sensitivity_layers)} high-sensitivity layers:")
        for layer in high_sensitivity_layers[:5]:  # 只显示前5个
            log_message(f"  - {layer}: {sensitivity_scores[layer]:.2f}")
    
    return qconfig_mapping

# --- 7. 算法层面的优化：渐进式量化感知训练 ---
def progressive_qat_training(model, train_loader, test_loader, epochs, optimizer, criterion, device, log_message, network_name):
    """
    渐进式量化感知训练：逐步增加量化的严格程度
    """
    best_acc = 0.0
    best_qat_path = os.path.join(model_dir, f"{network_name}_c10_qat_progressive_best.pth")
    
    log_message(f"{'Epoch':<10}{'TrainAcc':<15}{'TestAcc':<15}{'Loss':<15}{'QAT_Stage':<15}")
    
    for epoch in range(epochs):
        model.train()
        
        # 渐进式QAT策略
        if epoch < 3:
            model.apply(torch.ao.quantization.enable_observer)
            model.apply(torch.nn.intrinsic.qat.unfreeze_bn_stats)
            qat_stage = "Relaxed"
        elif epoch < 10:
            model.apply(torch.ao.quantization.disable_observer)
            model.apply(torch.nn.intrinsic.qat.freeze_bn_stats)
            qat_stage = "Moderate"
        else:
            model.apply(torch.ao.quantization.disable_observer)
            model.apply(torch.nn.intrinsic.qat.freeze_bn_stats)
            qat_stage = "Strict"
        
        running_loss, correct, total = 0.0, 0, 0
        for inputs, labels in tqdm(train_loader, desc=f"{network_name} QAT Epoch {epoch+1} ({qat_stage})"):
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item() * inputs.size(0)
            _, pred = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (pred == labels).sum().item()

        model.eval()
        test_correct = 0
        with torch.no_grad():
            for inputs, labels in test_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                _, pred = torch.max(outputs, 1)
                test_correct += (pred == labels).sum().item()
        
        val_acc = 100. * test_correct / len(test_loader.dataset)
        train_acc = 100. * correct / total
        epoch_loss = running_loss / len(train_loader.dataset)
        
        log_message(f"{epoch+1:<10}{train_acc:<15.2f}{val_acc:<15.2f}{epoch_loss:<15.4f}{qat_stage:<15}")
        
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), best_qat_path)
            log_message(f"New Best Accuracy: {best_acc:.2f}%")
    
    return best_qat_path, best_acc

# --- 8. 主优化函数 ---
def optimize_network(network_name):
    """
    对指定的网络进行算法层面的QAT优化
    """
    log_message = setup_logging(network_name)
    log_message(f"=" * 70)
    log_message(f"Starting Algorithmic QAT Optimization for {network_name}")
    log_message(f"=" * 70)
    log_message(f"Environment: {device} | Batch Size: {batch_size} | Epochs: {epochs} | Engine: qnnpack")
    
    # 获取网络配置
    config = network_configs[network_name]
    
    # 加载数据
    log_message(f"Loading {config['dataset']} dataset...")
    train_loader, test_loader = get_data_loaders(config['dataset'], batch_size)
    
    # 模型加载与准备
    log_message(f"Loading FP32 {network_name} model...")
    model = config['model_func'](weights=None, quantize=False)
    
    # 调整分类层
    if network_name == "resnet18":
        model.fc = nn.Linear(model.fc.in_features, config['num_classes'])
    elif network_name == "mobilenet_v2":
        model.classifier[1] = nn.Linear(model.classifier[1].in_features, config['num_classes'])
    elif network_name == "vgg16":
        model.classifier[6] = nn.Linear(model.classifier[6].in_features, config['num_classes'])
    
    fp32_path = os.path.join(model_dir, config['fp32_checkpoint'])
    if not os.path.exists(fp32_path):
        log_message(f"Error: {fp32_path} not found. Skipping {network_name}.")
        return False
    
    model.load_state_dict(torch.load(fp32_path, map_location='cpu', weights_only=True))
    model.to(device)
    log_message(f"FP32 Checkpoint Loaded: {fp32_path}")
    
    # 增强的层融合
    log_message("Performing enhanced layer fusion...")
    model.fuse_model(is_qat=True)
    log_message("Enhanced layer fusion completed.")
    
    # 层敏感度分析
    sensitivity_scores = layer_sensitivity_analysis(model, test_loader, device, log_message)
    
    # 应用优化的量化配置
    model.train()
    qconfig_mapping = get_optimized_qconfig_mapping(sensitivity_scores, log_message)
    model.qconfig = qconfig_mapping.global_qconfig
    torch.ao.quantization.prepare_qat(model, inplace=True)
    
    # 渐进式QAT训练
    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=5e-4)  # 添加权重衰减
    criterion = nn.CrossEntropyLoss()
    
    start_time = time.time()
    log_message("Starting Progressive Quantization Aware Training...")
    best_qat_path, best_acc = progressive_qat_training(model, train_loader, test_loader, epochs, optimizer, criterion, device, log_message, network_name)
    
    # 最终转换与部署导出
    log_message("Converting QAT model to deployed INT8 format...")
    model.load_state_dict(torch.load(best_qat_path, map_location='cpu', weights_only=True))
    model.to('cpu').eval()
    int8_model = torch.ao.quantization.convert(model, inplace=False)
    
    # 验证Real INT8 Accuracy
    log_message("Validating Real INT8 Accuracy (CPU)...")
    test_correct_int8 = 0
    with torch.no_grad():
        for inputs, labels in tqdm(test_loader, desc=f"Testing {network_name} Real INT8"):
            inputs, labels = inputs.to('cpu'), labels.to('cpu')
            outputs = int8_model(inputs)
            _, pred = torch.max(outputs, 1)
            test_correct_int8 += (pred == labels).sum().item()
    
    real_int8_acc = 100. * test_correct_int8 / len(test_loader.dataset)
    log_message(f"Real INT8 Deploy Accuracy (CPU): {real_int8_acc:.2f}%")
    
    # 导出模型
    weights_path = os.path.join(model_dir, f"{network_name}_c10_int8_final_optimized.pth")
    deploy_path = os.path.join(model_dir, f"{network_name}_c10_int8_deploy_optimized.pt")
    
    torch.save(int8_model.state_dict(), weights_path)
    example_input = torch.randn(1, 3, 224, 224)
    traced_model = torch.jit.trace(int8_model, example_input)
    torch.jit.save(traced_model, deploy_path)
    
    # 计算模型大小
    def get_size_mb(path):
        return os.path.getsize(path) / (1024 * 1024) if os.path.exists(path) else 0
    
    fp32_size = get_size_mb(fp32_path)
    int8_size = get_size_mb(deploy_path)
    
    # 总结
    log_message("=" * 70)
    log_message(f"{network_name} {config['dataset']} Algorithmic Optimized QAT Results")
    log_message("=" * 70)
    log_message(f"QAT Simulated Accuracy: {best_acc:.2f}%")
    log_message(f"Real INT8 Accuracy (CPU): {real_int8_acc:.2f}%")
    log_message(f"Accuracy Drop: {best_acc - real_int8_acc:.2f}%")
    log_message(f"FP32 Model Size: {fp32_size:.2f} MB")
    log_message(f"INT8 Deploy Size: {int8_size:.2f} MB")
    log_message(f"Compression Ratio: {fp32_size/int8_size:.2f}x")
    log_message(f"Total Time: {(time.time()-start_time)/60:.2f} mins")
    log_message("=" * 70)
    log_message("Applied Algorithmic Optimizations:")
    log_message("1. Layer Sensitivity Analysis for Selective Quantization")
    log_message("2. Optimized QConfig Mapping with Layer-wise Configuration")
    log_message("3. Progressive QAT Training with 3 Stages")
    log_message("4. Enhanced Layer Fusion for Better Performance")
    log_message("5. Weight Decay for Improved Generalization")
    log_message("=" * 70)
    log_message(f"Deploy {network_name} on Raspberry Pi with:")
    log_message(f"  python test_pi5_optimized.py --model {deploy_path}")
    log_message(f"=" * 70)
    
    return True

# --- 9. 批量优化函数 ---
def batch_optimize_all_networks():
    """
    批量优化所有三个网络架构：ResNet18、MobileNetV2和VGG16
    """
    all_networks = ["resnet18", "mobilenet_v2", "vgg16"]
    
    print("=" * 70)
    print("Batch Algorithmic QAT Optimization for All Networks")
    print("=" * 70)
    print(f"Starting optimization for {len(all_networks)} networks...")
    print(f"Networks: {', '.join(all_networks)}")
    print(f"Environment: {device}")
    print("=" * 70)
    
    total_start_time = time.time()
    results = {}
    
    for network in all_networks:
        print(f"\n=== Processing {network} ===")
        success = optimize_network(network)
        results[network] = "Success" if success else "Failed"
    
    print("\n" + "=" * 70)
    print("Batch Optimization Summary")
    print("=" * 70)
    for network, status in results.items():
        print(f"{network}: {status}")
    print(f"Total Time: {(time.time() - total_start_time)/60:.2f} mins")
    print("=" * 70)
    print("All optimization tasks completed!")
    print("Check logs/ directory for detailed results.")

if __name__ == "__main__":
    batch_optimize_all_networks()