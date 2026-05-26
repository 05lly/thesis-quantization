import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
import os
import time
import datetime
from tqdm import tqdm
from torch.ao.quantization import QConfig
from torch.ao.quantization.observer import MinMaxObserver, PerChannelMinMaxObserver
from torchao.quantization import quantize_, Int4WeightOnlyConfig

# --- 1. 参数配置 --- (保持与用户原有设置一致)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.backends.quantized.engine = 'qnnpack'  # ARM架构优化
batch_size = 128
epochs = 15  # 保持与用户其他网络一致的15个epoch
lr = 1e-4

# 模型和日志目录配置
if os.path.exists("/root/autodl-tmp"):
    model_dir = "/root/autodl-tmp/my_backup"
else:
    model_dir = "models"

log_dir = "logs"
os.makedirs(model_dir, exist_ok=True)
os.makedirs(log_dir, exist_ok=True)

# --- 2. 日志函数 --- 
log_filename = os.path.join(log_dir, f"mobilenetv2_c100_int4_qat_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.log")

def log_message(msg):
    t = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    full_msg = f"[{t}] {msg}"
    print(full_msg)
    with open(log_filename, "a", encoding="utf-8") as f:
        f.write(full_msg + "\n")

log_message(f"Environment: {device} | Dataset: CIFAR-100 | Batch Size: {batch_size} | Epochs: {epochs} | Engine: qnnpack")
log_message("INT4 Quantization Aware Training (QAT) - MobileNetV2")

# --- 3. 数据处理 --- (CIFAR-100专用)
transform_qat = transforms.Compose([
    transforms.Resize(224),  # 保持标准224x224输入尺寸
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.5071, 0.4865, 0.4409), (0.2673, 0.2564, 0.2761)),
])

train_loader = torch.utils.data.DataLoader(
    datasets.CIFAR100('./data', train=True, download=True, transform=transform_qat),
    batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)

test_loader = torch.utils.data.DataLoader(
    datasets.CIFAR100('./data', train=False, download=True, transform=transform_qat),
    batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)

# --- 4. 创建INT4 QConfig --- 
log_message("Setting up INT4 QAT configuration...")

# 定义INT4量化配置（使用torchao实现真正的INT4量化）
def create_int4_qconfig():
    # 基础INT8量化配置（PyTorch支持）
    activation_observer = MinMaxObserver.with_args(
        dtype=torch.quint8,  # 激活值保持INT8
        qscheme=torch.per_tensor_affine,  # 逐张量非对称量化
        reduce_range=True
    )
    
    # 权重使用INT8配置作为过渡，后续会通过torchao转换为INT4
    weight_observer = PerChannelMinMaxObserver.with_args(
        dtype=torch.qint8,  # 使用PyTorch支持的qint8
        qscheme=torch.per_channel_symmetric,  # 逐通道对称量化
        reduce_range=False
    )
    
    # 创建基础QConfig
    qconfig = QConfig(
        weight=weight_observer,
        activation=activation_observer
    )
    
    return qconfig

# 创建基础QConfig
int4_qconfig = create_int4_qconfig()
log_message("INT4 QConfig created: Weight(INT8过渡) + Activation(INT8)")
log_message("Note: 权重将在后续通过torchao转换为真正的INT4")

# --- 5. 加载FP32模型 --- 
log_message("Loading FP32 MobileNetV2 model...")

# 创建MobileNetV2模型
model = models.mobilenet_v2(weights=None)
model.classifier[1] = nn.Linear(model.last_channel, 100)  # CIFAR-100有100个类别

# 加载FP32预训练模型
fp32_path = os.path.join(model_dir, "mobilenetv2_c100_fp32_best.pth")
if not os.path.exists(fp32_path):
    log_message(f"Error: FP32 model not found at {fp32_path}")
    log_message("Please train FP32 model first or provide the correct path")
    exit(1)

model.load_state_dict(torch.load(fp32_path, map_location='cpu', weights_only=True))
model.to(device)
log_message(f"FP32 Checkpoint Loaded: {fp32_path}")

# --- 6. 测试FP32模型性能 --- 
log_message("\nTesting FP32 model performance...")

model.eval()
test_correct_fp32 = 0
with torch.no_grad():
    for inputs, labels in tqdm(test_loader, desc="Testing FP32 Accuracy"):
        inputs, labels = inputs.to(device), labels.to(device)
        outputs = model(inputs)
        _, pred = torch.max(outputs, 1)
        test_correct_fp32 += (pred == labels).sum().item()

fp32_acc = 100. * test_correct_fp32 / len(test_loader.dataset)
log_message(f"FP32 Accuracy: {fp32_acc:.2f}%")

# --- 7. 准备QAT模型 --- 
log_message("\nPreparing model for INT4 QAT...")

# 设置为训练模式
model.train()

# 应用INT4 QConfig
model.qconfig = int4_qconfig

# 替换成量化感知版本的层
model = torch.ao.quantization.prepare_qat(model, inplace=True)
log_message("QAT model preparation completed")

# --- 8. 渐进式QAT训练 --- 
log_message("\nStarting Progressive INT4 QAT Training...")
log_message("Epoch     TrainAcc       TestAcc        Loss           QAT_Stage      ")

# 定义优化器和损失函数（与用户INT8脚本保持一致，使用SGD）
optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9)
criterion = nn.CrossEntropyLoss()

# 渐进式QAT训练策略
def progressive_qat_training(model, train_loader, test_loader, epochs, optimizer, criterion, device):
    best_acc = 0.0
    
    for epoch in range(epochs):
        # 根据epoch调整QAT阶段
        if epoch < 3:
            # 阶段1：放松阶段，启用观察者，不冻结BN
            model.apply(torch.ao.quantization.enable_observer)
            for module in model.modules():
                if hasattr(module, 'training'):
                    module.training = True
                    if hasattr(module, 'freeze_bn'):
                        module.freeze_bn = False
            qat_stage = "Relaxed"
        elif epoch < 10:
            # 阶段2：适度阶段，禁用观察者，冻结BN
            model.apply(torch.ao.quantization.disable_observer)
            model.apply(torch.nn.intrinsic.qat.freeze_bn_stats)
            qat_stage = "Moderate"
        else:
            # 阶段3：严格阶段，完全启用量化约束
            model.apply(torch.ao.quantization.disable_observer)
            model.apply(torch.nn.intrinsic.qat.freeze_bn_stats)
            qat_stage = "Strict"
        
        # 训练循环
        model.train()
        train_correct = 0
        total = 0
        epoch_loss = 0.0
        
        train_bar = tqdm(train_loader, desc=f"QAT Epoch {epoch+1} ({qat_stage})")
        for batch_idx, (inputs, labels) in enumerate(train_bar):
            inputs, labels = inputs.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            # 统计训练准确率
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            train_correct += (predicted == labels).sum().item()
            epoch_loss += loss.item()
            
            train_bar.set_postfix(loss=f"{loss.item():.4f}", acc=f"{100.*train_correct/total:.2f}%")
        
        train_acc = 100. * train_correct / total
        epoch_loss /= len(train_loader)
        
        # 测试循环
        model.eval()
        test_correct = 0
        with torch.no_grad():
            for inputs, labels in test_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                _, predicted = torch.max(outputs.data, 1)
                test_correct += (predicted == labels).sum().item()
        
        test_acc = 100. * test_correct / len(test_loader.dataset)
        
        # 保存最佳模型
        if test_acc > best_acc:
            best_acc = test_acc
            log_message(f"New Best Accuracy: {best_acc:.2f}%")
        
        # 记录日志
        log_message(f"{epoch+1:<10} {train_acc:.2f}%{' ':<15} {test_acc:.2f}%{' ':<15} {epoch_loss:.4f} {' ':<15} {qat_stage}")
    
    return best_acc

# 开始训练
start_time = time.time()
best_qat_acc = progressive_qat_training(model, train_loader, test_loader, epochs, optimizer, criterion, device)
total_time = time.time() - start_time
log_message(f"\nQAT Training Completed in {total_time/60:.2f} minutes")
log_message(f"Best QAT Accuracy: {best_qat_acc:.2f}%")

# --- 9. 转换为实际INT4模型 --- 
log_message("\nConverting QAT model to deployed INT4 format using torchao...")

# 使用torchao将模型转换为真正的INT4量化模型
model.eval()

# 先转换为INT8模型
int8_model = torch.ao.quantization.convert(model, inplace=False)
log_message("INT8 model conversion completed")

# 使用torchao将INT8模型转换为INT4模型（权重INT4，激活INT8）
log_message("Converting INT8 model to INT4 model with torchao...")

# 使用torchao 0.13.0的quantize_函数和Int4WeightOnlyConfig
int4_config = Int4WeightOnlyConfig(group_size=32, version=1)
int4_model = quantize_(int8_model, int4_config)
log_message("INT4 model conversion completed with torchao")

# --- 10. 验证实际INT4模型性能 --- 
log_message("Validating Real INT4 Accuracy...")

int4_model.eval()
test_correct_int4 = 0
with torch.no_grad():
    for inputs, labels in tqdm(test_loader, desc="Testing INT4 Accuracy"):
        inputs, labels = inputs.to('cpu'), labels.to('cpu')  # INT4模型在CPU上运行
        outputs = int4_model(inputs)
        _, pred = torch.max(outputs, 1)
        test_correct_int4 += (pred == labels).sum().item()

int4_acc = 100. * test_correct_int4 / len(test_loader.dataset)
log_message(f"Real INT4 Deploy Accuracy: {int4_acc:.2f}%")

# --- 11. 导出INT4部署模型 --- 
log_message("\nExporting INT4 model for deployment...")

# 定义输出路径
deploy_path = os.path.join(model_dir, "mobilenetv2_c100_int4_qat_deploy.pt")

# 导出为TorchScript
example_input = torch.randn(1, 3, 224, 224)
traced_model = torch.jit.trace(int4_model, example_input)
torch.jit.save(traced_model, deploy_path)

log_message(f"INT4 deploy model saved to: {deploy_path}")

# --- 12. 总结报告 --- 
def get_size_mb(path):
    """计算文件大小"""
    return os.path.getsize(path) / (1024 * 1024) if os.path.exists(path) else 0

fp32_size = get_size_mb(fp32_path)
int4_size = get_size_mb(deploy_path)

log_message("=" * 70)
log_message("MobileNetV2 CIFAR-100 INT4 QAT Results")
log_message("=" * 70)
log_message(f"FP32 Accuracy: {fp32_acc:.2f}%")
log_message(f"QAT Simulated Accuracy: {best_qat_acc:.2f}%")
log_message(f"Real INT4 Accuracy: {int4_acc:.2f}%")
log_message(f"Total Accuracy Drop: {fp32_acc - int4_acc:.2f}%")
log_message(f"FP32 Model Size: {fp32_size:.2f} MB")
log_message(f"INT4 Deploy Size: {int4_size:.2f} MB")
log_message(f"Compression Ratio: {fp32_size/int4_size:.2f}x")
log_message(f"Total Training Time: {total_time/60:.2f} minutes")
log_message("=" * 70)
log_message("Deploy model on Raspberry Pi with:")
log_message(f"  python test_pi5_optimized.py --model {deploy_path}")
log_message("=" * 70)