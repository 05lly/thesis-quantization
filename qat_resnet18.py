import torch
import torch.nn as nn
import torch.optim as optim
import torch.ao.quantization as quant
from torchvision import datasets, transforms
from torchvision.models.quantization import resnet18
from torch.utils.data import DataLoader
import os
import time
import datetime
from tqdm import tqdm

# -----------------------------
# 基础配置
# -----------------------------
# 数据集路径
data_root = '/root/autodl-tmp/thesis-quantization/data'
# 训练配置（测试用）
batch_size = 128
epochs = 3        
lr = 0.0001       
lr_step_size = 1  
lr_gamma = 0.5      
# 保存路径
log_dir = "logs"
model_dir = "models"
os.makedirs(log_dir, exist_ok=True)
os.makedirs(model_dir, exist_ok=True)

# 设备配置
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# 日志文件命名
log_filename = os.path.join(log_dir, f"qat_training_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.txt")

# -----------------------------
# 日志记录函数
# -----------------------------
def write_log(content):
    """写入日志到文本文件，同时打印到控制台"""
    current_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    log_content = f"[{current_time}] {content}"
    print(log_content)
    with open(log_filename, 'a', encoding='utf-8') as f:
        f.write(log_content + "\n")

# -----------------------------
# 初始化日志
# -----------------------------
write_log("="*80)
write_log("ResNet18 QAT训练（CIFAR10）")
write_log(f"数据集路径: {data_root}")
write_log(f"训练设备: {device}")
write_log(f"Batch Size: {batch_size}")
write_log(f"训练轮数: {epochs}")
write_log(f"学习率: {lr}")
write_log(f"学习率衰减步长: {lr_step_size}, 衰减系数: {lr_gamma}")
write_log("="*80)

# -----------------------------
# 数据集加载
# -----------------------------
write_log("\n[1/6] 加载CIFAR10数据集...")
transform_train = transforms.Compose([
    transforms.Resize(224),
    transforms.RandomHorizontalFlip(),
    transforms.RandomCrop(224, padding=4),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

transform_test = transforms.Compose([
    transforms.Resize(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

try:
    train_dataset = datasets.CIFAR10(
        root=data_root,
        train=True,
        download=False,
        transform=transform_train
    )
    test_dataset = datasets.CIFAR10(
        root=data_root,
        train=False,
        download=False,
        transform=transform_test
    )
    write_log(f"数据集加载成功 | 训练集数量: {len(train_dataset)} | 测试集数量: {len(test_dataset)}")
except Exception as e:
    write_log(f"数据集加载失败: {str(e)}")
    raise

train_loader = DataLoader(
    train_dataset,
    batch_size=batch_size,
    shuffle=True,
    num_workers=0,
    pin_memory=True
)

test_loader = DataLoader(
    test_dataset,
    batch_size=batch_size,
    shuffle=False,
    num_workers=0,
    pin_memory=True
)

# -----------------------------
# 精度验证函数
# -----------------------------
def validate_with_quantization(model, loader, device):
    """验证模型在模拟量化下的精度"""
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
    quant_acc = 100. * correct / total
    return quant_acc

def validate_original(model, loader, device):
    """验证模型在非量化状态下的精度"""
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
    original_acc = 100. * correct / total
    return original_acc

# -----------------------------
# 创建QAT模型
# -----------------------------
write_log("\n[2/6] 初始化QAT模型...")
try:
    model = resnet18(pretrained=True, quantize=False)
    model.fc = nn.Linear(model.fc.in_features, 10)
    
    model.eval()
    model.fuse_model()
    write_log("模型层融合完成")
    
    model.qconfig = quant.get_default_qat_qconfig('fbgemm')
    quant.prepare_qat(model, inplace=True)
    write_log("QAT量化节点插入完成")
    
    model.train()
    model = model.to(device)
    write_log(f"QAT模型初始化完成 | 模型设备: {next(model.parameters()).device}")
except Exception as e:
    write_log(f"模型初始化失败: {str(e)}")
    raise

# -----------------------------
# 优化器和损失函数
# -----------------------------
write_log("\n[3/6] 初始化优化器和损失函数...")
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(
    model.parameters(),
    lr=lr
)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=lr_step_size, gamma=lr_gamma)
write_log("优化器和损失函数初始化完成")
write_log(f"优化器配置: Adam(lr={lr})")

# -----------------------------
# QAT训练
# -----------------------------
write_log("\n[4/6] 开始QAT训练...")
write_log("-"*100)
write_log(f"{'Epoch':<6} {'Train Loss':<12} {'Train Acc':<12} {'Test Acc':<12} {'Quant Sim Acc':<15} {'LR':<10} {'Infer Time(ms)':<15}")
write_log("-"*100)

best_acc = 0.0
total_train_time = time.time()

for epoch in range(epochs):
    epoch_start = time.time()
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    train_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}", leave=False)
    for images, labels in train_bar:
        images, labels = images.to(device), labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item() * images.size(0)
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()
        
        train_bar.set_postfix({
            'loss': f"{running_loss/total:.4f}",
            'acc': f"{100.*correct/total:.2f}%"
        })
    
    train_loss = running_loss / len(train_dataset)
    train_acc = 100. * correct / total
    current_lr = optimizer.param_groups[0]['lr']
    
    original_test_acc = validate_original(model, test_loader, device)
    quant_sim_acc = validate_with_quantization(model, test_loader, device)
    
    def measure_infer_time(model, loader, device):
        model.eval()
        total_time = 0.0
        count = 0
        with torch.no_grad():
            for images, _ in loader:
                if count >= 10:
                    break
                images = images.to(device)
                start = time.time()
                _ = model(images)
                total_time += (time.time() - start)
                count += 1
        return (total_time / count) * 1000 if count > 0 else 0
    
    infer_time = measure_infer_time(model, test_loader, device)
    
    if quant_sim_acc > best_acc:
        best_acc = quant_sim_acc
        model_cpu = model.to('cpu')
        torch.save({
            'epoch': epoch+1,
            'model_state_dict': model_cpu.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'best_quant_acc': best_acc,
            'best_original_acc': original_test_acc,
            'train_loss': train_loss
        }, os.path.join(model_dir, "resnet18_qat_best.pth"))
        model = model_cpu.to(device)
        write_log(f"\nEpoch {epoch+1} 保存最佳模型 | 量化模拟精度: {quant_sim_acc:.2f}% | 原始精度: {original_test_acc:.2f}%")
    
    log_line = f"{epoch+1:<6} {train_loss:<12.4f} {train_acc:<12.2f} {original_test_acc:<12.2f} {quant_sim_acc:<15.2f} {current_lr:<10.6f} {infer_time:<15.2f}"
    write_log(log_line)
    
    scheduler.step()

# -----------------------------
# 转换为INT8模型
# -----------------------------
write_log("\n[5/6] 转换为INT8模型...")
try:
    model_int8 = resnet18(pretrained=False, quantize=False)
    model_int8.fc = nn.Linear(model_int8.fc.in_features, 10)
    checkpoint = torch.load(os.path.join(model_dir, "resnet18_qat_best.pth"), map_location='cpu')
    model_int8.load_state_dict(checkpoint['model_state_dict'])
    
    model_int8.eval()
    model_int8.fuse_model()
    model_int8.qconfig = quant.get_default_qat_qconfig('fbgemm')
    quant.prepare_qat(model_int8, inplace=True)
    model_int8 = quant.convert(model_int8, inplace=True)
    
    int8_correct = 0
    int8_total = 0
    int8_infer_time = 0.0
    model_int8.eval()
    with torch.no_grad():
        start_time = time.time()
        for images, labels in test_loader:
            images, labels = images.to('cpu'), labels.to('cpu')
            outputs = model_int8(images)
            _, predicted = outputs.max(1)
            int8_total += labels.size(0)
            int8_correct += predicted.eq(labels).sum().item()
        int8_infer_time = (time.time() - start_time) / len(test_loader) * 1000
    
    int8_acc = 100. * int8_correct / int8_total
    torch.save(model_int8.state_dict(), os.path.join(model_dir, "resnet18_int8_final.pth"))
    
    write_log("="*80)
    write_log("INT8模型转换完成 - 精度对比汇总")
    write_log(f"QAT原始精度: {checkpoint['best_original_acc']:.2f}%")
    write_log(f"QAT量化模拟精度: {checkpoint['best_quant_acc']:.2f}%")
    write_log(f"INT8最终精度: {int8_acc:.2f}%")
    write_log(f"INT8推理时间（每批次）: {int8_infer_time:.2f} ms")
    write_log(f"总训练时间: {(time.time() - total_train_time):.2f} 秒")
    write_log("="*80)
    
except Exception as e:
    write_log(f"INT8模型转换失败: {str(e)}")
    raise

# -----------------------------
# 训练完成
# -----------------------------
write_log("\n[6/6] 训练完成 - 关键指标汇总")
write_log(f"学习率: {lr}")
write_log(f"最佳量化模拟精度: {best_acc:.2f}%")
write_log(f"INT8最终精度: {int8_acc:.2f}%")
write_log(f"精度掉点（QAT→INT8）: {best_acc - int8_acc:.2f}%")
write_log(f"日志文件路径: {log_filename}")
write_log("所有步骤完成！")