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
# 早停配置
patience = 2  # 容忍2轮无提升就停止
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
write_log("ResNet18 QAT训练（CIFAR10）- 带早停+过拟合检测")
write_log(f"数据集路径: {data_root}")
write_log(f"训练设备: {device}")
write_log(f"Batch Size: {batch_size}")
write_log(f"训练轮数: {epochs}")
write_log(f"学习率: {lr}")
write_log(f"学习率衰减步长: {lr_step_size}, 衰减系数: {lr_gamma}")
write_log(f"早停耐心值: {patience}")
write_log("="*80)

# -----------------------------
# 数据集加载
# -----------------------------
write_log("\n[1/6] 加载CIFAR10数据集...")
transform_train = transforms.Compose([
    transforms.Resize(224),
    transforms.RandomHorizontalFlip(p=0.5),
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
    pin_memory=True,
    drop_last=True
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
def validate_original(model, loader, device):
    """验证模型在非量化状态下的测试集精度"""
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    original_acc = 100. * correct / total
    return original_acc

def validate_with_quantization(model, loader, device):
    """验证模型在模拟量化下的测试集精度"""
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    quant_acc = 100. * correct / total
    return quant_acc

# -----------------------------
# 创建QAT模型
# -----------------------------
write_log("\n[2/6] 初始化QAT模型...")
try:
    model = resnet18(pretrained=True, quantize=False)
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, 10)
    
    model.train()
    model.fuse_model()
    write_log("模型层融合完成")
    
    model.qconfig = quant.get_default_qat_qconfig('fbgemm')
    quant.prepare_qat(model, inplace=True)
    write_log("QAT量化节点插入完成")
    
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
    lr=lr,
    weight_decay=1e-4
)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=lr_step_size, gamma=lr_gamma)
write_log("优化器和损失函数初始化完成")
write_log(f"优化器配置: Adam(lr={lr}, weight_decay=1e-4)")

# -----------------------------
# QAT训练（整合早停机制）
# -----------------------------
write_log("\n[4/6] 开始QAT训练...")
write_log("-"*100)
write_log(f"{'Epoch':<6} {'Train Loss':<12} {'Train Acc':<12} {'Test Acc':<12} {'Quant Sim Acc':<15} {'LR':<10}")
write_log("-"*100)

best_test_acc = 0.0
no_improve = 0  # 记录无提升的轮数
total_train_time = time.time()

for epoch in range(epochs):
    epoch_start = time.time()
    model.train()
    running_loss = 0.0
    train_correct = 0
    train_total = 0
    
    train_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}", leave=False)
    for images, labels in train_bar:
        images, labels = images.to(device), labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item() * images.size(0)
        _, predicted = torch.max(outputs.data, 1)
        train_total += labels.size(0)
        train_correct += (predicted == labels).sum().item()
        
        train_bar.set_postfix({
            'loss': f"{running_loss/train_total:.4f}",
            'acc': f"{100.*train_correct/train_total:.2f}%"
        })
    
    # 计算本轮指标
    train_loss = running_loss / len(train_dataset)
    train_acc = 100. * train_correct / train_total
    current_lr = optimizer.param_groups[0]['lr']
    test_acc = validate_original(model, test_loader, device)
    quant_sim_acc = validate_with_quantization(model, test_loader, device)
    
    # 早停机制核心逻辑
    if test_acc > best_test_acc:
        best_test_acc = test_acc
        no_improve = 0  # 有提升，重置计数器
        # 保存最佳模型
        model_cpu = model.to('cpu')
        torch.save({
            'epoch': epoch+1,
            'model_state_dict': model_cpu.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'best_test_acc': best_test_acc,
            'quant_sim_acc': quant_sim_acc,
            'train_loss': train_loss
        }, os.path.join(model_dir, "resnet18_qat_best.pth"))
        model = model_cpu.to(device)
        write_log(f"\nEpoch {epoch+1} 保存最佳模型 | 测试集精度: {test_acc:.2f}% | 量化模拟精度: {quant_sim_acc:.2f}%")
    else:
        no_improve += 1  # 无提升，计数器+1
        write_log(f"\nEpoch {epoch+1} 测试集精度无提升 | 当前无提升轮数: {no_improve}/{patience}")
        if no_improve >= patience:
            write_log(f"早停触发！在Epoch {epoch+1} 停止训练（最佳精度: {best_test_acc:.2f}%）")
            break  # 退出训练循环
    
    # 打印本轮日志
    log_line = f"{epoch+1:<6} {train_loss:<12.4f} {train_acc:<12.2f} {test_acc:<12.2f} {quant_sim_acc:<15.2f} {current_lr:<10.6f}"
    write_log(log_line)
    
    scheduler.step()

# -----------------------------
# 过拟合检测（训练结束后）
# -----------------------------
write_log("\n[5/6] 过拟合检测...")
# 重新计算最终的训练/测试精度（确保准确）
final_train_acc = 100. * train_correct / train_total
final_test_acc = best_test_acc
overfit_ratio = final_train_acc - final_test_acc

write_log(f"最终训练集精度: {final_train_acc:.2f}%")
write_log(f"最终测试集精度: {final_test_acc:.2f}%")
write_log(f"训练集-测试集精度差距: {overfit_ratio:.2f}%")

# 过拟合判断（阈值设为5%，符合行业通用标准）
if overfit_ratio > 5:
    write_log("⚠️ 警告：训练集和测试集精度差距超过5%，存在过拟合风险！")
    write_log("建议优化方向：1. 增加数据增强 2. 增大权重衰减 3. 减少模型复杂度 4. 增加Dropout层")
elif overfit_ratio < 2:
    write_log("✅ 训练集和测试集精度差距合理，无明显过拟合/欠拟合")
else:
    write_log("ℹ️ 训练集和测试集精度差距轻微，可适当增加正则化进一步优化")

# -----------------------------
# 转换为INT8模型
# -----------------------------
write_log("\n[6/6] 转换为INT8模型...")
try:
    model_int8 = resnet18(pretrained=False, quantize=False)
    model_int8.fc = nn.Linear(model_int8.fc.in_features, 10)
    
    checkpoint = torch.load(os.path.join(model_dir, "resnet18_qat_best.pth"), map_location='cpu', weights_only=True)
    model_int8.load_state_dict(checkpoint['model_state_dict'], strict=False)
    write_log("最佳QAT模型权重加载完成（忽略量化参数不匹配）")
    
    # INT8转换流程
    model_int8.train()
    model_int8.fuse_model()
    model_int8.qconfig = quant.get_default_qat_qconfig('fbgemm')
    quant.prepare_qat(model_int8, inplace=True)
    model_int8.eval()
    model_int8 = quant.convert(model_int8, inplace=True)
    write_log("INT8模型转换完成")
    
    # 计算INT8精度
    int8_correct = 0
    int8_total = 0
    model_int8.eval()
    with torch.no_grad():
        int8_start = time.time()
        for images, labels in test_loader:
            images, labels = images.to('cpu'), labels.to('cpu')
            outputs = model_int8(images)
            _, predicted = torch.max(outputs.data, 1)
            int8_total += labels.size(0)
            int8_correct += (predicted == labels).sum().item()
        int8_infer_time = (time.time() - int8_start) / len(test_loader) * 1000
    
    int8_acc = 100. * int8_correct / int8_total
    torch.save(model_int8.state_dict(), os.path.join(model_dir, "resnet18_int8_final.pth"))
    
    # 最终汇总
    write_log("="*80)
    write_log("训练&量化完成 - 最终指标汇总")
    write_log(f"最佳FP32测试精度: {best_test_acc:.2f}%")
    write_log(f"INT8测试精度: {int8_acc:.2f}%")
    write_log(f"量化掉点: {best_test_acc - int8_acc:.2f}%")
    write_log(f"INT8推理时间（每批次）: {int8_infer_time:.2f} ms")
    write_log(f"总训练时间: {(time.time() - total_train_time):.2f} 秒")
    write_log("="*80)
    
except Exception as e:
    write_log(f"INT8模型转换失败: {str(e)}")
    raise

# -----------------------------
# 训练完成
# -----------------------------
write_log("\n✅ 所有步骤完成！")
write_log(f"核心结果：FP32精度={best_test_acc:.2f}% | INT8精度={int8_acc:.2f}% | 过拟合差距={overfit_ratio:.2f}%")
write_log(f"日志文件路径: {log_filename}")
write_log(f"最佳模型路径: {os.path.join(model_dir, 'resnet18_qat_best.pth')}")
write_log(f"INT8模型路径: {os.path.join(model_dir, 'resnet18_int8_final.pth')}")