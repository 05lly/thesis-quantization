# qat_resnet18_fixed_v4.py
import torch
import torch.nn as nn
import torch.optim as optim
import torch.ao.quantization as tq
from torchvision import datasets, transforms, models
from tqdm import tqdm
import os
import logging
import warnings
warnings.filterwarnings('ignore')

# 设置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logger.info(f"Training on device: {device}")

# -------------------------------
# 数据集准备
# -------------------------------
data_root = '/root/autodl-tmp/thesis-quantization/data'
transform_train = transforms.Compose([
    transforms.Resize(224),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

transform_test = transforms.Compose([
    transforms.Resize(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

train_dataset = datasets.CIFAR10(root=data_root, train=True, download=False, transform=transform_train)
test_dataset = datasets.CIFAR10(root=data_root, train=False, download=False, transform=transform_test)

train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=4, pin_memory=True)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=4, pin_memory=True)

# -------------------------------
# 模型定义与融合
# -------------------------------
def create_model():
    model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
    model.fc = nn.Linear(model.fc.in_features, 10)
    return model

def fuse_model(model):
    """正确的ResNet18融合方式"""
    model.eval()
    for module_name, module in model.named_children():
        if "layer" in module_name:
            for basic_block_name, basic_block in module.named_children():
                tq.fuse_modules(basic_block, [['conv1', 'bn1', 'relu']], inplace=True)
                tq.fuse_modules(basic_block, [['conv2', 'bn2']], inplace=True)
    return model

# -------------------------------
# 初始化模型
# -------------------------------
model = create_model()
model = fuse_model(model)

# -------------------------------
# QAT准备
# -------------------------------
model.train()
model.qconfig = tq.get_default_qat_qconfig('fbgemm')
tq.prepare_qat(model, inplace=True)
model.to(device)

# -------------------------------
# 优化器设置
# -------------------------------
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.0001, momentum=0.9, weight_decay=1e-4)
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=30)

# -------------------------------
# 训练函数
# -------------------------------
def train_one_epoch(epoch):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    loop = tqdm(train_loader, desc=f"Epoch {epoch}")
    for inputs, targets in loop:
        inputs, targets = inputs.to(device), targets.to(device)
        
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        optimizer.step()
        
        running_loss += loss.item() * inputs.size(0)
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()
        
        loop.set_postfix({
            'loss': running_loss/total,
            'acc': 100.*correct/total
        })
    
    return running_loss/total, 100.*correct/total

def validate(model, loader):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for inputs, targets in loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            
            running_loss += loss.item() * inputs.size(0)
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
    
    return running_loss/total, 100.*correct/total

# -------------------------------
# 训练循环
# -------------------------------
best_acc = 0
num_epochs = 10  # 增加到10个epoch

logger.info("Starting QAT training...")
for epoch in range(1, num_epochs+1):
    train_loss, train_acc = train_one_epoch(epoch)
    val_loss, val_acc = validate(model, test_loader)
    
    logger.info(f"Epoch {epoch}: Train Loss={train_loss:.4f}, Train Acc={train_acc:.2f}%")
    logger.info(f"Validation: Loss={val_loss:.4f}, Acc={val_acc:.2f}%")
    
    if val_acc > best_acc:
        best_acc = val_acc
        # 保存训练好的QAT模型
        torch.save(model.state_dict(), "best_qat_model_weights.pth")
        logger.info(f"New best model saved with accuracy {best_acc:.2f}%")
    
    scheduler.step()

# -------------------------------
# 转换为INT8 - 修复版
# -------------------------------
logger.info("Converting to INT8...")

# 方法1：直接使用训练好的模型（最简单）
logger.info("Method 1: Converting trained model directly...")
model.eval()
model.to('cpu')
model_int8 = tq.convert(model)

# 验证INT8模型
val_loss, val_acc = validate(model_int8, test_loader)
logger.info(f"INT8 Model (direct) - Loss: {val_loss:.4f}, Acc: {val_acc:.2f}%")
logger.info(f"Accuracy drop: {best_acc - val_acc:.2f}%")

# 保存INT8模型
torch.save(model_int8.state_dict(), "resnet18_int8_direct.pth")

# 方法2：重新创建模型并加载权重（如果需要）
logger.info("Method 2: Creating new model and loading weights...")

def create_model_for_conversion():
    # 创建新模型
    new_model = create_model()
    new_model = fuse_model(new_model)
    
    # 设置为训练模式进行prepare_qat
    new_model.train()
    new_model.qconfig = tq.get_default_qat_qconfig('fbgemm')
    tq.prepare_qat(new_model, inplace=True)
    
    # 加载训练好的权重
    new_model.load_state_dict(torch.load("best_qat_model_weights.pth"))
    
    # 切换到eval模式进行转换
    new_model.eval()
    new_model.to('cpu')
    return new_model

# 创建新模型并转换
new_model = create_model_for_conversion()
new_model_int8 = tq.convert(new_model)

# 验证新INT8模型
val_loss2, val_acc2 = validate(new_model_int8, test_loader)
logger.info(f"INT8 Model (new) - Loss: {val_loss2:.4f}, Acc: {val_acc2:.2f}%")

# 保存第二个INT8模型
torch.save(new_model_int8.state_dict(), "resnet18_int8_new.pth")

# 保存模型信息
model_info = {
    'model_architecture': 'resnet18_qat_int8',
    'input_size': (3, 224, 224),
    'num_classes': 10,
    'accuracy_direct': val_acc,
    'accuracy_new': val_acc2,
    'best_fp32_acc': best_acc
}
torch.save(model_info, "resnet18_int8_info.pth")

logger.info("Training and quantization completed successfully!")

# -------------------------------
# 快速测试
# -------------------------------
def quick_test():
    """快速测试模型是否能正常运行"""
    logger.info("Running quick test...")
    
    # 测试直接转换的模型
    test_model = model_int8
    test_input = torch.randn(1, 3, 224, 224)
    with torch.no_grad():
        output = test_model(test_input)
    logger.info(f"Direct model test passed! Output shape: {output.shape}")
    
    # 测试新创建的模型
    test_model2 = new_model_int8
    with torch.no_grad():
        output2 = test_model2(test_input)
    logger.info(f"New model test passed! Output shape: {output2.shape}")
    
    return True

quick_test()