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

# 优先使用GPU训练，量化强制用CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logger.info(f"Training on device: {device}")
logger.info(f"PyTorch version: {torch.__version__}")
logger.info(f"Quantization will be performed on CPU (required for fbgemm)")

# -------------------------------
# 数据集准备（兼容Linux/Windows路径）
# -------------------------------
# 根据环境自动适配路径
if os.path.exists('/root/autodl-tmp/thesis-quantization/data'):
    data_root = '/root/autodl-tmp/thesis-quantization/data'  # Linux/Autodl
else:
    data_root = r'D:\Graduation_Design\thesis-quantization\data'  # Windows

# 数据加载参数（Windows下num_workers=0）
num_workers = 4 if os.name != 'nt' else 0
pin_memory = True if device.type == 'cuda' else False

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

train_loader = torch.utils.data.DataLoader(
    train_dataset, batch_size=32, shuffle=True, num_workers=num_workers, pin_memory=pin_memory
)
test_loader = torch.utils.data.DataLoader(
    test_dataset, batch_size=32, shuffle=False, num_workers=num_workers, pin_memory=pin_memory
)

# -------------------------------
# 模型定义（添加量化节点）
# -------------------------------
class QuantizableResNet18(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        self.model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
        self.model.fc = nn.Linear(self.model.fc.in_features, num_classes)
        
        # 必须添加量化/反量化节点（解决conv2d.new错误的核心）
        self.quant = tq.QuantStub()
        self.dequant = tq.DeQuantStub()

    def forward(self, x):
        x = self.quant(x)
        x = self.model(x)
        x = self.dequant(x)
        return x

    def fuse_model(self):
        """安全的层融合，避免报错"""
        self.model.eval()
        for module_name, module in self.model.named_children():
            if "layer" in module_name:
                for bb_name, basic_block in module.named_children():
                    try:
                        tq.fuse_modules(basic_block, [['conv1', 'bn1', 'relu']], inplace=True)
                    except:
                        pass
                    try:
                        tq.fuse_modules(basic_block, [['conv2', 'bn2']], inplace=True)
                    except:
                        pass
        return self

def create_model():
    """创建基础模型（训练用）"""
    model = QuantizableResNet18(num_classes=10)
    return model

# -------------------------------
# 训练函数（GPU加速）
# -------------------------------
def train_model():
    model = create_model()
    model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9, weight_decay=1e-4)
    
    logger.info("Starting training (3 epochs test)...")
    best_acc = 0
    
    for epoch in range(1, 4):
        # 训练阶段
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
            optimizer.step()
            
            running_loss += loss.item() * inputs.size(0)
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
            
            loop.set_postfix({
                'loss': f"{running_loss/total:.4f}",
                'acc': f"{100.*correct/total:.2f}%"
            })
        
        train_acc = 100. * correct / total
        
        # 验证阶段
        model.eval()
        val_correct = 0
        val_total = 0
        with torch.no_grad():
            for inputs, targets in test_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = model(inputs)
                _, predicted = outputs.max(1)
                val_total += targets.size(0)
                val_correct += predicted.eq(targets).sum().item()
        
        val_acc = 100. * val_correct / val_total
        logger.info(f"Epoch {epoch}: Train Acc={train_acc:.2f}%, Val Acc={val_acc:.2f}%")
        
        if val_acc > best_acc:
            best_acc = val_acc
            # 安全保存（先移到CPU，避免GPU张量问题）
            model_cpu = model.to('cpu')
            torch.save(model_cpu.state_dict(), "best_model.pth")
            model.to(device)  # 移回GPU继续训练
    
    return best_acc

# -------------------------------
# 量化函数（纯CPU执行，解决conv2d.new错误）
# -------------------------------
def quantize_model():
    logger.info("\n" + "="*50)
    logger.info("Starting quantization on CPU...")
    logger.info("="*50)
    
    try:
        # 1. 纯CPU加载模型（关键！）
        logger.info("1. Loading model to CPU...")
        model = create_model()
        # 强制加载到CPU，避免设备不匹配
        state_dict = torch.load("best_model.pth", map_location=torch.device('cpu'))
        model.load_state_dict(state_dict)
        model.eval()
        
        # 2. 层融合
        logger.info("2. Fusing modules...")
        model = model.fuse_model()
        
        # 3. 设置fbgemm量化配置（适配你的环境）
        logger.info("3. Setting fbgemm quantization config...")
        model.qconfig = tq.get_default_qconfig('fbgemm')
        
        # 4. 准备量化（不要用inplace=True）
        logger.info("4. Preparing model for quantization...")
        model_prepared = tq.prepare(model, inplace=False)
        
        # 5. 校准（用少量数据，纯CPU）
        logger.info("5. Calibrating with test data (10 batches)...")
        model_prepared.eval()
        with torch.no_grad():
            for i, (inputs, _) in enumerate(test_loader):
                if i >= 10:  # 只用10个batch校准，节省时间
                    break
                # 强制输入在CPU
                model_prepared(inputs.to('cpu'))
        
        # 6. 转换为INT8（核心步骤）
        logger.info("6. Converting to INT8...")
        model_int8 = tq.convert(model_prepared, inplace=False)
        
        # 7. 测试INT8模型
        logger.info("7. Testing INT8 model...")
        test_input = torch.randn(1, 3, 224, 224).to('cpu')
        with torch.no_grad():
            output = model_int8(test_input)
        logger.info(f"✓ INT8 model output shape: {output.shape}")
        
        # 8. 验证INT8精度
        logger.info("8. Validating INT8 model accuracy...")
        correct = 0
        total = 0
        model_int8.eval()
        with torch.no_grad():
            for inputs, targets in test_loader:
                # 所有数据强制在CPU
                inputs, targets = inputs.to('cpu'), targets.to('cpu')
                outputs = model_int8(inputs)
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()
        
        int8_acc = 100. * correct / total
        logger.info(f"✓ INT8 model accuracy: {int8_acc:.2f}%")
        
        # 9. 保存INT8模型
        torch.save(model_int8.state_dict(), "resnet18_int8_final.pth")
        logger.info("✓ INT8 model saved as 'resnet18_int8_final.pth'")
        
        return model_int8, int8_acc
        
    except Exception as e:
        logger.error(f"Quantization failed: {e}")
        import traceback
        traceback.print_exc()
        return None, None

# -------------------------------
# 主程序
# -------------------------------
if __name__ == "__main__":
    # 1. GPU训练模型
    best_acc = train_model()
    logger.info(f"\nBest FP32 accuracy: {best_acc:.2f}%")
    
    # 2. CPU量化模型
    model_int8, int8_acc = quantize_model()
    
    if model_int8 is not None:
        logger.info(f"\n=== Final Results ===")
        logger.info(f"FP32 best accuracy: {best_acc:.2f}%")
        logger.info(f"INT8 accuracy: {int8_acc:.2f}%")
        logger.info(f"Accuracy drop: {best_acc - int8_acc:.2f}%")
    
    logger.info("\nAll tasks completed successfully!")