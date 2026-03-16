# qat_resnet18_working.py
import torch
import torch.nn as nn
import torch.optim as optim
import torch.ao.quantization as tq
from torchvision import datasets, transforms
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
logger.info(f"PyTorch version: {torch.__version__}")

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

train_loader = torch.utils.data.DataLoader(
    train_dataset, batch_size=32, shuffle=True, num_workers=4, pin_memory=True
)
test_loader = torch.utils.data.DataLoader(
    test_dataset, batch_size=32, shuffle=False, num_workers=4, pin_memory=True
)

# -------------------------------
# 可量化的BasicBlock（完整修复）
# -------------------------------
class QuantizableBasicBlock(nn.Module):
    """完全可量化的BasicBlock"""
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super().__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu1 = nn.ReLU(inplace=False)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.relu2 = nn.ReLU(inplace=False)
        self.downsample = downsample
        self.stride = stride
        
        # 用于量化残差连接的节点
        self.skip_add = nn.quantized.FloatFunctional()

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu1(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        # 使用FloatFunctional进行安全的加法操作（解决aten::add.out错误）
        out = self.skip_add.add_relu(out, identity)

        return out

class QuantizableResNet18(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        self.inplanes = 64
        
        # 第一层
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=False)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        
        # 残差层
        self.layer1 = self._make_layer(QuantizableBasicBlock, 64, 2)
        self.layer2 = self._make_layer(QuantizableBasicBlock, 128, 2, stride=2)
        self.layer3 = self._make_layer(QuantizableBasicBlock, 256, 2, stride=2)
        self.layer4 = self._make_layer(QuantizableBasicBlock, 512, 2, stride=2)
        
        # 分类头
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512, num_classes)
        
        # 量化节点
        self.quant = tq.QuantStub()
        self.dequant = tq.DeQuantStub()

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.quant(x)
        
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        
        x = self.dequant(x)
        return x

    def fuse_model(self):
        """安全的层融合"""
        # 融合第一层
        tq.fuse_modules(self, [['conv1', 'bn1', 'relu']], inplace=True)
        
        # 融合残差块
        for module_name in ['layer1', 'layer2', 'layer3', 'layer4']:
            layer = getattr(self, module_name)
            for basic_block in layer:
                # 融合conv1, bn1, relu1
                tq.fuse_modules(basic_block, [['conv1', 'bn1', 'relu1']], inplace=True)
                # 融合conv2, bn2
                tq.fuse_modules(basic_block, [['conv2', 'bn2']], inplace=True)
        
        return self

# -------------------------------
# 训练函数
# -------------------------------
def train_model():
    model = QuantizableResNet18(num_classes=10)
    model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9, weight_decay=1e-4)
    
    logger.info("Starting training (3 epochs test)...")
    best_acc = 0
    
    for epoch in range(1, 4):
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
        
        # 验证
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
            # 保存到CPU
            model_cpu = model.to('cpu')
            torch.save(model_cpu.state_dict(), "best_model.pth")
            model.to(device)
    
    return best_acc

# -------------------------------
# 量化函数（使用FloatFunctional）
# -------------------------------
def quantize_model():
    logger.info("\n" + "="*50)
    logger.info("Starting quantization on CPU...")
    logger.info("="*50)
    
    try:
        # 1. 加载模型到CPU
        logger.info("1. Loading model to CPU...")
        model = QuantizableResNet18(num_classes=10)
        state_dict = torch.load("best_model.pth", map_location='cpu')
        model.load_state_dict(state_dict)
        model.eval()
        
        # 2. 层融合
        logger.info("2. Fusing modules...")
        model = model.fuse_model()
        
        # 3. 设置量化配置
        logger.info("3. Setting quantization config...")
        model.qconfig = tq.get_default_qconfig('fbgemm')
        
        # 4. 准备量化
        logger.info("4. Preparing model for quantization...")
        model_prepared = tq.prepare(model, inplace=False)
        
        # 5. 校准
        logger.info("5. Calibrating with test data...")
        model_prepared.eval()
        with torch.no_grad():
            for i, (inputs, _) in enumerate(test_loader):
                if i >= 10:
                    break
                model_prepared(inputs)
        
        # 6. 转换为INT8
        logger.info("6. Converting to INT8...")
        model_int8 = tq.convert(model_prepared, inplace=False)
        
        # 7. 测试
        logger.info("7. Testing INT8 model...")
        test_input = torch.randn(1, 3, 224, 224)
        with torch.no_grad():
            output = model_int8(test_input)
        logger.info(f"✓ INT8 model output shape: {output.shape}")
        
        # 8. 验证精度
        logger.info("8. Validating INT8 model accuracy...")
        correct = 0
        total = 0
        with torch.no_grad():
            for inputs, targets in test_loader:
                outputs = model_int8(inputs)
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()
        
        int8_acc = 100. * correct / total
        logger.info(f"✓ INT8 model accuracy: {int8_acc:.2f}%")
        
        # 9. 保存
        torch.save(model_int8.state_dict(), "resnet18_int8_final.pth")
        logger.info("✓ INT8 model saved successfully!")
        
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
    # 训练
    best_acc = train_model()
    logger.info(f"\nBest FP32 accuracy: {best_acc:.2f}%")
    
    # 量化
    model_int8, int8_acc = quantize_model()
    
    if model_int8 is not None:
        logger.info(f"\n=== Final Results ===")
        logger.info(f"FP32 best accuracy: {best_acc:.2f}%")
        logger.info(f"INT8 accuracy: {int8_acc:.2f}%")
        logger.info(f"Accuracy drop: {best_acc - int8_acc:.2f}%")
    
    logger.info("\nAll tasks completed!")