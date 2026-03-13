import os
import torch
import torch.nn as nn
import torch.ao.quantization as tq
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader

def main():
    # ==================== 配置 ====================
    device = torch.device("cpu")
    batch_size = 64
    num_test_batches = 50

    # ==================== CIFAR-10 测试集 ====================
    transform_test = transforms.Compose([
        transforms.Resize(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])
    ])

    test_dataset = datasets.CIFAR10(root="./data", train=False, download=True, transform=transform_test)
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False, num_workers=0,  # Windows 下用 0
        pin_memory=True
    )

    # ==================== 设置量化引擎 ====================
    torch.backends.quantized.engine = 'fbgemm' if 'fbgemm' in torch.backends.quantized.supported_engines else 'qnnpack'
    print(f"Quantization engine: {torch.backends.quantized.engine}")

    # ==================== 加载 QAT checkpoint ====================
    qat_path = "models/resnet18_qat_best.pth"
    if not os.path.exists(qat_path):
        raise FileNotFoundError(f"{qat_path} 不存在")

    checkpoint = torch.load(qat_path, map_location='cpu')
    print("Checkpoint 类型:", type(checkpoint))
    if isinstance(checkpoint, dict):
        print("Checkpoint 包含的键数:", len(checkpoint))
        print("部分键示例:", list(checkpoint.keys())[:10])
    else:
        print("Checkpoint 不是 dict，可能是 state_dict 没有键")

    # ==================== 创建 QAT 模型 ====================
    model = models.resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, 10)
    model.qconfig = tq.get_default_qat_qconfig(torch.backends.quantized.engine)
    model_prepared = tq.prepare_qat(model, inplace=False)

    # 加载 checkpoint
    missing, unexpected = model_prepared.load_state_dict(checkpoint, strict=False)
    print(f"缺失键: {missing[:5] if missing else '无'}")
    print(f"意外键: {unexpected[:5] if unexpected else '无'}")
    print("✓ QAT 模型加载完成")

    # ==================== 测试函数 ====================
    def test_model(model, loader, desc="模型"):
        model.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for i, (images, labels) in enumerate(loader):
                if i >= num_test_batches:
                    break
                images, labels = images.cpu(), labels.cpu()
                outputs = model(images)
                _, pred = outputs.max(1)
                correct += pred.eq(labels).sum().item()
                total += labels.size(0)
        acc = 100. * correct / total
        print(f"✓ {desc} 准确率 (前 {num_test_batches} batch): {acc:.2f}%")
        return acc

    # ==================== 测试 QAT 精度 ====================
    qat_acc = test_model(model_prepared, test_loader, "QAT 模型 (CPU)")

    # ==================== 转换为 INT8 ====================
    model_prepared.eval()
    model_prepared.cpu()
    try:
        model_int8 = tq.convert(model_prepared, inplace=False)
        print("✓ INT8 转换成功")
    except Exception as e:
        print(f"❌ INT8 转换失败: {e}")
        import traceback
        traceback.print_exc()
        exit(1)

    # ==================== 测试 INT8 精度 ====================
    int8_acc = test_model(model_int8, test_loader, "INT8 模型 (CPU)")

    # ==================== 模型大小 ====================
    qat_size = os.path.getsize(qat_path)/(1024*1024)
    int8_save_path = "models/resnet18_int8_final.pth"
    torch.save(model_int8.state_dict(), int8_save_path)
    int8_size = os.path.getsize(int8_save_path)/(1024*1024)
    fp32_path = "models/resnet18_fp32.pth"
    fp32_size = os.path.getsize(fp32_path)/(1024*1024) if os.path.exists(fp32_path) else None

    # ==================== 输出报告 ====================
    print("\n" + "="*60)
    print("QAT -> INT8 Conversion Report (CPU)")
    print("="*60)
    print(f"QAT Accuracy  : {qat_acc:.2f}%")
    print(f"INT8 Accuracy : {int8_acc:.2f}%")
    print(f"Accuracy Drop : {qat_acc - int8_acc:.2f}%\n")
    print("Model Size Comparison (MB)")
    print("-"*45)
    if fp32_size:
        print(f"{'FP32':<15} {fp32_size:>8.2f} MB")
    print(f"{'QAT':<15} {qat_size:>8.2f} MB")
    print(f"{'INT8':<15} {int8_size:>8.2f} MB")
    if fp32_size:
        print(f"\nCompression Ratio (FP32/INT8): {fp32_size/int8_size:.2f}x")
    print("="*60)
    print(f"\nINT8 模型已保存到: {int8_save_path}")

if __name__ == "__main__":
    main()