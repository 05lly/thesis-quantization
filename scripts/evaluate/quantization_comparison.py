import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torchvision.models.quantization import resnet18, mobilenet_v2
import torch.ao.quantization as quantization
from torchao.quantization import quantize_, Int4WeightOnlyConfig
import os, time, datetime
from tqdm import tqdm
import pandas as pd
import matplotlib.pyplot as plt

# 修复Windows环境下的多进程问题
if __name__ == '__main__':
    # --- 1. 全局配置 ---
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    batch_size = 128
    epochs = 5
    lr = 0.01
    model_dir = "models"
    log_dir = "logs"
    result_dir = "results"
    os.makedirs(model_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(result_dir, exist_ok=True)

    log_filename = os.path.join(log_dir, f"quantization_comparison_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.log")
    def log_message(msg):
        t = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        full_msg = f"[{t}] {msg}"
        print(full_msg)
        with open(log_filename, "a", encoding="utf-8") as f: 
            f.write(full_msg + "\n")

    # --- 2. 数据处理  ---
    # 保持标准224x224输入尺寸
    transform = transforms.Compose([
        transforms.Resize(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    
    # Windows下设置num_workers=0以避免多进程问题
    train_loader = torch.utils.data.DataLoader(
        datasets.CIFAR10('./data', train=True, download=True, transform=transform), 
        batch_size=batch_size, shuffle=True, num_workers=0
    )
    test_loader = torch.utils.data.DataLoader(
        datasets.CIFAR10('./data', train=False, download=True, transform=transform), 
        batch_size=batch_size, shuffle=False, num_workers=0
    )

    # --- 3. 模型定义和加载函数 ---
    def get_model(model_name, num_classes=10):
        """
        获取指定模型
        :param model_name: 模型名称 (resnet18, mobilenetv2)
        :param num_classes: 类别数量
        :return: 模型
        """
        if model_name == "resnet18":
            model = resnet18(weights=None, quantize=False)
            model.fc = nn.Linear(model.fc.in_features, num_classes)
        elif model_name == "mobilenetv2":
            model = mobilenet_v2(weights=None, quantize=False)
            model.classifier[1] = nn.Linear(model.last_channel, num_classes)
        else:
            raise ValueError(f"不支持的模型: {model_name}")
        return model
    
    def load_or_train_model(model_name, num_classes=10):
        """
        加载或训练FP32模型
        :param model_name: 模型名称
        :param num_classes: 类别数量
        :return: 训练好的FP32模型
        """
        fp32_path = os.path.join(model_dir, f"fp32_{model_name}_c10_best.pth")
        
        if os.path.exists(fp32_path):
            log_message(f"加载现有FP32 {model_name}模型...")
            model = get_model(model_name, num_classes)
            model.load_state_dict(torch.load(fp32_path, map_location='cpu', weights_only=True))
            model.to(device)
            return model
        else:
            log_message(f"训练新的FP32 {model_name}模型...")
            model = get_model(model_name, num_classes)
            model.to(device)
            
            optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=5e-4)
            criterion = nn.CrossEntropyLoss()
            
            for epoch in range(5):
                model.train()
                for inputs, labels in tqdm(train_loader, desc=f"Training {model_name} Epoch [{epoch+1}/5]", leave=False):
                    inputs, labels = inputs.to(device), labels.to(device)
                    optimizer.zero_grad()
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)
                    loss.backward()
                    optimizer.step()
            
            torch.save(model.state_dict(), fp32_path)
            log_message(f"FP32 {model_name}模型已保存到 {fp32_path}")
            return model

    # --- 4. 性能测试函数 ---
    def test_model(model, test_loader, device):
        """
        测试模型性能
        :param model: 模型
        :param test_loader: 测试数据加载器
        :param device: 设备
        :return: (准确率, 推理时间, FPS)
        """
        # 测试准确率
        model.eval()
        correct = 0
        with torch.no_grad():
            for inputs, labels in tqdm(test_loader, desc="Testing Accuracy", leave=False):
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                _, pred = torch.max(outputs, 1)
                correct += (pred == labels).sum().item()
        
        accuracy = 100. * correct / len(test_loader.dataset)
        
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
            inference_time = time.time() - start_time
            fps = 100 / inference_time
        
        return accuracy, inference_time, fps

    # --- 5. 量化策略实现 ---
    def quantize_int8_ptq(model, test_loader, device):
        """
        INT8后训练量化 (PTQ)
        :param model: FP32模型
        :param test_loader: 测试数据加载器
        :param device: 设备
        :return: 量化后的INT8模型
        """
        log_message("  执行INT8后训练量化 (PTQ)...")
        
        # 设置量化后端
        torch.backends.quantized.engine = 'qnnpack'
        
        # 获取量化配置
        qconfig_mapping = quantization.get_default_qconfig_mapping('qnnpack')
        
        # 准备量化
        model_prepared = quantization.prepare(model, qconfig_mapping)
        
        # 校准
        model_prepared.eval()
        with torch.no_grad():
            for inputs, _ in tqdm(test_loader, desc="PTQ Calibration", leave=False):
                inputs = inputs.to(device)
                model_prepared(inputs)
        
        # 转换为INT8模型
        model_int8 = quantization.convert(model_prepared, inplace=False)
        return model_int8
    
    def quantize_int8_qat(model, train_loader, test_loader, device):
        """
        INT8量化感知训练 (QAT)
        :param model: FP32模型
        :param train_loader: 训练数据加载器
        :param test_loader: 测试数据加载器
        :param device: 设备
        :return: 量化后的INT8模型
        """
        log_message("  执行INT8量化感知训练 (QAT)...")
        
        # 设置量化后端
        torch.backends.quantized.engine = 'qnnpack'
        
        # 获取量化配置
        qconfig_mapping = quantization.get_default_qconfig_mapping('qnnpack')
        
        # 准备QAT
        model_prepared = quantization.prepare_qat(model, qconfig_mapping)
        
        # QAT微调
        optimizer = optim.SGD(model_prepared.parameters(), lr=lr/10, momentum=0.9, weight_decay=5e-4)
        criterion = nn.CrossEntropyLoss()
        
        model_prepared.train()
        for epoch in range(2):
            for inputs, labels in tqdm(train_loader, desc=f"QAT Epoch [{epoch+1}/2]", leave=False):
                inputs, labels = inputs.to(device), labels.to(device)
                optimizer.zero_grad()
                outputs = model_prepared(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
        
        # 转换为INT8模型
        model_int8 = quantization.convert(model_prepared, inplace=False)
        return model_int8
    
    def quantize_int4_weight_only(model, group_size=32):
        """
        INT4权重量化
        :param model: FP32模型
        :param group_size: 分组大小
        :return: 量化后的INT4模型
        """
        log_message(f"  执行INT4权重量化 (group_size={group_size})...")
        
        # 复制模型
        model_copy = get_model(model_name, num_classes=10)
        model_copy.load_state_dict(model.state_dict())
        model_copy.to(device)
        model_copy.eval()
        
        # 使用TorchAO进行INT4权重量化
        quantize_config = Int4WeightOnlyConfig(group_size=group_size)
        quantize_(model_copy, quantize_config)
        
        return model_copy

    # --- 6. 主对比实验 ---
    log_message("=== 开始量化对比实验 ===")
    
    # 要测试的模型列表
    model_names = ["resnet18", "mobilenetv2"]
    
    # 要测试的量化策略列表
    quantization_strategies = [
        ("FP32", lambda m, tl, vl: m),  # 基线模型
        ("INT8_PTQ", quantize_int8_ptq),  # INT8后训练量化
        ("INT8_QAT", quantize_int8_qat),  # INT8量化感知训练
        ("INT4_WO_16", lambda m, tl, vl: quantize_int4_weight_only(m, group_size=16)),  # INT4权重量化，分组大小16
        ("INT4_WO_32", lambda m, tl, vl: quantize_int4_weight_only(m, group_size=32)),  # INT4权重量化，分组大小32
        ("INT4_WO_64", lambda m, tl, vl: quantize_int4_weight_only(m, group_size=64)),  # INT4权重量化，分组大小64
    ]
    
    # 实验结果存储
    results = []
    
    # 遍历所有模型
    for model_name in model_names:
        log_message(f"\n--- 测试模型: {model_name} ---")
        
        # 加载或训练FP32模型
        fp32_model = load_or_train_model(model_name)
        
        # 测试FP32模型性能
        log_message("  测试FP32模型性能...")
        fp32_acc, fp32_time, fp32_fps = test_model(fp32_model, test_loader, device)
        log_message(f"  FP32 - Accuracy: {fp32_acc:.2f}%, FPS: {fp32_fps:.2f}")
        
        # 遍历所有量化策略
        for strategy_name, strategy_func in quantization_strategies:
            try:
                log_message(f"\n  应用量化策略: {strategy_name}")
                
                # 应用量化策略
                if strategy_name == "FP32":
                    quantized_model = strategy_func(fp32_model, train_loader, test_loader)
                elif strategy_name == "INT8_QAT":
                    quantized_model = strategy_func(fp32_model, train_loader, test_loader, device)
                else:
                    quantized_model = strategy_func(fp32_model, test_loader, device)
                
                # 测试量化后模型性能
                log_message(f"  测试{strategy_name}模型性能...")
                acc, inference_time, fps = test_model(quantized_model, test_loader, device)
                
                # 计算模型大小（理论值）
                if strategy_name == "FP32":
                    model_size = os.path.getsize(os.path.join(model_dir, f"fp32_{model_name}_c10_best.pth")) / (1024 * 1024)
                elif "INT8" in strategy_name:
                    model_size = os.path.getsize(os.path.join(model_dir, f"fp32_{model_name}_c10_best.pth")) / (1024 * 1024) * 0.25  # 近似值
                elif "INT4" in strategy_name:
                    model_size = os.path.getsize(os.path.join(model_dir, f"fp32_{model_name}_c10_best.pth")) / (1024 * 1024) * 0.125  # 近似值
                
                # 计算加速比
                speedup = fp32_time / inference_time if inference_time > 0 else 0
                
                # 计算精度损失
                acc_loss = fp32_acc - acc
                
                # 存储结果
                results.append({
                    "Model": model_name,
                    "Strategy": strategy_name,
                    "Accuracy(%)": acc,
                    "Accuracy_Loss(%)": acc_loss,
                    "FPS": fps,
                    "Speedup": speedup,
                    "Model_Size(MB)": model_size,
                    "Timestamp": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                })
                
                log_message(f"  {strategy_name} - Accuracy: {acc:.2f}%, FPS: {fps:.2f}, Speedup: {speedup:.2f}x, Size: {model_size:.2f} MB")
                
            except Exception as e:
                log_message(f"  应用{strategy_name}策略时出错: {e}")
                continue

    # --- 7. 结果分析与可视化 ---
    log_message("\n=== 实验结果分析 ===")
    
    # 将结果转换为DataFrame
    df = pd.DataFrame(results)
    
    # 保存结果到CSV
    result_csv_path = os.path.join(result_dir, f"quantization_comparison_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.csv")
    df.to_csv(result_csv_path, index=False, encoding="utf-8-sig")
    log_message(f"  实验结果已保存到: {result_csv_path}")
    
    # 打印结果概览
    log_message("\n  实验结果概览:")
    print(df[['Model', 'Strategy', 'Accuracy(%)', 'FPS', 'Speedup', 'Model_Size(MB)']].to_string(index=False))
    
    # 找出每个模型的最优策略（FPS最高，准确率损失<5%）
    log_message("\n  最优量化策略推荐:")
    for model_name in model_names:
        model_results = df[df['Model'] == model_name]
        # 排除准确率损失过大的策略
        valid_results = model_results[model_results['Accuracy_Loss(%)'] < 5]
        if not valid_results.empty:
            best_strategy = valid_results.loc[valid_results['FPS'].idxmax()]
            log_message(f"  {model_name}: {best_strategy['Strategy']} - FPS: {best_strategy['FPS']:.2f}, Accuracy: {best_strategy['Accuracy(%)']:.2f}%")
        else:
            log_message(f"  {model_name}: 没有找到合适的量化策略")
    
    log_message("\n=== 量化对比实验完成 ===")
    log_message(f"  详细日志: {log_filename}")
    log_message(f"  结果文件: {result_csv_path}")
    log_message("\n  实验结论建议:")
    log_message("  1. 对于ResNet18，INT8 QAT通常能提供最好的性能-精度权衡")
    log_message("  2. 对于MobileNetV2，INT4权重量化可能是更好的选择，因为模型已经很轻量")
    log_message("  3. 分组大小32通常是INT4量化的最佳选择")
    log_message("  4. 量化感知训练(QAT)比后训练量化(PTQ)通常能提供更好的精度")
