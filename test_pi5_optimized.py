import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torchvision import datasets
import time
import numpy as np
import os
import logging
from datetime import datetime
import psutil
import argparse

# 环境配置
DATA_PATH = './data'
torch.set_num_threads(4)  # 树莓派4核CPU优化
torch.backends.quantized.engine = 'qnnpack'  # ARM架构优化

def setup_logger(log_filename):
    """设置日志记录器"""
    logging.basicConfig(level=logging.INFO, format='%(message)s',
                        handlers=[logging.FileHandler(log_filename), logging.StreamHandler()])
    return logging.getLogger()

def log_msg(logger, msg):
    """记录日志"""
    logger.info(msg)

def get_dataloader(dataset_name, input_size=224, batch_size=1):
    """
    根据模型名自动匹配数据集和归一化参数
    :param dataset_name: 模型文件名
    :param input_size: 输入图片尺寸（保持224x224）
    :param batch_size: 批次大小
    :return: 数据加载器
    """
    try:
        if "c100" in dataset_name.lower():
            # CIFAR-100 标准均值标准差
            norm = transforms.Normalize((0.5071, 0.4865, 0.4409), (0.2673, 0.2564, 0.2761))
            testset = datasets.CIFAR100(root=DATA_PATH, train=False, download=False,
                                        transform=transforms.Compose([
                                            transforms.Resize(input_size),
                                            transforms.ToTensor(),
                                            norm
                                        ]))
            log_msg(logger, f"  [数据读取] 匹配到 CIFAR-100 测试集，输入尺寸: {input_size}")
        else:
            # CIFAR-10 标准均值标准差
            norm = transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
            testset = datasets.CIFAR10(root=DATA_PATH, train=False, download=False,
                                       transform=transforms.Compose([
                                           transforms.Resize(input_size),
                                           transforms.ToTensor(),
                                           norm
                                       ]))
            log_msg(logger, f"  [数据读取] 匹配到 CIFAR-10 测试集，输入尺寸: {input_size}")
        
        # 使用0个worker以避免树莓派上的多进程问题
        return torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=0)
    except Exception as e:
        log_msg(logger, f"  [数据读取错误] {e}")
        return None

def load_model(model_path, device='cpu'):
    """
    加载模型
    :param model_path: 模型文件路径
    :param device: 运行设备
    :return: 加载好的模型
    """
    log_msg(logger, f"  [模型加载] 正在加载 {os.path.basename(model_path)}...")
    
    try:
        # 尝试加载TorchScript模型
        model = torch.jit.load(model_path, map_location=device).eval()
        log_msg(logger, f"  [模型加载] 成功加载 TorchScript 模型")
        return model
    except Exception as e:
        log_msg(logger, f"  [模型加载] TorchScript 加载失败，尝试直接加载 state_dict: {e}")
        
        # 根据模型名创建对应的模型结构
        model_name = os.path.basename(model_path).lower()
        
        if "resnet18" in model_name:
            from torchvision.models.quantization import resnet18
            model = resnet18(weights=None, quantize=False)
            num_classes = 100 if "c100" in model_name else 10
            model.fc = nn.Linear(model.fc.in_features, num_classes)
        elif "mobilenetv2" in model_name:
            from torchvision.models.quantization import mobilenet_v2
            model = mobilenet_v2(weights=None, quantize=False)
            num_classes = 100 if "c100" in model_name else 10
            model.classifier[1] = nn.Linear(model.last_channel, num_classes)
        else:
            log_msg(logger, f"  [模型加载] 不支持的模型类型: {model_name}")
            return None
        
        try:
            model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
            model.to(device)
            model.eval()
            log_msg(logger, f"  [模型加载] 成功加载 state_dict 模型")
            return model
        except Exception as e:
            log_msg(logger, f"  [模型加载] state_dict 加载失败: {e}")
            return None

def evaluate_accuracy(model, test_loader, device='cpu', max_samples=1000):
    """
    评估模型准确率
    :param model: 模型
    :param test_loader: 测试数据加载器
    :param device: 运行设备
    :param max_samples: 最大测试样本数（避免测试时间过长）
    :return: 准确率
    """
    if not test_loader:
        return "N/A"
    
    log_msg(logger, f"  [精度评估] 正在评估准确率（最多{max_samples}个样本）...")
    correct, total = 0, 0
    
    with torch.no_grad():
        for i, (imgs, labels) in enumerate(test_loader):
            if i >= max_samples:  # 限制测试样本数
                break
                
            imgs, labels = imgs.to(device), labels.to(device)
            outputs = model(imgs)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    
    acc = f"{100 * correct / total:.2f}%"
    log_msg(logger, f"  [精度评估] 准确率: {acc}")
    return acc

def evaluate_performance(model, input_size=224, device='cpu'):
    """
    评估模型性能（延迟、FPS、吞吐量）
    :param model: 模型
    :param input_size: 输入图片尺寸
    :param device: 运行设备
    :return: (平均延迟, FPS, 吞吐量)
    """
    # 1. 推理延迟与FPS测试 (单张图片)
    dummy_input = torch.randn(1, 3, input_size, input_size).to(device)
    
    log_msg(logger, f"  [性能测试] 正在测试单张图片推理性能...")
    
    with torch.no_grad():
        # 预热
        for _ in range(50):
            _ = model(dummy_input)
        
        # 测试
        latencies = []
        start_time = time.time()
        for _ in range(100):
            t0 = time.time()
            _ = model(dummy_input)
            latencies.append((time.time() - t0) * 1000)  # 转换为毫秒
        
        total_time = time.time() - start_time
        avg_latency = np.mean(latencies)
        fps = 100 / total_time  # 计算FPS
    
    log_msg(logger, f"  [性能测试] 平均延迟: {avg_latency:.2f} ms")
    log_msg(logger, f"  [性能测试] FPS: {fps:.2f}")
    
    # 2. 吞吐量测试 (批量处理)
    batch_size = 4
    batch_input = torch.randn(batch_size, 3, input_size, input_size).to(device)
    
    log_msg(logger, f"  [性能测试] 正在测试批量处理吞吐量 (batch_size={batch_size})...")
    
    with torch.no_grad():
        # 预热
        for _ in range(20):
            _ = model(batch_input)
        
        # 测试
        start_time = time.time()
        for _ in range(50):
            _ = model(batch_input)
        
        total_time = time.time() - start_time
        throughput = (50 * batch_size) / total_time
    
    log_msg(logger, f"  [性能测试] 吞吐量: {throughput:.2f} 张/秒")
    
    return avg_latency, fps, throughput

def evaluate_memory_usage(model, input_size=224, device='cpu'):
    """
    评估模型内存占用
    :param model: 模型
    :param input_size: 输入图片尺寸
    :param device: 运行设备
    :return: 内存占用 (MB)
    """
    process = psutil.Process()
    
    # 测试前内存占用
    initial_memory = process.memory_info().rss / 1024 / 1024  # MB
    
    # 运行模型
    dummy_input = torch.randn(1, 3, input_size, input_size).to(device)
    
    with torch.no_grad():
        for _ in range(10):
            _ = model(dummy_input)
    
    # 测试后内存占用
    final_memory = process.memory_info().rss / 1024 / 1024  # MB
    
    mem_usage = final_memory - initial_memory
    log_msg(logger, f"  [内存占用] 增加: {mem_usage:.2f} MB")
    
    return mem_usage

def evaluate_model(model_path, input_size=224, device='cpu', is_detection=False):
    """
    完整评估模型性能
    :param model_path: 模型文件路径
    :param input_size: 输入图片尺寸（保持224x224）
    :param device: 运行设备
    :param is_detection: 是否为目标检测模型
    :return: 评估结果列表
    """
    model_name = os.path.basename(model_path)
    log_msg(logger, f"\n>>> 正在评估模型: {model_name}")
    
    # 识别精度类型
    if "int4" in model_name.lower():
        dtype = "INT4"
    elif "int8" in model_name.lower():
        dtype = "INT8"
    else:
        dtype = "FP32"
    
    # 计算模型大小
    model_size = os.path.getsize(model_path) / (1024 * 1024)  # MB
    log_msg(logger, f"  [模型信息] 精度类型: {dtype}, 大小: {model_size:.2f} MB")
    
    # 加载模型
    model = load_model(model_path, device)
    if not model:
        return [model_name, dtype, f"{model_size:.2f}", "N/A", "N/A", "N/A", "N/A", "N/A", "N/A"]
    
    # 评估性能
    avg_latency, fps, throughput = evaluate_performance(model, input_size, device)
    
    # 评估内存占用
    mem_usage = evaluate_memory_usage(model, input_size, device)
    
    # 评估准确率（仅分类模型）
    acc = "N/A"
    if not is_detection:
        test_loader = get_dataloader(model_name, input_size, batch_size=1)
        acc = evaluate_accuracy(model, test_loader, device, max_samples=1000)
    
    # 实时性判别
    realtime = "YES" if fps >= 24 else "NO"
    
    return [model_name, dtype, f"{model_size:.2f}", f"{avg_latency:.2f}", f"{fps:.2f}", f"{throughput:.2f}", f"{mem_usage:.2f}", acc, realtime]

if __name__ == "__main__":
    # 解析命令行参数
    parser = argparse.ArgumentParser(description="树莓派模型性能测试脚本")
    parser.add_argument('--model', type=str, help='指定单个模型文件路径')
    parser.add_argument('--input-size', type=int, default=224, help='输入图片尺寸（默认224x224）')
    args = parser.parse_args()
    
    # 设置日志
    log_filename = f"Pi5_Experiment_Optimized_{datetime.now().strftime('%Y%m%d_%H%M')}.log"
    logger = setup_logger(log_filename)
    
    # 设备信息
    device = torch.device("cpu")  # 树莓派通常使用CPU
    log_msg(logger, f"[测试环境] 设备: {device}, PyTorch版本: {torch.__version__}")
    log_msg(logger, f"[测试环境] 树莓派CPU线程数: 4")
    log_msg(logger, f"[测试环境] 输入图片尺寸: {args.input_size}x{args.input_size}")
    log_msg(logger, f"[测试环境] 日志文件: {log_filename}")
    
    # 模型列表
    if args.model:
        # 测试指定的单个模型
        models_to_test = [args.model]
    else:
        # 测试所有可用的ResNet18模型
        models_to_test = [
            # ResNet18模型
            "resnet18_c10_fp32_deploy.pt",
            "resnet18_c10_int8_deploy.pt",
            "resnet18_c10_int4_deploy.pt",
            "resnet18_c100_fp32_deploy.pt",
            "resnet18_c100_int8_deploy.pt",
            "resnet18_c100_int4_deploy.pt",
            # 优化后的模型
            "resnet18_int8_optimized_c10_scripted.pt",
            "resnet18_int4_weight_only_c10_scripted.pt",
            # MobileNetV2模型（作为对比）
            "mobilenetv2_c10_fp32_deploy.pt",
            "mobilenetv2_c10_int8_deploy.pt",
            "mobilenetv2_c10_int4_deploy.pt",
        ]
    
    # 开始测试
    log_msg(logger, "="*120)
    log_msg(logger, f"{'Model Name':<40} | {'Type':<6} | {'Size(MB)':<10} | {'Lat(ms)':<10} | {'FPS':<8} | {'Throughput':<15} | {'Mem(MB)':<10} | {'Accuracy':<12} | {'RealTime'}")
    log_msg(logger, "="*120)
    
    results = []
    
    for model_path in models_to_test:
        if os.path.exists(model_path):
            # 评估模型
            result = evaluate_model(model_path, input_size=args.input_size, device=device)
            results.append(result)
            
            # 打印结果
            log_msg(logger, f"{result[0]:<40} | {result[1]:<6} | {result[2]:<10} | {result[3]:<10} | {result[4]:<8} | {result[5]:<15} | {result[6]:<10} | {result[7]:<12} | {result[8]}")
            
            # 散热保护
            log_msg(logger, "  >> 散热保护：等待30秒进行下一次测试...")
            time.sleep(30)
        else:
            log_msg(logger, f"{model_path:<40} | {'N/A':<6} | {'N/A':<10} | {'N/A':<10} | {'N/A':<8} | {'N/A':<15} | {'N/A':<10} | {'N/A':<12} | {'N/A'}")
            log_msg(logger, f"  >> 模型文件不存在: {model_path}")
    
    # 生成总结
    log_msg(logger, "="*120)
    log_msg(logger, "\n=== 测试总结 ===")
    
    # 统计符合实时性要求的模型
    realtime_models = [r for r in results if r[8] == "YES"]
    log_msg(logger, f"符合实时性要求(≥24 FPS)的模型数量: {len(realtime_models)}/{len(results)}")
    
    if realtime_models:
        log_msg(logger, "\n符合实时性要求的模型:")
        for r in realtime_models:
            log_msg(logger, f"  - {r[0]}: {r[4]} FPS, {r[7]} 准确率")
    
    # 找出性能最优的模型
    if results:
        best_fps_model = max(results, key=lambda x: float(x[4]) if x[4] != "N/A" else 0)
        best_acc_model = max(results, key=lambda x: float(x[7].replace('%', '')) if x[7] != "N/A" else 0)
        
        log_msg(logger, f"\n性能最优模型(FPS): {best_fps_model[0]} - {best_fps_model[4]} FPS")
        log_msg(logger, f"准确率最高模型: {best_acc_model[0]} - {best_acc_model[7]} 准确率")
    
    log_msg(logger, f"\n测试完成，日志已保存至: {log_filename}")
    log_msg(logger, "\n测试指标说明:")
    log_msg(logger, "- Size(MB): 模型文件大小")
    log_msg(logger, "- Lat(ms): 单张图片推理延迟")
    log_msg(logger, "- FPS: 每秒处理帧数")
    log_msg(logger, "- Throughput: 批量处理吞吐量 (batch_size=4)")
    log_msg(logger, "- Mem(MB): 内存占用增加量")
    log_msg(logger, "- Accuracy: 分类准确率")
    log_msg(logger, "- RealTime: 是否达到实时要求(≥24 FPS)")
