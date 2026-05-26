import torch
import torch.nn as nn
from torchvision.models.quantization import mobilenet_v2
import os
import time
import datetime

# --- 1. 配置 ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_dir = "models"
export_dir = "deploy_models"
os.makedirs(export_dir, exist_ok=True)

# --- 2. 导出分类模型 --- 
def export_classification_model(model_name, dataset, quant_type):
    """
    导出分类模型为TorchScript格式
    :param model_name: 模型名称 (mobilenetv2, resnet18, vgg16)
    :param dataset: 数据集 (c10, c100)
    :param quant_type: 量化类型 (fp32, int8, int4)
    """
    print(f"\n=== 导出 {model_name}_{dataset}_{quant_type} 模型 ===")
    
    # 加载模型结构
    if model_name == "mobilenetv2":
        model = mobilenet_v2(weights=None, quantize=False)
        num_classes = 100 if dataset == "c100" else 10
        model.classifier[1] = nn.Linear(model.last_channel, num_classes)
    else:
        print(f"暂不支持 {model_name} 模型导出")
        return
    
    # 加载模型权重
    if quant_type == "fp32":
        weight_path = os.path.join(model_dir, f"fp32_{model_name}_{dataset}_best.pth")
    elif quant_type == "int8":
        weight_path = os.path.join(model_dir, f"{model_name}_int8_{dataset}_best.pth")
    elif quant_type == "int4":
        weight_path = os.path.join(model_dir, f"{model_name}_int4_weight_only_{dataset}.pth")
    else:
        print(f"不支持的量化类型: {quant_type}")
        return
    
    if not os.path.exists(weight_path):
        print(f"找不到权重文件: {weight_path}")
        return
    
    # 加载权重
    model.load_state_dict(torch.load(weight_path, map_location='cpu', weights_only=True))
    model.to(device)
    model.eval()
    
    # 创建示例输入
    input_size = 224
    dummy_input = torch.randn(1, 3, input_size, input_size).to(device)
    
    # 导出为TorchScript
    export_path = os.path.join(export_dir, f"{model_name}_{dataset}_{quant_type}_deploy.pt")
    
    # 对于量化模型，需要特别处理
    if quant_type in ["int8", "int4"]:
        # 量化模型通常在CPU上运行
        model.to('cpu')
        dummy_input = dummy_input.to('cpu')
    
    # 跟踪模型
    traced_model = torch.jit.trace(model, dummy_input)
    
    # 保存模型
    traced_model.save(export_path)
    print(f"模型已导出到: {export_path}")
    print(f"模型大小: {os.path.getsize(export_path) / (1024 * 1024):.2f} MB")

# --- 3. 导出目标检测模型 --- 
def export_detection_model(quant_type):
    """
    导出目标检测模型为TorchScript格式
    :param quant_type: 量化类型 (fp32, int8, int4)
    """
    print(f"\n=== 导出 simple_yolo_{quant_type} 模型 ===")
    
    # 加载模型结构
    from object_detection import SimpleYOLO
    model = SimpleYOLO(num_classes=10)
    
    # 加载模型权重
    if quant_type == "fp32":
        weight_path = os.path.join(model_dir, "simple_yolo_best.pth")
    elif quant_type == "int8":
        weight_path = os.path.join(model_dir, "simple_yolo_int8.pth")
    elif quant_type == "int4":
        weight_path = os.path.join(model_dir, "simple_yolo_int4.pth")
    else:
        print(f"不支持的量化类型: {quant_type}")
        return
    
    if not os.path.exists(weight_path):
        print(f"找不到权重文件: {weight_path}")
        return
    
    # 加载权重
    model.load_state_dict(torch.load(weight_path, map_location='cpu', weights_only=True))
    model.to(device)
    model.eval()
    
    # 创建示例输入
    input_size = 640
    dummy_input = torch.randn(1, 3, input_size, input_size).to(device)
    
    # 导出为TorchScript
    export_path = os.path.join(export_dir, f"simple_yolo_{quant_type}_deploy.pt")
    
    # 对于量化模型，需要特别处理
    if quant_type in ["int8", "int4"]:
        # 量化模型通常在CPU上运行
        model.to('cpu')
        dummy_input = dummy_input.to('cpu')
    
    # 跟踪模型
    traced_model = torch.jit.trace(model, dummy_input)
    
    # 保存模型
    traced_model.save(export_path)
    print(f"模型已导出到: {export_path}")
    print(f"模型大小: {os.path.getsize(export_path) / (1024 * 1024):.2f} MB")

# --- 4. 主程序 --- 
if __name__ == "__main__":
    print("=== 模型导出工具 ===")
    print(f"当前时间: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"设备: {device}")
    
    # 导出分类模型
    # 注意：需要先运行相应的训练/量化脚本生成权重文件
    # export_classification_model("mobilenetv2", "c10", "fp32")
    # export_classification_model("mobilenetv2", "c10", "int8")
    # export_classification_model("mobilenetv2", "c10", "int4")
    # export_classification_model("mobilenetv2", "c100", "fp32")
    # export_classification_model("mobilenetv2", "c100", "int8")
    # export_classification_model("mobilenetv2", "c100", "int4")
    
    # 导出目标检测模型
    # export_detection_model("fp32")
    # export_detection_model("int8")
    # export_detection_model("int4")
    
    print("\n=== 导出完成 ===")
    print("请根据需要取消注释相应的导出命令")
    print("导出的模型将保存在 deploy_models 目录中")
