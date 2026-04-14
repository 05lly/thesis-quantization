import torch
import torch.ao.quantization
import os


def final_export():
    # 1. 准备环境
    torch.backends.quantized.engine = 'qnnpack'
    model_dir = "/root/autodl-tmp/my_backup"
    
    # 2. 建立空壳模型
    model = QuantizableVGG16(num_classes=10)
    
    # 3. 必须先设为 train
    model.train() 
    model.fuse_model()
    model.qconfig = torch.ao.quantization.get_default_qat_qconfig('qnnpack')
    torch.ao.quantization.prepare_qat(model, inplace=True)
    
    # 4. 加载训练好的权重
    checkpoint_path = os.path.join(model_dir, "vgg16_qat_best_weights.pth")
    model.load_state_dict(torch.load(checkpoint_path, map_location='cpu'))
    
    # 5. 切换到 eval，
    model.eval()
    model.to('cpu')
    
    # 6. 执行真正的转换
    print("正在将模拟量化转换为真实 INT8 算子...")
    int8_model = torch.ao.quantization.convert(model, inplace=False)
    
    # 7. 保存为树莓派专用的部署文件
    example_input = torch.randn(1, 3, 224, 224)
    traced_model = torch.jit.trace(int8_model, example_input)
    save_path = os.path.join(model_dir, "vgg16_cifar10_int8_deploy_final.pt")
    torch.jit.save(traced_model, save_path)
    
    print(f"真正的 INT8 模型已生成：{save_path}")

if __name__ == "__main__":
    final_export()