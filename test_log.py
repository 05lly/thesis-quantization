import os
import datetime

# 测试日志功能
log_dir = 'logs'
os.makedirs(log_dir, exist_ok=True)

# 创建日志文件名
log_filename = os.path.join(log_dir, f'qat_resnet18_optimized_test_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.log')

# 定义日志函数
def log_message(msg):
    t = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    full_msg = f'[{t}] {msg}'
    print(full_msg)
    with open(log_filename, 'a', encoding='utf-8') as f:
        f.write(full_msg + '\n')

# 测试日志写入
log_message('Test log message 1')
log_message('Test log message 2')

# 验证日志文件是否创建并包含内容
if os.path.exists(log_filename):
    print(f'\nLog file created successfully: {log_filename}')
    with open(log_filename, 'r', encoding='utf-8') as f:
        content = f.read()
        print('Log content:')
        print(content)
else:
    print('\nERROR: Log file was not created!')
