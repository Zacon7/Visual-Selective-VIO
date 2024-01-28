import matplotlib.pyplot as plt
from pathlib import Path
import re

# 从日志文件中读取内容
log_file_path = 'results/train/fastflow_hard_flow6/logs/fastflow_hard_flow6.txt'
with open(log_file_path, 'r') as file:
    log_content = file.read()

# 使用正则表达式匹配 total loss 的数值
total_loss_pattern = re.compile(r'total loss: ([0-9.]+)')
matches = total_loss_pattern.findall(log_content)

# 将匹配到的数值转换为浮点数列表
loss_list = [float(match) for match in matches]

# 计算 epoch 数量和 batch 数量
num_epochs = 100
num_batches = len(loss_list) // num_epochs

# 计算每个 epoch 的平均 loss
average_loss_by_epoch = [sum(loss_list[i * num_batches:(i + 1) * num_batches]) / num_batches for i in range(num_epochs)]

# 绘制 loss 曲线图
plt.figure(figsize=(10, 6))

# 绘制每个 batch 的淡化 loss 曲线
plt.plot(range(1, len(loss_list) + 1), loss_list, color='gray', linewidth=0.7, label='Batch Loss')

# 绘制每个 epoch 的平均 loss
epoch_indices = [i * num_batches + num_batches // 2 for i in range(num_epochs)]
plt.plot(epoch_indices, average_loss_by_epoch, marker='.', markersize=10, linestyle='-', color='red',
         label='Epoch Average Loss')

# 在指定 epoch 处添加文本标注
for epoch in [0, 40, 80, 99]:  # 使用99代替100，确保不越界
    if 0 <= epoch < num_epochs:  # 添加检查以确保索引在有效范围内
        plt.text(epoch * num_batches + num_batches // 2, average_loss_by_epoch[epoch], f'Epoch {epoch}', fontsize=10,
                 color='blue', ha='center', va='bottom', bbox=dict(facecolor='white', alpha=0.7))

plt.xlabel('Batch', fontsize=14)
plt.ylabel('Batch Loss', fontsize=14)
plt.title('Training Loss Over Batches and Epochs', fontsize=16)
plt.legend()
plt.grid(True, linestyle='--', alpha=0.7)
plt.gcf().canvas.manager.set_window_title(Path(log_file_path).name)
plt.show()
