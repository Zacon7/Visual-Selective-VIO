import matplotlib.pyplot as plt
from pathlib import Path
import re

# 从日志文件中读取内容
log_file_path = 'results/train/fastflow_hard_dwconv128/logs/fastflow_hard_dwconv128.txt'
with open(log_file_path, 'r') as file:
    log_content = file.read()

# 使用正则表达式匹配 total loss 的数值
total_loss_pattern = re.compile(r'total loss: ([0-9.]+)')
matches = total_loss_pattern.findall(log_content)

# 将匹配到的数值转换为浮点数列表
loss_list = [float(match) for match in matches]

# 打印提取的 total loss 数值
# print(loss_list)


# 假设你有一个名为 'loss_list' 的列表，包含 N*100 条 loss 数据
# loss_list = [your_loss_data_here]

# 计算 epoch 数量和 batch 数量
num_epochs = 100
num_batches = len(loss_list) // num_epochs

# 将 loss 数据按照每个 epoch 分割
loss_by_epoch = [loss_list[i * num_batches:(i + 1) * num_batches] for i in range(num_epochs)]

# 计算每个 epoch 的平均 loss
average_loss_by_epoch = [sum(epoch_loss) / num_batches for epoch_loss in loss_by_epoch]

# 绘制 loss 曲线图
plt.plot(range(1, num_epochs + 1), average_loss_by_epoch, marker='o')
# plt.plot(range(1, len(loss_list) + 1), loss_list)
plt.xlabel('Epoch')
plt.ylabel('Average Loss')
plt.title('Training Loss Over Epochs')
plt.gcf().canvas.set_window_title(Path(log_file_path).name)
plt.grid(True)
plt.show()
