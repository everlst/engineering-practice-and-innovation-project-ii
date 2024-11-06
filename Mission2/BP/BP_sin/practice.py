import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

# 检查是否有可用的GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Step 1: 生成正弦曲线数据
x = np.linspace(0, 2 * np.pi, 1000)  # 从0到2π之间生成1000个点
y = np.sin(x)  # 目标输出是正弦函数的值


# Step 2: 构建BP神经网络
class SimpleNet(nn.Module):
    def __init__(self):
        super(SimpleNet, self).__init__()
        self.fc1 = nn.Linear(1, 640)  # 输入层到隐藏层，320个神经元
        self.fc2 = nn.Linear(640, 640)  # 隐藏层到隐藏层
        self.fc3 = nn.Linear(640, 1)  # 隐藏层到输出层

    def forward(self, x):
        x = torch.relu(self.fc1(x))  # 使用ReLU激活函数
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)  # 最后一层是线性输出
        return x


# Step 3: 数据转换并移动到GPU
x_train = torch.tensor(x, dtype=torch.float32).view(-1, 1).to(device)
y_train = torch.tensor(y, dtype=torch.float32).view(-1, 1).to(device)

# Step 4: 实例化网络和定义损失函数及优化器
net = SimpleNet().to(device)  # 将模型移动到GPU
criterion = nn.MSELoss()  # 使用均方误差作为损失函数
optimizer = optim.Adam(net.parameters(), lr=0.001)  # 使用Adam优化器

# Step 5: 训练模型
epochs = 5000  # 设置训练轮数
for epoch in range(epochs):
    net.train()
    optimizer.zero_grad()  # 梯度清零
    output = net(x_train)  # 前向传播
    loss = criterion(output, y_train)  # 计算损失
    loss.backward()  # 反向传播
    optimizer.step()  # 更新权重

    if epoch % 100 == 0:  # 每100个epoch打印一次loss
        print(f"Epoch {epoch}/{epochs}, Loss: {loss.item()}")

# Step 6: 预测并绘制拟合效果
net.eval()  # 预测时设为评估模式
y_pred = net(x_train).detach().cpu().numpy()  # 预测结果并转换为numpy数组，并移回CPU

# 绘制原始正弦曲线和拟合曲线
plt.plot(x, y, label="True Sine")
plt.plot(x, y_pred, label="Fitted Curve", linestyle="dashed")
plt.legend()
plt.title("BP Neural Network Fitting Sine Curve (GPU)")
plt.show()
