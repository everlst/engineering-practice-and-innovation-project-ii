import numpy as np
import matplotlib.pyplot as plt

# 创建从 0 到 2π 的 x 值
x = np.linspace(0, 2 * np.pi, 100)  # 生成从 0 到 2π 之间的 100 个点

# 计算正弦值
y = np.sin(x)

# 绘制正弦曲线
plt.plot(x, y)

# 添加标题和标签
plt.title("Sine Wave")
plt.xlabel("x values")
plt.ylabel("sin(x)")

# 显示网格
plt.grid(True)

# 显示图像
plt.show()
