import pickle
import numpy as np
from PIL import Image
import cv2
import matplotlib.pyplot as plt

# 加载上传的图片
image_path = "F:\\Gitee\\engineering-practice-and-innovation-project-ii\\Mission2\\BP\\img\\tt5.jpg"
image = cv2.imread(image_path)

# 将图片转换为灰度图
gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# 反转黑白颜色
inverted_image = cv2.bitwise_not(gray_image)

# 将图像展平为一维数组
processed_image = inverted_image.flatten()

# 归一化图像数据以匹配训练数据的范围
processed_image = processed_image / 16.0  # 训练数据的范围是 0 到 16


# 展示反转后的图片
plt.imshow(inverted_image, cmap="gray")
plt.show()

# 载入训练好的模型
model_path = "F:\\Gitee\\engineering-practice-and-innovation-project-ii\\Mission2\\BP\\models\\trained_model.m"
with open(model_path, "rb") as f:
    clf = pickle.load(f)

# 使用加载的模型进行预测
prediction = clf.predict(processed_image)
predicted_digit = np.argmax(prediction)

print("预测结果: ", predicted_digit)
