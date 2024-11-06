import cv2
import numpy as np

# 读取图片
image = cv2.imread("C:/Users/27310/Pictures/DSC_8817.JPG")

# 检查图片是否成功加载
if image is None:
    print("无法加载图片")
else:

    scale_percent = 12  # 图片大小缩小成12%
    width = int(image.shape[1] * scale_percent / 100)
    height = int(image.shape[0] * scale_percent / 100)
    resized_image = cv2.resize(image, (width, height))

    # 使用cv2.split()分离RGB通道
    B, G, R = cv2.split(resized_image)

    # 显示原始图片
    cv2.imshow("Original Image", resized_image)

    # 显示分离的各个通道
    cv2.imshow("Red Channel", R)
    cv2.imshow("Green Channel", G)
    cv2.imshow("Blue Channel", B)

    # 等待用户按下任意键退出
    cv2.waitKey(0)

# 关闭所有窗口
cv2.destroyAllWindows()
