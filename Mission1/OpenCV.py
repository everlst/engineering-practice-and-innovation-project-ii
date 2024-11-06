"""
import cv2

# 打开摄像头，参数1表示使用usb摄像头
cap = cv2.VideoCapture(1)


# 循环读取摄像头的每一帧
while True:
    # 读取一帧图像，返回值 ret 表示是否成功
    ret, frame = cap.read()

    # 如果读取成功
    if ret:
        # 显示图像
        cv2.imshow("Camera", frame)
    else:
        print("无法获取帧")
        break

    # 等待用户按下 'q' 键退出
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# 释放摄像头资源并关闭窗口
cap.release()
cv2.destroyAllWindows()
"""

import cv2

# 读取图片，路径为本地图片路径
image = cv2.imread("C:/Users/27310/Pictures/Screenshots/test 2024-06-03 171807.png")

# 检查图片是否成功加载
if image is None:
    print("无法加载图片")
else:
    # 显示图片
    cv2.imshow("Image", image)

    # 等待用户按下任意键退出
    cv2.waitKey(0)

# 关闭所有窗口
cv2.destroyAllWindows()
