import cv2
import numpy as np
import tkinter as tk
from tkinter import filedialog
import os


# 定义处理图片的函数
def process_image(image_path):
    image = cv2.imread(image_path)

    if image is None:
        print("无法加载图片")
        return

    scale_percent = 12  # 图片缩小比例
    width = int(image.shape[1] * scale_percent / 100)
    height = int(image.shape[0] * scale_percent / 100)
    resized_image = cv2.resize(image, (width, height))

    # 分离RGB通道
    B, G, R = cv2.split(resized_image)

    # 显示原始图片
    cv2.imshow("Original Image", resized_image)

    # 显示各个通道
    cv2.imshow("Red Channel", R)
    cv2.imshow("Green Channel", G)
    cv2.imshow("Blue Channel", B)

    # 等待用户按键关闭窗口
    cv2.waitKey(0)
    cv2.destroyAllWindows()


# 定义处理视频的函数
def process_video(video_path):
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print("无法打开视频")
        return

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # 缩小视频帧
        scale_percent = 30  # 缩小比例可以根据需要调整
        width = int(frame.shape[1] * scale_percent / 100)
        height = int(frame.shape[0] * scale_percent / 100)
        resized_frame = cv2.resize(frame, (width, height))

        # 分离RGB通道
        B, G, R = cv2.split(resized_frame)

        # 显示原始视频帧和通道
        cv2.imshow("Original Video Frame", resized_frame)
        cv2.imshow("Red Channel", R)
        cv2.imshow("Green Channel", G)
        cv2.imshow("Blue Channel", B)

        # 按下'q'键退出播放
        if cv2.waitKey(30) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


# 定义使用摄像头的处理函数
def process_camera():
    cap = cv2.VideoCapture(1)  # 1表示USB摄像头

    if not cap.isOpened():
        print("无法打开摄像头")
        return

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # 缩小摄像头帧
        scale_percent = 50  # 缩小比例可以根据需要调整
        width = int(frame.shape[1] * scale_percent / 100)
        height = int(frame.shape[0] * scale_percent / 100)
        resized_frame = cv2.resize(frame, (width, height))

        # 分离RGB通道
        B, G, R = cv2.split(resized_frame)

        # 显示原始摄像头帧和通道
        cv2.imshow("Camera Frame", resized_frame)
        cv2.imshow("Red Channel", R)
        cv2.imshow("Green Channel", G)
        cv2.imshow("Blue Channel", B)

        # 按下'q'键退出播放
        if cv2.waitKey(30) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


# 定义选择文件并处理的函数
def select_file():
    file_path = filedialog.askopenfilename()
    if file_path:
        ext = os.path.splitext(file_path)[1].lower()
        if ext in [".jpg", ".jpeg", ".png"]:
            process_image(file_path)
        elif ext in [".mp4", ".avi", ".mov"]:
            process_video(file_path)
        else:
            print("不支持的文件格式")


# 创建一个简单的GUI
root = tk.Tk()
root.title("文件选择器")

# 创建一个按钮来选择图片或视频文件
button_file = tk.Button(root, text="选择图片或视频文件", command=select_file)
button_file.pack(pady=10)

# 创建一个按钮来使用摄像头
button_camera = tk.Button(root, text="使用摄像头", command=process_camera)
button_camera.pack(pady=10)

# 启动GUI
root.mainloop()
