import cv2

# 加载预训练的 Haar Cascade 分类器模型
face_cascade = cv2.CascadeClassifier(
    "F:/PythonProgram/cv-face/haarcascade_frontalface_default.xml"
)

# 打开摄像头
cap = cv2.VideoCapture(1)

# 循环读取摄像头的每一帧
while True:
    # 读取一帧图像
    ret, frame = cap.read()

    # 如果读取成功
    if ret:
        # 将图像转换为灰度图像
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # 使用分类器检测人脸，返回人脸区域的坐标
        faces = face_cascade.detectMultiScale(
            gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30)
        )

        # 在检测到的人脸区域绘制矩形框
        for x, y, w, h in faces:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)

        # 显示带有检测结果的图像
        cv2.imshow("Face Detection", frame)

    # 按 'q' 键退出程序
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# 释放摄像头资源并关闭所有窗口
cap.release()
cv2.destroyAllWindows()
