import face_recognition
import cv2
import os

# 定义已知人脸的目录路径
KNOWN_FACES_DIR = "known_faces"  # 本地存储已知人脸的目录
TOLERANCE = 0.6  # 人脸识别相似度阈值
MODEL = "hog"  # 可选: "hog"（CPU）或 "cnn"（更准确但需要GPU）

# 加载所有已知人脸的图像并生成编码
known_face_encodings = []
known_face_names = []

print("正在加载已知人脸...")

# 遍历存储已知人脸的目录
for filename in os.listdir(KNOWN_FACES_DIR):
    # 加载人脸图像
    image_path = os.path.join(KNOWN_FACES_DIR, filename)
    image = face_recognition.load_image_file(image_path)

    # 获取人脸编码（提取人脸特征）
    encoding = face_recognition.face_encodings(image)[0]
    known_face_encodings.append(encoding)

    # 使用文件名（不带扩展名）作为人脸的名称
    name = os.path.splitext(filename)[0]
    known_face_names.append(name)

print(f"已加载 {len(known_face_encodings)} 张已知人脸")

# 打开摄像头
video_capture = cv2.VideoCapture(0)

print("正在打开摄像头...")

while True:
    # 捕获摄像头中的一帧
    ret, frame = video_capture.read()

    # 将 BGR 转换为 RGB
    rgb_frame = frame[:, :, ::-1]

    # 查找图像中的所有人脸位置和人脸编码
    face_locations = face_recognition.face_locations(rgb_frame, model=MODEL)
    face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)

    # 遍历检测到的每张人脸
    for face_encoding, face_location in zip(face_encodings, face_locations):
        # 检查当前人脸是否匹配已知人脸
        matches = face_recognition.compare_faces(
            known_face_encodings, face_encoding, TOLERANCE
        )
        name = "Unknown"

        # 如果有匹配，使用匹配的已知人脸名称
        if True in matches:
            match_index = matches.index(True)
            name = known_face_names[match_index]

        # 获取人脸位置 (top, right, bottom, left)
        top, right, bottom, left = face_location

        # 在检测到的人脸上绘制矩形框
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)

        # 在检测到的人脸下方显示人名
        cv2.rectangle(
            frame, (left, bottom - 35), (right, bottom), (0, 255, 0), cv2.FILLED
        )
        font = cv2.FONT_HERSHEY_DUPLEX
        cv2.putText(frame, name, (left + 6, bottom - 6), font, 1.0, (255, 255, 255), 1)

    # 显示摄像头捕捉的图像
    cv2.imshow("Face Recognition", frame)

    # 按下 'q' 键退出程序
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# 释放摄像头资源并关闭窗口
video_capture.release()
cv2.destroyAllWindows()
