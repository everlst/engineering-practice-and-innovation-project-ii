from flask import Flask, render_template, Response
import cv2

app = Flask(__name__)

# 加载预训练的Haar Cascade分类器模型
face_cascade = cv2.CascadeClassifier(
    "F:/PythonProgram/cv-face/haarcascade_frontalface_default.xml"
)

# 打开摄像头
cap = cv2.VideoCapture(0)


def generate_frames():
    while True:
        # 从摄像头读取一帧
        success, frame = cap.read()
        if not success:
            break
        else:
            # 将图像转换为灰度图像
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            # 检测人脸
            faces = face_cascade.detectMultiScale(gray, 1.1, 4)

            # 在检测到的人脸区域绘制矩形框
            for x, y, w, h in faces:
                cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)

            # 将图像编码为JPEG格式
            ret, buffer = cv2.imencode(".jpg", frame)
            frame = buffer.tobytes()

            # 使用流的方式返回视频帧
            yield (b"--frame\r\n" b"Content-Type: image/jpeg\r\n\r\n" + frame + b"\r\n")


@app.route("/")
def index():
    # 渲染主页面
    return render_template("index.html")


@app.route("/video_feed")
def video_feed():
    # 返回视频流响应
    return Response(
        generate_frames(), mimetype="multipart/x-mixed-replace; boundary=frame"
    )


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
