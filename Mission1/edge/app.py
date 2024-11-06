import cv2
from flask import Flask, render_template, Response

app = Flask(__name__)

# 获取摄像头视频流
camera = cv2.VideoCapture(1)  # 1 代表usb摄像头


def generate_frames():
    while True:
        # 读取摄像头帧
        success, frame = camera.read()
        if not success:
            break
        else:
            # 将帧转换为灰度图像
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            # 应用 Canny 边缘检测
            edges = cv2.Canny(gray, 100, 150)

            # 将边缘检测结果编码为JPEG格式
            ret, buffer = cv2.imencode(".jpg", edges)
            frame = buffer.tobytes()

            # 使用生成器返回处理后的帧
            yield (b"--frame\r\n" b"Content-Type: image/jpeg\r\n\r\n" + frame + b"\r\n")


@app.route("/")
def index():
    # 返回HTML页面
    return render_template("index.html")


@app.route("/video_feed")
def video_feed():
    # 视频流路由
    return Response(
        generate_frames(), mimetype="multipart/x-mixed-replace; boundary=frame"
    )


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
