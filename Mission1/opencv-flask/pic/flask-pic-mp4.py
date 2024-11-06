from flask import (
    Flask,
    render_template,
    request,
    redirect,
    url_for,
    send_from_directory,
)
import cv2
import os
from werkzeug.utils import secure_filename

app = Flask(__name__)

# 获取当前脚本文件的目录
script_dir = os.path.dirname(os.path.abspath(__file__))

# 基于脚本目录设置上传文件和处理文件的文件夹路径
UPLOAD_FOLDER = os.path.join(script_dir, "uploads/")
PROCESSED_FOLDER = os.path.join(script_dir, "processed/")
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
app.config["PROCESSED_FOLDER"] = PROCESSED_FOLDER

# 打印绝对路径以确认
print(f"Upload folder absolute path: {UPLOAD_FOLDER}")
print(f"Processed folder absolute path: {PROCESSED_FOLDER}")

# 允许上传的文件扩展名
ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg", "mp4"}


# 判断文件扩展名是否符合要求
def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


# 首页：展示上传页面
@app.route("/")
def index():
    return render_template("index.html")


# 处理上传的文件（图片或视频）
@app.route("/upload", methods=["POST"])
def upload_file():
    if "file" not in request.files:
        return redirect(request.url)

    file = request.files["file"]

    if file.filename == "":
        return redirect(request.url)

    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        file_path = os.path.join(app.config["UPLOAD_FOLDER"], filename)

        # 保存上传的文件
        file.save(file_path)

        # 根据文件类型处理
        if filename.rsplit(".", 1)[1].lower() in {"png", "jpg", "jpeg"}:
            # 处理图片并检测人脸
            processed_image_path = detect_faces(file_path, filename)
            image_url = url_for(
                "display_image", filename=os.path.basename(processed_image_path)
            )
            return render_template("index.html", image_url=image_url)

        elif filename.rsplit(".", 1)[1].lower() == "mp4":
            # 处理视频并检测人脸
            processed_video_path = process_video(file_path, filename)
            video_url = url_for(
                "display_video", filename=os.path.basename(processed_video_path)
            )
            return render_template("index.html", video_url=video_url)

    return redirect(request.url)


# 显示处理后的图片
@app.route("/uploads/<filename>")
def display_image(filename):
    return send_from_directory(app.config["PROCESSED_FOLDER"], filename)


# 显示处理后的视频
@app.route("/processed/<filename>")
def display_video(filename):
    # 手动设置 MIME 类型为 video/mp4，确保浏览器正确识别
    return send_from_directory(
        app.config["PROCESSED_FOLDER"], filename, mimetype="video/mp4"
    )


# 人脸检测函数（用于图片）
def detect_faces(file_path, filename):
    # 加载图片
    image = cv2.imread(file_path)

    # 加载Haar Cascade分类器
    face_cascade = cv2.CascadeClassifier(
        cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    )

    # 转换为灰度图
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # 检测人脸
    faces = face_cascade.detectMultiScale(gray, 1.1, 4)

    # 在检测到的人脸上绘制矩形框
    for x, y, w, h in faces:
        cv2.rectangle(image, (x, y), (x + w, y + h), (255, 0, 0), 2)

    # 保存处理后的图片
    processed_image_path = os.path.join(
        app.config["PROCESSED_FOLDER"], "processed_" + filename
    )
    cv2.imwrite(processed_image_path, image)

    return processed_image_path


# 人脸检测函数（用于视频）
def process_video(file_path, filename):
    # 加载Haar Cascade分类器
    face_cascade = cv2.CascadeClassifier(
        cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    )

    # 打开视频文件
    cap = cv2.VideoCapture(file_path)

    # 获取视频的属性（如宽度、高度、帧率）
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)

    # 定义保存处理后视频的路径
    processed_video_path = os.path.join(
        app.config["PROCESSED_FOLDER"], "processed_" + filename
    )

    # 使用相同的分辨率和帧率创建视频写入对象
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")  # MP4编码器
    out = cv2.VideoWriter(processed_video_path, fourcc, fps, (width, height))

    # 逐帧处理视频
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # 将帧转换为灰度图像
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # 检测人脸
        faces = face_cascade.detectMultiScale(gray, 1.1, 4)

        # 在检测到的人脸区域绘制矩形框
        for x, y, w, h in faces:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)

        # 将处理后的帧写入输出视频
        out.write(frame)

    # 释放资源
    cap.release()
    out.release()

    return processed_video_path


if __name__ == "__main__":
    # 确保文件夹存在
    if not os.path.exists(UPLOAD_FOLDER):
        os.makedirs(UPLOAD_FOLDER)
    if not os.path.exists(PROCESSED_FOLDER):
        os.makedirs(PROCESSED_FOLDER)

    app.run(host="192.168.0.105", port=5000, debug=True)
