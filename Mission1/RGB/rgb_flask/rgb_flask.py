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

# 确保上传文件夹存在
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)
if not os.path.exists(PROCESSED_FOLDER):
    os.makedirs(PROCESSED_FOLDER)


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/upload", methods=["POST"])
def upload_image():
    if "file" not in request.files:
        return redirect(request.url)

    file = request.files["file"]

    if file.filename == "":
        return redirect(request.url)

    if file:
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config["UPLOAD_FOLDER"], filename)
        file.save(filepath)

        # 读取图片并进行RGB通道分离
        image = cv2.imread(filepath)

        scale_percent = 12  # 图片缩小比例
        width = int(image.shape[1] * scale_percent / 100)
        height = int(image.shape[0] * scale_percent / 100)
        resized_image = cv2.resize(image, (width, height))

        b, g, r = cv2.split(resized_image)

        # 保存分离后的图片
        cv2.imwrite(os.path.join(app.config["PROCESSED_FOLDER"], "red_channel.jpg"), r)
        cv2.imwrite(
            os.path.join(app.config["PROCESSED_FOLDER"], "green_channel.jpg"), g
        )
        cv2.imwrite(os.path.join(app.config["PROCESSED_FOLDER"], "blue_channel.jpg"), b)
        cv2.imwrite(
            os.path.join(app.config["PROCESSED_FOLDER"], "original.jpg"), resized_image
        )

        return redirect(url_for("show_result"))


@app.route("/processed/<filename>")
def uploaded_file(filename):
    return send_from_directory(app.config["PROCESSED_FOLDER"], filename)


@app.route("/result")
def show_result():
    return render_template("result.html")


if __name__ == "__main__":
    app.run(host="192.168.0.105", port=5000, debug=True)
