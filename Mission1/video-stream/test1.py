from flask import Flask, render_template, url_for
import os

app = Flask(__name__)

# 获取当前脚本文件的目录
script_dir = os.path.dirname(os.path.abspath(__file__))

# 定义视频文件夹路径
VIDEO_FOLDER = os.path.join(script_dir, "static/videos")


# 首页：展示视频列表并选择播放
@app.route("/")
def index():
    # 获取视频文件夹中的所有视频文件
    video_files = os.listdir(VIDEO_FOLDER)

    # 生成每个视频的URL
    video_urls = {
        video: url_for("static", filename=f"videos/{video}") for video in video_files
    }

    return render_template("index.html", video_urls=video_urls)


if __name__ == "__main__":
    app.run(host="192.168.0.105", port=5000, debug=True)
