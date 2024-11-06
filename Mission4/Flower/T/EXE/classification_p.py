import sys, os
import numpy as np
import cv2 as cv

from inferemote.atlas_remote import AtlasRemote
from inferemote.image_encoder import ImageEncoder


class Flower_CNN(AtlasRemote):

    def __init__(self, remote="localhost", port=8931, wait=5, **kwargs):
        # 确保只传递所需的参数
        super().__init__(port=port, remote=remote, **kwargs)
        self.wait = wait

    def pre_process(self, bgr_img):
        # 将BGR图像转换为RGB
        rgb_img = cv.cvtColor(bgr_img, cv.COLOR_BGR2RGB)
        # 调整图像大小为 (100, 100)
        rgb_img = cv.resize(rgb_img, (100, 100))
        # 转换为 NCHW 格式
        rgb_img = rgb_img.transpose(2, 0, 1).copy().astype(np.float32)
        # 将图像数据编码为字节流
        return rgb_img.tobytes()

    def post_process(self, result):
        # 将远程推理结果从字节流转换为 numpy 数组
        blob = np.frombuffer(result[0], np.float32)

        # 计算 Softmax 概率
        vals = blob.flatten()
        max_val = np.max(vals)
        vals = np.exp(vals - max_val)
        sum_val = np.sum(vals)
        probs = vals / sum_val
        # 获取概率最高的类别
        top_k = probs.argsort()[-1:-6:-1]
        top1 = top_k[0]
        class_names = {
            0: "daisy",
            1: "dandelion",
            2: "roses",
            3: "sunflowers",
            4: "tulips",
        }
        # 获取预测的花名
        fl_name = class_names.get(top1, "Unknown")
        return fl_name

    def make_image(self, fl_name, orig_shape):
        # 将 "Prediction:" 和花卉类型内容分成两行
        text1 = "Prediction:"
        text2 = fl_name

        # 打印预测结果
        print("FLOWER_CLASS:", text1, text2)

        # 创建空白图像
        image = np.zeros((300, 400, 3), dtype=np.uint8)

        # 绘制 "Prediction:" 文本在图像上
        image = cv.putText(
            image, text1, (10, 50), cv.FONT_HERSHEY_COMPLEX, 1.0, (0, 255, 0), 2
        )

        # 绘制花卉类型文本在图像的下一行
        image = cv.putText(
            image, text2, (10, 100), cv.FONT_HERSHEY_COMPLEX, 1.0, (0, 255, 0), 2
        )

        return image
