import io
import os
import win32clipboard
import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk, ImageGrab
import numpy as np
import cv2 as cv


from inferemote.testing import AiremoteTest
from classification_p import Flower_CNN

# 指定保存文件夹路径
save_folder = r"F:\Gitee\engineering-practice-and-innovation-project-ii\Mission4\Flower\T\Clipboard"  # 替换为你想要保存的文件夹路径
save_path = os.path.join(save_folder, "clipboard_image.png")


class MyTest(AiremoteTest):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        """ An airemote object """
        self.air = Flower_CNN(remote="localhost", port=8931, wait=5)

    """ Define a callback function for inferencing, which will be called for every single image """

    def run(self, image):
        orig_shape = image.shape[:2]
        result = self.air.inference_remote(image)
        new_image = self.air.make_image(result, orig_shape)
        return new_image


class FlowerApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Flower Classification")
        self.root.geometry("500x500")

        # Add a button to open an image file
        self.btn_open = tk.Button(
            root, text="选择图片文件", command=self.open_image_file
        )
        self.btn_open.pack(pady=20)

        # Add a button to paste from clipboard
        self.btn_paste = tk.Button(
            root, text="从剪贴板粘贴图片", command=self.paste_from_clipboard
        )
        self.btn_paste.pack(pady=20)

        # Add a label to show the image
        self.img_label = tk.Label(root)
        self.img_label.pack()

        # Initialize Flower_CNN model with default parameters
        self.air = Flower_CNN(remote="localhost", port=8931, wait=5)

    def open_image_file(self):
        file_path = filedialog.askopenfilename(
            title="选择图片文件",
            filetypes=(
                ("JPEG files", "*.jpg"),
                ("PNG files", "*.png"),
                ("All files", "*.*"),
            ),
        )
        if file_path:
            image = cv.imread(file_path)
            self.process_and_display_image(image)

    def paste_from_clipboard(self):
        try:
            # 获取剪贴板图像
            win32clipboard.OpenClipboard()
            data = win32clipboard.GetClipboardData(win32clipboard.CF_DIB)
            win32clipboard.CloseClipboard()
            # 处理图像数据
            image = Image.open(io.BytesIO(data))
            # 保存图像到指定文件夹
            image.save(save_path, "PNG")
            image = cv.imread(save_path)
            self.process_and_display_image(image)
        except Exception as e:
            messagebox.showerror("错误", f"无法从剪贴板读取图片: {e}")

    def process_and_display_image(self, image):
        # Run inference and display the result
        orig_shape = image.shape[:2]
        result = self.air.inference_remote(image)
        new_image = self.air.make_image(result, orig_shape)

        # Convert to PIL format to display in Tkinter
        new_image = Image.fromarray(cv.cvtColor(new_image, cv.COLOR_BGR2RGB))
        tk_image = ImageTk.PhotoImage(new_image)

        # Update the label with the new image
        self.img_label.configure(image=tk_image)
        self.img_label.image = tk_image


if __name__ == "__main__":
    root = tk.Tk()
    app = FlowerApp(root)
    root.mainloop()
