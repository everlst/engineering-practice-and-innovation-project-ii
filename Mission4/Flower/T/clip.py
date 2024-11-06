import io
import os
import win32clipboard
from PIL import Image

# 指定保存文件夹路径
save_folder = r"F:\Gitee\engineering-practice-and-innovation-project-ii\Mission4\Flower\T\Clipboard"  # 替换为你想要保存的文件夹路径
save_path = os.path.join(save_folder, "clipboard_image.png")

# 获取剪贴板图像
win32clipboard.OpenClipboard()
data = win32clipboard.GetClipboardData(win32clipboard.CF_DIB)
win32clipboard.CloseClipboard()

# 检查图像格式
if data is None:
    print("剪贴板中没有图像数据")
    exit()

# 处理图像数据
image = Image.open(io.BytesIO(data))

# 保存图像到指定文件夹
image.save(save_path, "PNG")
print(f"成功将图像保存到 {save_path}")
