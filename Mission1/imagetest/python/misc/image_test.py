"""
Inferemote: a Remote Inference Toolkit for Ascend 310

"""

""" Prepare a test """
import cv2 as cv


# 颜色反转/高斯双边滤波
def bilateral_filter(image):
    image = cv.bitwise_not(image)
    image = cv.bilateralFilter(image, 0, 100, 15)
    return image


from inferemote.testing import ImageTest


class MyTest(ImageTest):
    """Define a callback function for inferencing, which will be called for every single image"""

    def run(self, image):
        """an image must be returned in the same shape"""
        new_img = bilateral_filter(image)
        return new_img
        #return image


if __name__ == "__main__":

    t = MyTest(mode="liveweb", threads=100)
    t.start(
        ###input="C:/Users/27310/",
        #input="C:/Users/27310/Videos/1_1_1.mp4",
        input="camera",
        mode="show",
    )
