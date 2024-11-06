import numpy as np
import cv2 as cv

from inferemote.testing import AiremoteTest

# from lenet import LeNet
from classification import Flower_CNN


class MyTest(AiremoteTest):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        """ An airemote object """
        self.air = Flower_CNN()

    """ Define a callback function for inferencing, which will be called for every single image """

    def run(self, image):
        orig_shape = image.shape[:2]
        result = self.air.inference_remote(image)
        new_image = self.air.make_image(result, orig_shape)
        return new_image


if __name__ == "__main__":
    #url_image = r"F:\Gitee\engineering-practice-and-innovation-project-ii\Mission4\Flower\flowers_photos\roses\roses_94.jpg"

    url_image = r"F:\Gitee\engineering-practice-and-innovation-project-ii\Mission4\Flower\flowers_photos\sunflowers\sunflowers_26.jpg"

    air = Flower_CNN()
    t = MyTest(air=air, input=url_image, mode="liveweb")

    t.start(remote="adk", mode="show")
