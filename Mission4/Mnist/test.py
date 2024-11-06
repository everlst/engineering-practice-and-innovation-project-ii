# easydict模块用于以属性的方式访问字典的值
from easydict import EasyDict as edict

# glob模块主要用于查找符合特定规则的文件路径名，类似使用windows下的文件搜索
import glob

# os模块主要用于处理文件和目录
import os

import numpy as np
import matplotlib.pyplot as plt

import mindspore

# 导入mindspore框架数据集
import mindspore.dataset as ds

# vision.c_transforms模块是处理图像增强的高性能模块，用于数据增强图像数据改进训练模型。
import mindspore.dataset.vision.c_transforms as CV

# c_transforms模块提供常用操作，包括OneHotOp和TypeCast
import mindspore.dataset.transforms.c_transforms as C
from mindspore.common import dtype as mstype
from mindspore import context

# 导入模块用于初始化截断正态分布
from mindspore.common.initializer import TruncatedNormal
from mindspore import nn
from mindspore.train import Model
from mindspore.train.callback import (
    ModelCheckpoint,
    CheckpointConfig,
    LossMonitor,
    TimeMonitor,
)
from mindspore.train.serialization import load_checkpoint, load_param_into_net
from mindspore import Tensor

# 设置MindSpore的执行模式和设备
context.set_context(mode=context.GRAPH_MODE, device_target="Ascend")
