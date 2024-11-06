from easydict import EasyDict as edict
import numpy as np
from mindspore import Tensor
import mindspore.common.dtype as mstype
from mindspore.train.serialization import load_checkpoint, load_param_into_net, export

# 定义配置
cfg = edict(
    {
        "channel": 3,  # 图片通道数
        "image_width": 100,  # 图片宽度
        "image_height": 100,  # 图片高度
        "num_class": 5,  # 类别数
        "dropout_ratio": 0.5,
        "sigma": 0.01,
    }
)

# 实例化网络
net = Identification_Net(
    num_class=cfg.num_class, channel=cfg.channel, dropout_ratio=cfg.dropout_ratio
)

# 加载训练好的模型参数
param_dict = load_checkpoint(
    "/home/ma-user/work/checkpoint_classification-400_66.ckpt"
)  # 使用实际的.ckpt文件路径
load_param_into_net(net, param_dict)

# 定义一个dummy输入
dummy_input = Tensor(
    np.random.randn(1, cfg.channel, cfg.image_width, cfg.image_height), mstype.float32
)

# 导出为 air 格式
export(net, input_tensor, file_name="model_classification.air", file_format="AIR")
