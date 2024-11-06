# Copyright 2020 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================
"""Lenet Tutorial
This sample code is applicable to CPU, GPU and Ascend.
"""
import os
import stat
import shutil
import argparse
import mindspore.dataset as ds
import mindspore.nn as nn
from mindspore import context, Model, load_checkpoint, load_param_into_net
from mindspore.common.initializer import Normal
from mindspore.train.callback import ModelCheckpoint, CheckpointConfig, LossMonitor
from mindspore.dataset.vision import Resize, Rescale, HWC2CHW, Inter
from mindspore.dataset.transforms import TypeCast
from mindspore.nn import Accuracy
from mindspore import dtype as mstype
from mindspore.nn import SoftmaxCrossEntropyWithLogits


# from mindspore.nn import SoftmaxCrossEntropyWithLogits
# from utils.dataset import download_dataset

folder_path = "F:\Gitee\engineering-practice-and-innovation-project-ii\Mission2\LeNet\mnist\model_save"


def create_dataset(data_path, batch_size=32, repeat_size=1, num_parallel_workers=1):
    """create dataset for train or test
    Args:
        data_path: Data path
        batch_size: The number of data records in each group
        repeat_size: The number of replicated data records
        num_parallel_workers: The number of parallel workers
    """
    # define dataset
    mnist_ds = ds.MnistDataset(data_path)

    # define operation parameters
    resize_height, resize_width = 32, 32
    rescale = 1.0 / 255.0
    shift = 0.0
    rescale_nml = 1 / 0.3081
    shift_nml = -1 * 0.1307 / 0.3081

    # define map operations
    resize_op = Resize(
        (resize_height, resize_width), interpolation=Inter.LINEAR
    )  # Resize images to (32, 32)
    rescale_nml_op = Rescale(rescale_nml, shift_nml)  # normalize images
    rescale_op = Rescale(rescale, shift)  # rescale images
    hwc2chw_op = (
        HWC2CHW()
    )  # change shape from (height, width, channel) to (channel, height, width) to fit network.
    type_cast_op = TypeCast(mstype.int32)

    # apply map operations on images
    mnist_ds = mnist_ds.map(
        operations=type_cast_op,
        input_columns="label",
        num_parallel_workers=num_parallel_workers,
    )
    mnist_ds = mnist_ds.map(
        operations=resize_op,
        input_columns="image",
        num_parallel_workers=num_parallel_workers,
    )
    mnist_ds = mnist_ds.map(
        operations=rescale_op,
        input_columns="image",
        num_parallel_workers=num_parallel_workers,
    )
    mnist_ds = mnist_ds.map(
        operations=rescale_nml_op,
        input_columns="image",
        num_parallel_workers=num_parallel_workers,
    )
    mnist_ds = mnist_ds.map(
        operations=hwc2chw_op,
        input_columns="image",
        num_parallel_workers=num_parallel_workers,
    )

    # apply DatasetOps
    buffer_size = 10000
    mnist_ds = mnist_ds.shuffle(
        buffer_size=buffer_size
    )  # 10000 as in LeNet train script
    mnist_ds = mnist_ds.batch(batch_size, drop_remainder=True)
    mnist_ds = mnist_ds.repeat(repeat_size)

    return mnist_ds


class LeNet5(nn.Cell):
    """Lenet network structure."""

    # define the operator required
    def __init__(self, num_class=10, num_channel=1):
        super(LeNet5, self).__init__()
        self.conv1 = nn.Conv2d(num_channel, 6, 5, pad_mode="pad")
        self.conv2 = nn.Conv2d(6, 16, 5, pad_mode="pad")
        self.fc1 = nn.Dense(16 * 5 * 5, 120, weight_init=Normal(0.02))
        self.fc2 = nn.Dense(120, 84, weight_init=Normal(0.02))
        self.fc3 = nn.Dense(84, num_class, weight_init=Normal(0.02))
        self.relu = nn.ReLU()
        self.max_pool2d = nn.MaxPool2d(kernel_size=2, stride=2)
        self.flatten = nn.Flatten()

    # use the preceding operators to construct networks
    def construct(self, x):
        x = self.max_pool2d(self.relu(self.conv1(x)))
        x = self.max_pool2d(self.relu(self.conv2(x)))
        x = self.flatten(x)
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        return x


def train_net(network_model, epoch_size, data_path, repeat_size, ckpoint_cb, sink_mode):
    """Define the training method."""
    print("============== Starting Training ==============")
    # load training dataset
    ds_train = create_dataset(os.path.join(data_path, "train"), 32, repeat_size)
    network_model.train(
        epoch_size,
        ds_train,
        callbacks=[ckpoint_cb, LossMonitor()],
        dataset_sink_mode=sink_mode,
    )


def load_latest_checkpoint(folder_path):
    """加载指定文件夹下最新的ckpt文件"""
    ckpt_files = [f for f in os.listdir(folder_path) if f.endswith(".ckpt")]

    if not ckpt_files:
        print("没有找到ckpt文件。")
        return None

    # 获取最新的ckpt文件
    latest_file = max(
        ckpt_files, key=lambda f: os.path.getmtime(os.path.join(folder_path, f))
    )
    latest_file_path = os.path.join(folder_path, latest_file)

    print(f"加载最新的ckpt文件: {latest_file_path}")
    return latest_file_path


def test_net(network, network_model, data_path):
    """Define the evaluation method."""
    print("============== Starting Testing ==============")
    # load the saved model for evaluation
    latest_ckpt = load_latest_checkpoint(folder_path)
    param_dict = load_checkpoint(latest_ckpt)
    # load parameter to the network
    load_param_into_net(network, param_dict)
    # load testing dataset
    ds_eval = create_dataset(os.path.join(data_path, "test"))
    acc = network_model.eval(ds_eval, dataset_sink_mode=False)
    print("============== Accuracy:{} ==============".format(acc))


def delete_files_in_folder(folder_path):
    """删除指定文件夹中的所有文件和子文件夹，如果文件夹为空则不执行。"""
    # 检查文件夹是否存在
    if os.path.exists(folder_path) and os.path.isdir(folder_path):
        # 列出文件夹中的所有文件和子文件夹
        files = os.listdir(folder_path)

        # 检查文件夹是否为空
        if files:
            # 遍历文件夹中的所有文件
            for file in files:
                file_path = os.path.join(folder_path, file)
                # 更改文件权限
                os.chmod(file_path, stat.S_IWUSR)  # 给用户写权限
                try:
                    # 删除文件
                    if os.path.isfile(file_path):
                        os.remove(file_path)
                        print(f"{file_path} 已成功删除。")
                    # 如果是文件夹，可以选择递归删除
                    elif os.path.isdir(file_path):
                        shutil.rmtree(file_path)
                        print(f"{file_path} 目录已成功删除。")
                except PermissionError:
                    print(f"没有权限删除 {file_path}。")
                except Exception as e:
                    print(f"删除 {file_path} 时发生错误: {e}")
        else:
            print("文件夹为空，不执行删除操作。")
    else:
        print(f"文件夹 {folder_path} 不存在。")


if __name__ == "__main__":

    delete_files_in_folder(folder_path)

    parser = argparse.ArgumentParser(description="MindSpore LeNet Example")
    parser.add_argument(
        "--device_target",
        type=str,
        default="CPU",
        choices=["Ascend", "GPU", "CPU"],
        help="device where the code will be implemented (default: CPU)",
    )
    args = parser.parse_args()
    context.set_context(mode=context.GRAPH_MODE, device_target=args.device_target)
    dataset_sink_mode = not args.device_target == "CPU"
    # download mnist dataset
    # download_dataset()
    # learning rate setting
    lr = 0.01
    momentum = 0.9
    dataset_size = 1
    mnist_path = "Mission2/LeNet/mnist/MNIST_Data"
    # define the loss function
    net_loss = SoftmaxCrossEntropyWithLogits(sparse=True, reduction="mean")
    train_epoch = 1
    # create the network
    net = LeNet5()
    # define the optimizer
    net_opt = nn.Momentum(net.trainable_params(), lr, momentum)
    config_ck = CheckpointConfig(
        save_checkpoint_steps=1875, keep_checkpoint_max=1, integrated_save=False
    )
    # save the network model and parameters for subsequence fine-tuning
    ckpoint = ModelCheckpoint(
        prefix="checkpoint_lenet",
        config=config_ck,
        directory="F:\Gitee\engineering-practice-and-innovation-project-ii\Mission2\LeNet\mnist\model_save",
    )

    # group layers into an object with training and evaluation features
    model = Model(net, net_loss, net_opt, metrics={"Accuracy": Accuracy()})

    train_net(model, train_epoch, mnist_path, dataset_size, ckpoint, dataset_sink_mode)
    test_net(net, model, mnist_path)
