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
from mindspore.nn.optim import Adam  # Using Adam optimizer

folder_path = "F:\\Gitee\\engineering-practice-and-innovation-project-ii\\Mission2\\LeNet\\fashion_mnist\\model_save"


def create_dataset(data_path, batch_size=32, repeat_size=1, num_parallel_workers=1):
    """Create dataset for train or test."""
    mnist_ds = ds.MnistDataset(data_path)

    resize_height, resize_width = 32, 32
    rescale = 1.0 / 255.0
    shift = 0.0
    rescale_nml = 1 / 0.3081
    shift_nml = -1 * 0.1307 / 0.3081

    resize_op = Resize((resize_height, resize_width), interpolation=Inter.LINEAR)
    rescale_op = Rescale(rescale, shift)
    rescale_nml_op = Rescale(rescale_nml, shift_nml)
    hwc2chw_op = HWC2CHW()
    type_cast_op = TypeCast(mstype.int32)

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

    buffer_size = 10000
    mnist_ds = mnist_ds.shuffle(buffer_size=buffer_size)
    mnist_ds = mnist_ds.batch(batch_size, drop_remainder=True)
    mnist_ds = mnist_ds.repeat(repeat_size)

    return mnist_ds


class VGGModifiedLeNet5(nn.Cell):
    """Modified LeNet based on VGG structure."""

    def __init__(self, num_class=10, num_channel=1):
        super(VGGModifiedLeNet5, self).__init__()
        self.conv1 = nn.Conv2d(num_channel, 64, 3, pad_mode="pad", padding=1)
        self.bn1 = nn.BatchNorm2d(64)
        self.conv2 = nn.Conv2d(64, 64, 3, pad_mode="pad", padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 128, 3, pad_mode="pad", padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        self.conv4 = nn.Conv2d(128, 128, 3, pad_mode="pad", padding=1)
        self.bn4 = nn.BatchNorm2d(128)
        self.fc1 = nn.Dense(128 * 8 * 8, 256, weight_init=Normal(0.02))
        self.fc2 = nn.Dense(256, 256, weight_init=Normal(0.02))
        self.fc3 = nn.Dense(256, num_class, weight_init=Normal(0.02))
        self.relu = nn.ReLU()
        self.max_pool2d = nn.MaxPool2d(kernel_size=2, stride=2)
        self.flatten = nn.Flatten()

    def construct(self, x):
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.max_pool2d(self.relu(self.bn2(self.conv2(x))))
        x = self.relu(self.bn3(self.conv3(x)))
        x = self.max_pool2d(self.relu(self.bn4(self.conv4(x))))
        x = self.flatten(x)
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        return x


def train_net(network_model, epoch_size, data_path, repeat_size, ckpoint_cb, sink_mode):
    """Define the training method."""
    print("============== Starting Training ==============")
    ds_train = create_dataset(os.path.join(data_path, "train"), 32, repeat_size)
    network_model.train(
        epoch_size,
        ds_train,
        callbacks=[ckpoint_cb, LossMonitor()],
        dataset_sink_mode=sink_mode,
    )


def load_latest_checkpoint(folder_path):
    """Load the latest ckpt file in the folder."""
    ckpt_files = [f for f in os.listdir(folder_path) if f.endswith(".ckpt")]
    if not ckpt_files:
        print("No ckpt files found.")
        return None
    latest_file = max(
        ckpt_files, key=lambda f: os.path.getmtime(os.path.join(folder_path, f))
    )
    latest_file_path = os.path.join(folder_path, latest_file)
    print(f"Loading latest ckpt file: {latest_file_path}")
    return latest_file_path


def test_net(network, network_model, data_path):
    """Define the evaluation method."""
    print("============== Starting Testing ==============")
    latest_ckpt = load_latest_checkpoint(folder_path)
    if latest_ckpt:
        param_dict = load_checkpoint(latest_ckpt)
        load_param_into_net(network, param_dict)
    ds_eval = create_dataset(os.path.join(data_path, "test"))
    acc = network_model.eval(ds_eval, dataset_sink_mode=False)
    print(f"============== Accuracy: {acc} ==============")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="MindSpore VGG-Modified LeNet Example")
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

    lr = 0.001
    momentum = 0.9
    dataset_size = 1
    mnist_path = "Mission2/LeNet/fashion_mnist/FASHION_MNIST_Data"

    net_loss = SoftmaxCrossEntropyWithLogits(sparse=True, reduction="mean")
    train_epoch = 10
    net = VGGModifiedLeNet5()
    # 替换 Adam 优化器初始化
    net_opt = Adam(
        net.trainable_params(), learning_rate=lr
    )  # 使用正确的参数名 'learning_rate'

    config_ck = CheckpointConfig(
        save_checkpoint_steps=1875, keep_checkpoint_max=1, integrated_save=False
    )
    ckpoint = ModelCheckpoint(
        prefix="checkpoint_vgg_lenet", config=config_ck, directory=folder_path
    )

    model = Model(net, net_loss, net_opt, metrics={"Accuracy": Accuracy()})

    train_net(model, train_epoch, mnist_path, dataset_size, ckpoint, dataset_sink_mode)
    test_net(net, model, mnist_path)
