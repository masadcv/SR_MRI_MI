import json
import random

import einops
import numpy as np


def load_json(file):
    try:
        with open(file, "r") as f:
            return json.load(f)
    except:
        raise IOError("json file {} not found".format(file))


def save_json(data, file):
    try:
        with open(file, "w") as f:
            json.dump(data, f, indent=4)
    except:
        raise IOError("json file {} write error".format(file))


class AverageMeter(object):
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count if self.count != 0 else 0


def imageify_numpy(data):
    if data.min() < 0.0 or data.max() > 1.0:
        data = (data - data.min()) / (data.max() - data.min())

    data *= 255
    return data.astype(np.uint8)


def calculate_random_crop_loc(data, patch_size):
    x = random.randint(0, data.shape[2] - patch_size[1])
    y = random.randint(0, data.shape[1] - patch_size[0])
    return y, x


def log_images_tensorboard(tb_writer, output, target, epoch, patch_size=(256, 256)):
    output = output.detach().cpu().numpy()
    target = target.detach().cpu().numpy()

    output = output[0, ...] if output.ndim > 3 else output
    target = target[0, ...] if target.ndim > 3 else target

    y, x = calculate_random_crop_loc(output, patch_size=patch_size)
    output = output[:, y : y + patch_size[0], x : x + patch_size[1]]
    target = target[:, y : y + patch_size[0], x : x + patch_size[1]]

    data = np.concatenate([output, target], axis=2)
    data = imageify_numpy(data)

    channels = data.shape[0]
    if channels % 4 == 0:
        a = 4
        b = int(channels / 4)
    elif channels % 2 == 0:
        b = 2
        a = int(channels / 2)
    else:
        b = 1
        a = channels

    data = einops.rearrange(data, "(a b) h w -> (a h) (b w)", a=a, b=b)

    tb_writer.add_image(
        "Prediction | Target", data, global_step=epoch, dataformats="HW"
    )
