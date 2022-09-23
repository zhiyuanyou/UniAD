from .efficientnet import *  # noqa F401
from .resnet import *  # noqa F401

backbone_info = {
    "resnet18": {
        "layers": [1, 2, 3, 4],
        "planes": [64, 128, 256, 512],
        "strides": [4, 8, 16, 32],
    },
    "resnet34": {
        "layers": [1, 2, 3, 4],
        "planes": [64, 128, 256, 512],
        "strides": [4, 8, 16, 32],
    },
    "resnet50": {
        "layers": [1, 2, 3, 4],
        "planes": [256, 512, 1024, 2048],
        "strides": [4, 8, 16, 32],
    },
    "resnet101": {
        "layers": [1, 2, 3, 4],
        "planes": [256, 512, 1024, 2048],
        "strides": [4, 8, 16, 32],
    },
    "wide_resnet50_2": {
        "layers": [1, 2, 3, 4],
        "planes": [256, 512, 1024, 2048],
        "strides": [4, 8, 16, 32],
    },
    "efficientnet_b0": {
        "layers": [1, 2, 3, 4, 5],
        "blocks": [0, 2, 4, 10, 15],
        "planes": [16, 24, 40, 112, 320],
        "strides": [2, 4, 8, 16, 32],
    },
    "efficientnet_b1": {
        "layers": [1, 2, 3, 4, 5],
        "blocks": [1, 4, 7, 15, 22],
        "planes": [16, 24, 40, 112, 320],
        "strides": [2, 4, 8, 16, 32],
    },
    "efficientnet_b2": {
        "layers": [1, 2, 3, 4, 5],
        "blocks": [1, 4, 7, 15, 22],
        "planes": [16, 24, 48, 120, 352],
        "strides": [2, 4, 8, 16, 32],
    },
    "efficientnet_b3": {
        "layers": [1, 2, 3, 4, 5],
        "blocks": [1, 4, 7, 17, 25],
        "planes": [24, 32, 48, 136, 384],
        "strides": [2, 4, 8, 16, 32],
    },
    "efficientnet_b4": {
        "layers": [1, 2, 3, 4, 5],
        "blocks": [1, 5, 9, 21, 31],
        "planes": [24, 32, 56, 160, 448],
        "strides": [2, 4, 8, 16, 32],
    },
    "efficientnet_b5": {
        "layers": [1, 2, 3, 4, 5],
        "blocks": [2, 7, 12, 26, 38],
        "planes": [24, 40, 64, 176, 512],
        "strides": [2, 4, 8, 16, 32],
    },
    "efficientnet_b6": {
        "layers": [1, 2, 3, 4, 5],
        "blocks": [2, 8, 14, 30, 44],
        "planes": [32, 40, 72, 200, 576],
        "strides": [2, 4, 8, 16, 32],
    },
}
