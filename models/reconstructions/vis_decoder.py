import torch.nn as nn
from models.initializer import initialize_from_cfg


def conv3x3(inplanes, outplanes, stride=1, groups=1, dilation=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(
        inplanes,
        outplanes,
        kernel_size=3,
        stride=stride,
        padding=dilation,
        groups=groups,
        bias=False,
        dilation=dilation,
    )


def conv1x1(inplanes, outplanes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(inplanes, outplanes, kernel_size=1, stride=stride, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(
        self,
        inplanes,
        planes,
        stride=1,
        shortcut=None,
        groups=1,
        base_width=64,
        dilation=1,
        norm_layer=None,
    ):
        super(BasicBlock, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if groups != 1 or base_width != 64:
            raise ValueError("BasicBlock only supports groups=1 and base_width=64")
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")
        self.conv1 = conv3x3(inplanes, planes, stride=1)
        self.upsample = None
        if stride != 1:
            self.upsample = nn.Upsample(scale_factor=stride, mode="bilinear")
        self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = norm_layer(planes)

        self.shortcut = shortcut
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        if self.upsample is not None:
            out = self.upsample(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.shortcut is not None:
            identity = self.shortcut(x)

        out += identity
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    # Bottleneck in torchvision places the stride for downsampling at 3x3 convolution(self.conv2)
    # while original implementation places the stride at the first 1x1 convolution(self.conv1)
    # according to "Deep residual learning for image recognition"https://arxiv.org/abs/1512.03385.
    # This variant is also known as ResNet V1.5 and improves accuracy according to
    # https://ngc.nvidia.com/catalog/model-scripts/nvidia:resnet_50_v1_5_for_pytorch.

    expansion = 4

    def __init__(
        self,
        inplanes,
        planes,
        stride=1,
        upsample=None,
        groups=1,
        base_width=64,
        dilation=1,
        norm_layer=None,
    ):
        super(Bottleneck, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        width = int(planes * (base_width / 64.0)) * groups
        # Both self.conv2 and self.upsample layers upsample the input when stride != 1
        self.conv1 = conv1x1(inplanes, width)
        self.bn1 = norm_layer(width)
        self.conv2 = conv3x3(width, width, stride, groups, dilation)
        self.bn2 = norm_layer(width)
        self.conv3 = conv1x1(width, planes * self.expansion)
        self.bn3 = norm_layer(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.upsample = upsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.upsample is not None:
            identity = self.upsample(x)

        out += identity
        out = self.relu(out)

        return out


class ResNet(nn.Module):
    def __init__(
        self,
        inplanes,
        instrides,
        block,
        layers,
        groups=1,
        width_per_group=64,
        norm_layer=None,
        initializer=None,
    ):
        super(ResNet, self).__init__()
        assert isinstance(inplanes, list) and len(inplanes) == 1
        assert isinstance(instrides, list) and len(instrides) == 1
        self.inplanes = inplanes[0]
        self.instrides = instrides[0]

        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer
        self.dilation = 1
        layer_planes = [64, 128, 256, 512]
        if self.instrides == 32:
            layer_strides = [2, 2, 2, 1]
        elif self.instrides == 16:
            layer_strides = [1, 2, 2, 1]
        else:
            raise NotImplementedError

        self.groups = groups
        self.base_width = width_per_group
        self.layer4 = self._make_layer(
            block, layer_planes[3], layers[3], stride=layer_strides[3]
        )
        self.layer3 = self._make_layer(
            block, layer_planes[2], layers[2], stride=layer_strides[2]
        )
        self.layer2 = self._make_layer(
            block, layer_planes[1], layers[1], stride=layer_strides[1]
        )
        self.layer1 = self._make_layer(
            block, layer_planes[1], layers[0], stride=layer_strides[0]
        )
        self.upsample1 = nn.Upsample(scale_factor=2, mode="bilinear")
        self.conv1 = nn.Conv2d(
            self.inplanes, self.inplanes, kernel_size=7, stride=1, padding=3, bias=False
        )
        self.bn1 = norm_layer(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.upsample2 = nn.Upsample(scale_factor=2, mode="bilinear")
        self.conv2 = nn.Conv2d(self.inplanes, 3, kernel_size=1, stride=1, bias=False)
        initialize_from_cfg(self, initializer)

    def _make_layer(self, block, planes, blocks, stride=1):
        norm_layer = self._norm_layer
        shortcut = None
        previous_dilation = self.dilation
        if stride != 1 or self.inplanes != planes * block.expansion:
            shortcut = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride=1),
                nn.Upsample(scale_factor=stride, mode="bilinear"),
                norm_layer(planes * block.expansion),
            )

        layers = []
        layers.append(
            block(
                self.inplanes,
                planes,
                stride,
                shortcut,
                self.groups,
                self.base_width,
                previous_dilation,
                norm_layer,
            )
        )
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(
                block(
                    self.inplanes,
                    planes,
                    groups=self.groups,
                    base_width=self.base_width,
                    dilation=self.dilation,
                    norm_layer=norm_layer,
                )
            )

        return nn.Sequential(*layers)

    @property
    def layer0(self):
        return nn.Sequential(
            self.upsample1, self.conv1, self.bn1, self.relu, self.upsample2, self.conv2
        )

    def forward(self, input):
        x = input["feature_align"]

        for layer_idx in range(4, -1, -1):
            layer = getattr(self, f"layer{layer_idx}", None)
            if layer is not None:
                x = layer(x)

        return {"image_rec": x}


def VisDecoder(block_type, **kwargs):
    if block_type == "basic":
        return ResNet(block=BasicBlock, layers=[3, 4, 6, 3], **kwargs)
    elif block_type == "bottle":
        return ResNet(block=Bottleneck, layers=[3, 4, 6, 3], **kwargs)
    else:
        raise NotImplementedError
