# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
import torch
import torch.nn.functional as F
from torch import nn

# 对于 ResNet-50-C4 来说, 只需要上面的 ResNet 模型即可完成特征提取任务,
# 但是对于 ResNet-50-FPN 来说, 我们还需要实现 FPN 网络以获得更强的特征提取能力,
# 在 backbone.py 文件中的 build_resnet_fpn_backbone(cfg) 函数中,
# 就使用了 fpn = fpn_module.FPN(...) 来创建一个 FPN 类的实例对象,
# 并且利用 nn.Sequential() 将 ResNet 和 FPN 组合在一起形成一个模型, 并将其返回


class FPN(nn.Module):
    """
    Module that adds FPN on top of a list of feature maps.
    The feature maps are currently supposed to be in increasing depth
    order, and must be consecutive
    """
    # 在一系列的 feature map （实际上就是 stage2-5 的最后一层输出）添加 FPN
    # 这些 feature maps 的 depth 假定是不断递增的，并且 feature maps 必须是连续的（从stage角度）

    def __init__(
        self, in_channels_list, out_channels, conv_block, top_blocks=None
    ):
        # in_channels_list (list[int]): 指示了送入 fpn 的每个 feature map 的通道数
        # out_channels (int): FPN表征的通道数, 所有的特征图谱最终都会转换成这个通道数大小
        # top_blocks (nn.Module or None): 当提供了 top_blocks 时, 就会在 FPN 的最后
        # 的输出上进行一个额外的 op, 然后 result 会扩展成 result list 返回

        """
        Arguments:
            in_channels_list (list[int]): number of channels for each feature map that
                will be fed
            out_channels (int): number of channels of the FPN representation
            top_blocks (nn.Module or None): if provided, an extra operation will
                be performed on the output of the last (smallest resolution)
                FPN output, and the result will extend the result list
        """
        super(FPN, self).__init__()

        # 创建两个空列表
        self.inner_blocks = []
        self.layer_blocks = []
        # 假设我们使用的是 ResNet-50-FPN 和配置, 则 in_channels_list 的值为:
        # [256, 512, 1024, 2048]
        for idx, in_channels in enumerate(in_channels_list, 1):  # 下标从1开始
            # 用下标起名，fpn_inner1, fpn_inner2, fpn_inner3, fpn_inner4
            inner_block = "fpn_inner{}".format(idx)
            # fpn_layer1, fpn_layer2, fpn_layer3, fpn_layer4
            layer_block = "fpn_layer{}".format(idx)

            if in_channels == 0:
                continue
            # 创建 inner_block 模块, 这里 in_channels 为各个stage输出的通道数
            # out_channels 为 256, 定义在用户配置文件中
            # 这里的卷积核大小为1, 该卷积层主要作用为改变通道数到 out_channels(降维)
            inner_block_module = conv_block(in_channels, out_channels, 1)
            # 改变 channels 后, 在每一个 stage 的特征图谱上再进行 3×3 的卷积计算, 通道数不变
            layer_block_module = conv_block(out_channels, out_channels, 3, 1)

            # 在当前的特征图谱上添加 FPN
            self.add_module(inner_block, inner_block_module)  # name module
            self.add_module(layer_block, layer_block_module)

            # 将当前 stage 的 fpn 模块的名字添加到对应的列表中
            self.inner_blocks.append(inner_block)
            self.layer_blocks.append(layer_block)

        # 将top_blocks作为 FPN 类的成员变量
        self.top_blocks = top_blocks

    def forward(self, x):
        """
        Arguments:
            x (list[Tensor]): feature maps for each feature level.
        Returns:
            results (tuple[Tensor]): feature maps after FPN layers.
                They are ordered from highest resolution first.
        """
        # x (list[Tensor]): 每个 feature level 的 feature maps,
        # ResNet的计算结果正好满足 FPN 的输入要求, 也因此可以使用 nn.Sequential 将二者直接结合
        # results (tuple[Tensor]): 经过FPN后的特征图谱组成的列表, 排列顺序是高分辨率的在前

        # 先计算最后一层(分辨率最低)特征图谱的fpn结果.
        last_inner = getattr(self, self.inner_blocks[-1])(x[-1])

        # 创建一个空的结果列表
        results = []
        results.append(getattr(self, self.layer_blocks[-1])(last_inner))
        for feature, inner_block, layer_block in zip(
            x[:-1][::-1], self.inner_blocks[:-1][::-1], self.layer_blocks[:-1][::-1]
        ):
            if not inner_block:
                continue
            inner_top_down = F.interpolate(last_inner, scale_factor=2, mode="nearest")
            inner_lateral = getattr(self, inner_block)(feature)
            # TODO use size instead of scale to make it robust to different sizes
            # inner_top_down = F.upsample(last_inner, size=inner_lateral.shape[-2:],
            # mode='bilinear', align_corners=False)
            last_inner = inner_lateral + inner_top_down
            results.insert(0, getattr(self, layer_block)(last_inner))

        if isinstance(self.top_blocks, LastLevelP6P7):
            last_results = self.top_blocks(x[-1], results[-1])
            results.extend(last_results)
        elif isinstance(self.top_blocks, LastLevelMaxPool):
            last_results = self.top_blocks(results[-1])
            results.extend(last_results)

        return tuple(results)


# 最后一级的 max pool 层
class LastLevelMaxPool(nn.Module):
    def forward(self, x):
        return [F.max_pool2d(x, 1, 2, 0)]


class LastLevelP6P7(nn.Module):
    """
    This module is used in RetinaNet to generate extra layers, P6 and P7.
    """
    def __init__(self, in_channels, out_channels):
        super(LastLevelP6P7, self).__init__()
        self.p6 = nn.Conv2d(in_channels, out_channels, 3, 2, 1)
        self.p7 = nn.Conv2d(out_channels, out_channels, 3, 2, 1)
        for module in [self.p6, self.p7]:
            nn.init.kaiming_uniform_(module.weight, a=1)
            nn.init.constant_(module.bias, 0)
        self.use_P5 = in_channels == out_channels

    def forward(self, c5, p5):
        x = p5 if self.use_P5 else c5
        p6 = self.p6(x)
        p7 = self.p7(F.relu(p6))
        return [p6, p7]
