# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
"""
Implements the Generalized R-CNN framework
这个文件定义了MaskrcnnBenchmark 的 GeneralizedRCNN 类, 用于表示各种组合后的目标检测模型
"""

import torch
from torch import nn

# 该函数定义于 ./maskrcnn_benchmark/structures/image_list.py 文件中
from maskrcnn_benchmark.structures.image_list import to_image_list

from ..backbone import build_backbone
from ..rpn.rpn import build_rpn
from ..roi_heads.roi_heads import build_roi_heads


# 定义类的具体实现
class GeneralizedRCNN(nn.Module):
    """
    该类是 MaskrcnnBenchmark 中所有模型的共同抽象, 目前支持 boxes 和 masks 两种形式的标签
    该类主要包含以下三个部分:
    - backbone
    - rpn(OPTION)
    - heads: 利用前面网络输出的 features 和 proposals 来计算 detections / masks.
    """

    # 根据配置信息初始化模型
    def __init__(self, cfg):
        # super() 函数是用于调用父类(超类)的一个方法。
        # super(type[, object-or-type])
        super(GeneralizedRCNN, self).__init__()

        # MaskrcnnBenchmark 模型的创建主要依赖于三个函数,
        # build_backbone(cfg),
        # build_rpn(cfg),
        # build_roi_heads(cfg)

        # 根据配置信息创建 backbone 网络
        self.backbone = build_backbone(cfg)

        # 根据配置信息创建 RPN 网络
        self.rpn = build_rpn(cfg, self.backbone.out_channels)

        # 根据配置信息创建 roi_heads
        self.roi_heads = build_roi_heads(cfg, self.backbone.out_channels)

    # 定义模型的前向传播过程
    def forward(self, images, targets=None):
        """
        参数
        Arguments:
            images (list[Tensor] or ImageList): images to be processed
            targets (list[BoxList]): ground-truth boxes present in the image (optional)

        返回值
        Returns:
            result (list[BoxList] or dict[Tensor]): the output from the model.
            在训练阶段，返回字典类型的模型损失
            在测试阶段，返回模型的预测结果
                During training, it returns a dict[Tensor] which contains the losses.
                During testing, it returns list[BoxList] contains additional fields
                like `scores`, `labels` and `mask` (for Mask R-CNN models).
        """
        # 当 training 设置为 True 时，必须提供 targets
        if self.training and targets is None:
            raise ValueError("In training mode, targets should be passed")

        # 将图片的数据类型转换成 ImageList
        # to_image_list 函数位于 MaskrcnnBenchmark 的结构模块当中
        images = to_image_list(images)

        # 利用 backbone 网络获取图片的 features
        features = self.backbone(images.tensors)

        # 利用 RPN 网络获取 proposals 和相应的 loss
        proposals, proposal_losses = self.rpn(images, features, targets)

        # 如果 roi_heads 不为 None 的话，就计算输出的结果
        if self.roi_heads:
            x, result, detector_losses = self.roi_heads(features, proposals, targets)
        else:
            # RPN-only models don't have roi_heads
            x = features
            result = proposals
            detector_losses = {}

        # 在训练模式下，输出损失值
        if self.training:
            losses = {}
            losses.update(detector_losses)
            losses.update(proposal_losses)
            return losses

        # 如果不在训练模式下，则输出模型的预测结果
        return result
