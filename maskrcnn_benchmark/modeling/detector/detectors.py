# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
from .generalized_rcnn import GeneralizedRCNN


_DETECTION_META_ARCHITECTURES = {"GeneralizedRCNN": GeneralizedRCNN}


# cfg为配置信息
# 根据给定的配置信息实例化一个build_detection_model类对象
# 该类定义在 ./maskrcnn_benchmark/modeling/detector/generalized_rcnn.py 文件中
# build_detection_model()这个函数是创建模型的入口函数，也是唯一的模型创建函数
def build_detection_model(cfg):
    # 构建一个模型字典
    meta_arch = _DETECTION_META_ARCHITECTURES[cfg.MODEL.META_ARCHITECTURE]
    # 下面的语句等价于
    # return GeneralizedRCNN(cfg)
    return meta_arch(cfg)
