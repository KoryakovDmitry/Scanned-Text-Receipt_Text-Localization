# -*- coding: utf-8 -*-
# @Time    : 2019/8/23 21:57
# @Author  : zhoujun

# edit for efficientnet /xuatpham

import torch
from torch import nn
import torch.nn.functional as F
from models.modules import *

backbone_dict = {'resnet18': {'models': resnet18, 'out': [64, 128, 256, 512]},
                 'resnet34': {'models': resnet34, 'out': [64, 128, 256, 512]},
                 'resnet50': {'models': resnet50, 'out': [256, 512, 1024, 2048]},
                 'resnet101': {'models': resnet101, 'out': [256, 512, 1024, 2048]},
                 'resnet152': {'models': resnet152, 'out': [256, 512, 1024, 2048]},
                 'resnext50_32x4d': {'models': resnext50_32x4d, 'out': [256, 512, 1024, 2048]},
                 'resnext101_32x8d': {'models': resnext101_32x8d, 'out': [256, 512, 1024, 2048]},
                 'shufflenetv2': {'models': shufflenet_v2_x1_0, 'out': [24, 116, 232, 464]},
                 'efficientnet_b0': {'models': efficientnet_b0, 'out': [24, 40, 112, 192, 1280]}, # [2, 4, 9, 14]                 
                 'efficientnet_b1': {'models': efficientnet_b1, 'out': [24, 80, 112, 192, 1280]}, # [4, 11, 15, 20]
                 
                 # Changed MBblocks
                 'efficientnet_b2': {'models': efficientnet_b2, 'out': [24, 88, 120, 208, 1408]}, # [4, 11, 15, 20]
                #  'efficientnet_b2': {'models': efficientnet_b2, 'out': [16, 24, 48, 120, 1408]}, # [1, 4, 7, 15]                 
                 
                 'efficientnet_b3': {'models': efficientnet_b3, 'out': [24, 32, 48, 136, 1536]}, # [1, 4, 7, 17]                 
                 'efficientnet_b4': {'models': efficientnet_b4, 'out': [24, 32, 56, 160, 1792]}, # [1, 5, 9, 21]                 
                 'efficientnet_b5': {'models': efficientnet_b5, 'out': [24, 40, 64, 176, 2048]}, # [2, 7, 12, 26]
                 }

segmentation_head_dict = {'FPN': FPN, 'FPEM_FFM': FPEM_FFM, 'FPEM_FFM_effnet': FPEM_FFM_effnet}


class Model(nn.Module):
    def __init__(self, model_config: dict):
        """
        PANnet
        :param model_config: 模型配置
        """
        super().__init__()
        backbone = model_config['backbone']
        pretrained = model_config['pretrained']
        segmentation_head = model_config['segmentation_head']

        assert backbone in backbone_dict, 'backbone must in: {}'.format(backbone_dict)
        assert segmentation_head in segmentation_head_dict, 'segmentation_head must in: {}'.format(
            segmentation_head_dict)

        backbone_model, backbone_out = backbone_dict[backbone]['models'], backbone_dict[backbone]['out']
        self.backbone = backbone_model(pretrained=pretrained)
        self.segmentation_head = segmentation_head_dict[segmentation_head](backbone_out, **model_config)
        self.name = '{}_{}'.format(backbone, segmentation_head)

    def forward(self, x):
        _, _, H, W = x.size()
        backbone_out = self.backbone(x)
        segmentation_head_out = self.segmentation_head(backbone_out)
        y = F.interpolate(segmentation_head_out, size=(H, W), mode='bilinear', align_corners=True)
        return y


if __name__ == '__main__':
    device = torch.device('cpu')
    x = torch.zeros(1, 3, 640, 640).to(device)

    model_config = {
        'backbone': 'shufflenetv2',
        'fpem_repeat': 4,
        'pretrained': True, 
        'result_num': 7,
        'segmentation_head': 'FPEM_FFM'
    }
    model = Model(model_config=model_config).to(device)
    y = model(x)
    print(y.shape)
    # print(model)
    # torch.save(model.state_dict(), 'PAN.pth')
