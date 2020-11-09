# efficientnet backbone /xuatpham

import torch.nn as nn
from efficientnet_pytorch import EfficientNet
from .relu_fn_act import relu_fn


__all__ = ['EfficientNet_Extractor', 'efficientnet_b0', 'efficientnet_b1', 
            'efficientnet_b2', 'efficientnet_b3', 'efficientnet_b4', 'efficientnet_b5']


class EfficientNet_Extractor(nn.Module):

    def __init__(self, pretrained=True, mbblocks=[2, 7, 12, 26], backbone_name='efficientnet-b5'):
        super().__init__()        
        
        self.mbblocks = mbblocks
        
        if pretrained:
            self.model = EfficientNet.from_pretrained(backbone_name)
        else:
            self.model = EfficientNet.from_name(backbone_name)

    def forward(self, x):
        # get output of model._bn0 before the first MBConvBlock then apply swish/relu_fn activation function on it
        out = relu_fn(self.model._bn0(self.model._conv_stem(x)))
        get_mbblocks = []

        for idx, mbblock in enumerate(self.model._blocks):
            drop_connect_rate = self.model._global_params.drop_connect_rate
            
            if drop_connect_rate:
                drop_connect_rate *= float(idx) / len(self.model._blocks)
                print(drop_connect_rate)

            out = mbblock(out, drop_connect_rate=drop_connect_rate)

            if idx in self.mbblocks:
                get_mbblocks.append(out)

        # get output of model._bn1 after the last MBConvBlock then apply swish/relu_fn activation function on it
        last = relu_fn(self.model._bn1(self.model._conv_head(out)))
        get_mbblocks.append(last)

        return get_mbblocks # return according to mbblocks


def efficientnet_b0(pretrained=True, mbblocks=[2, 4, 9, 14], backbone_name='efficientnet-b0'):
    return EfficientNet_Extractor(pretrained, mbblocks, backbone_name)

def efficientnet_b1(pretrained=True, mbblocks=[4, 11, 15, 20], backbone_name='efficientnet-b1'):
    return EfficientNet_Extractor(pretrained, mbblocks, backbone_name)

def efficientnet_b2(pretrained=True, mbblocks=[1, 4, 7, 15], backbone_name='efficientnet-b2'):
    mbblocks=[4, 11, 15, 20] # Changed MBblocks
    return EfficientNet_Extractor(pretrained, mbblocks, backbone_name)    

def efficientnet_b3(pretrained=True, mbblocks=[1, 4, 7, 17], backbone_name='efficientnet-b3'):
    return EfficientNet_Extractor(pretrained, mbblocks, backbone_name)

def efficientnet_b4(pretrained=True, mbblocks=[1, 5, 9, 21], backbone_name='efficientnet-b4'):
    return EfficientNet_Extractor(pretrained, mbblocks, backbone_name)

def efficientnet_b5(pretrained=True, mbblocks=[2, 7, 12, 26], backbone_name='efficientnet-b5'):
    return EfficientNet_Extractor(pretrained, mbblocks, backbone_name)


if __name__ == '__main__':

    from PIL import Image
    from torchvision import transforms

    tfms = transforms.Compose([transforms.Resize(640), transforms.ToTensor()])
    img = tfms(Image.open('panda.jpg')).unsqueeze(0)
    model = EfficientNet.from_pretrained('efficientnet-b2')    
    out = relu_fn(model._bn0(model._conv_stem(img)))
    
    for idx, mbblock in enumerate(model._blocks):
        out = mbblock(out)
        print(idx, out.shape)

    last = relu_fn(model._bn1(model._conv_head(out)))        
    print(idx + 1, last.shape)
    print('==================================================')
    ef = efficientnet_b3()
    out_ef = ef(img)
    for block in out_ef:
        print(block.shape)
