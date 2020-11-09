# xuatpham

# https://github.com/lukemelas/EfficientNet-PyTorch/issues/18#issuecomment-511677853
# https://github.com/lukemelas/EfficientNet-PyTorch/issues/27
# more about relu_fn: https://medium.com/@neuralnets/swish-activation-function-by-google-53e1ea86f820

import torch
import torch.nn as nn


sigmoid = torch.nn.Sigmoid()


class Swish(torch.autograd.Function):
    @staticmethod
    def forward(ctx, i):
        result = i * sigmoid(i)
        ctx.save_for_backward(i)
        return result

    @staticmethod
    def backward(ctx, grad_output):
        i = ctx.saved_variables[0]
        sigmoid_i = sigmoid(i)
        return grad_output * (sigmoid_i * (1 + i * (1 - sigmoid_i)))


swish = Swish.apply


class Swish_module(nn.Module):
    def forward(self, x):
        return swish(x)


swish_layer = Swish_module()


def relu_fn(x):
    """ Swish activation function """
    # return x * torch.sigmoid(x)
    return swish_layer(x)



if __name__ == '__main__':
    
    from efficientnet_pytorch import EfficientNet
    from PIL import Image
    from torchvision import transforms

    tfms = transforms.Compose([
        transforms.Resize(224), transforms.ToTensor()])
    img = tfms(Image.open('panda.jpg')).unsqueeze(0)
    print('img shape', img.shape)
    model = EfficientNet.from_pretrained('efficientnet-b0')
    out = relu_fn(model._bn0(model._conv_stem(img)))
    print('output shape', out.shape)
