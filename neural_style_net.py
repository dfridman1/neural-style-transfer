from collections import defaultdict

import torch.nn as nn


class NeuralStyleNet(nn.Module):
    def __init__(self, features, content_layers=('conv_4',),
                 style_layers=('conv_1', 'conv_2', 'conv_3', 'conv_4', 'conv_5')):
        super(NeuralStyleNet, self).__init__()
        self.features = features
        self.content_layers = content_layers
        self.style_layers = style_layers

    def forward(self, x):
        i = 0
        content_outputs, style_outputs = {}, {}
        for layer in list(self.features):
            x = layer(x.clone())
            if isinstance(layer, nn.Conv2d):
                i += 1
                name = 'conv_' + str(i)
            elif isinstance(layer, nn.MaxPool2d):
                name = 'pool_' + str(i)
            elif isinstance(layer, nn.ReLU):
                name = 'relu_' + str(i)
            else:
                raise ValueError()
            if name in self.content_layers:
                content_outputs[name] = x
            if name in self.style_layers:
                style_outputs[name] = x
        outputs = {'content': content_outputs, 'style': style_outputs}
        return outputs


class ContentLoss(nn.MSELoss):
    def __init__(self, weight=1):
        super(ContentLoss, self).__init__()
        self.weight = weight

    def forward(self, input, target):
        return super(ContentLoss, self).forward(input.mul(self.weight), target.mul(self.weight))


class StyleLoss(nn.MSELoss):
    def __init__(self, weight=1000):
        super(StyleLoss, self).__init__()
        self.weight = weight
        self.gram = Gram()

    def forward(self, input, target):
        return super(StyleLoss, self).forward(self.gram(input).mul(self.weight), self.gram(target).mul(self.weight))


class Gram(nn.Module):
    def forward(self, x):
        n, c, w, h = x.size()
        x = x.view(n * c, w * h)
        return x.mm(x.t()).div(n * c * w * h)
