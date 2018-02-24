from collections import defaultdict

import torch.nn as nn

content_layers_default = ['conv_4']
style_layers_default = ['conv_1', 'conv_2', 'conv_3', 'conv_4', 'conv_5']


class NeuralStyleNet(nn.Module):
    def __init__(self, features, content_variable, style_variable):
        super(NeuralStyleNet, self).__init__()
        self.features = features
        self.content_variable = content_variable
        self.style_variable = style_variable

    def forward(self, x):
        i = 0
        content, style = self.content_variable, self.style_variable
        outputs = defaultdict(list)
        for layer in list(self.features):
            x, content, style = layer(x.clone()), layer(content.clone()), layer(style.clone())
            if isinstance(layer, nn.Conv2d):
                i += 1
                name = 'conv_' + str(i)
                if name in content_layers_default:
                    outputs['content'].append((x, content))
                if name in style_layers_default:
                    outputs['style'].append((x, style))
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
