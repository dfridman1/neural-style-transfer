import torch
import torch.optim as optim
from torch.autograd import Variable
from torchvision.transforms import ToPILImage

from neural_style_net import ContentLoss, StyleLoss


class Solver(object):
    def __init__(self, model, content_var, style_var,
                 content_weight=1, style_weight=1000,
                 num_iters=200, gpu=False):
        self.model = model
        self.input_var, self.content_var, self.style_var = self._prepare_input(content_var, style_var, gpu)
        self.num_iters = num_iters
        self.gpu = gpu

        self.content_criterion = ContentLoss(weight=content_weight)
        self.style_criterion = StyleLoss(weight=style_weight)
        self.optimizer = optim.LBFGS([self.input_var])

    @staticmethod
    def _prepare_input(content_var, style_var, gpu=False):
        input_tensor = content_var.data.clone()
        if gpu:
            input_tensor = input_tensor.cuda()
            dtype = torch.cuda.FloatTensor
        else:
            dtype = torch.FloatTensor
        input_var = Variable(input_tensor.type(dtype), requires_grad=True)
        return input_var, content_var, style_var

    def train(self):
        iteration = [0]
        while iteration[0] <= self.num_iters:
            def closure():
                self.optimizer.zero_grad()
                self.input_var.data.clamp_(0, 1)
                outputs = self.model(self.input_var)
                style_loss = content_loss = 0
                for x, y in outputs['content']:
                    loss = self.content_criterion(x, y)
                    content_loss += loss
                    loss.backward(retain_graph=True)
                for x, y in outputs['style']:
                    loss = self.style_criterion(x, y)
                    style_loss += loss
                    loss.backward(retain_graph=True)
                if iteration[0] % 50 == 0:
                    print(
                        'it: {}, content: {}, style: {}'.format(iteration[0], content_loss.data[0], style_loss.data[0]))
                iteration[0] += 1
                return content_loss + style_loss

            self.optimizer.step(closure)
        self.input_var.data.clamp_(0, 1)
        styled_image = self.input_var
        if self.gpu:
            styled_image = styled_image.cpu()
        return ToPILImage()(styled_image.data[0])
