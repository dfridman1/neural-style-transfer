import argparse

import torch
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
from torch.autograd import Variable

from neural_style_net import NeuralStyleNet
from solver import Solver


def load_image(path):
    print('loading image')
    return Image.open(path)


def load_image_from_url(url):
    raise NotImplementedError


def load_features_extractor(model_name):
    net = getattr(models, model_name)(pretrained=True)
    for param in net.parameters():
        param.requires_grad = False
    return net.features


def load_variable(pil_image, gpu=False):
    im_size = 512 if gpu else 224
    dtype = torch.cuda.FloatTensor if gpu else torch.FloatTensor
    tsfm = transforms.Compose([
        transforms.Scale(im_size),
        transforms.ToTensor()
    ])
    tensor = tsfm(pil_image)
    if gpu:
        tensor = tensor.cuda()
    tensor = tensor.type(dtype)
    return Variable(tensor, requires_grad=False).unsqueeze(0)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--force_cpu', action='store_true')
    parser.add_argument('--net', choices=['vgg16', 'vgg19'], default='vgg19')
    parser.add_argument('--num_iters', type=int, default=300)
    parser.add_argument('--output_path')

    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--content_path')
    group.add_argument('--content_url')

    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--style_path')
    group.add_argument('--style_url')

    return parser.parse_args()


def main():
    args = parse_args()
    gpu = torch.cuda.is_available() and not args.force_cpu

    content_image = load_image_from_url(args.content_url) if args.content_url else load_image(args.content_path)
    style_image = load_image_from_url(args.style_url) if args.style_url else load_image(args.style_path)

    content_var = load_variable(content_image, gpu=gpu)
    style_var = load_variable(style_image, gpu=gpu)

    net = load_features_extractor(args.net)
    model = NeuralStyleNet(net, content_var, style_var)
    if gpu:
        model.cuda()

    solver = Solver(model, content_var, style_var, num_iters=args.num_iters, gpu=gpu)
    styled_image = solver.train()
    styled_image.save(args.output_path)


if __name__ == '__main__':
    main()
