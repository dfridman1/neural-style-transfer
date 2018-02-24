import argparse

import torch

from neural_style_net import NeuralStyleNet
from solver import Solver
from utils import load_image_from_url, load_image, load_variable, load_features_extractor


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
    model = NeuralStyleNet(net)
    if gpu:
        model.cuda()

    solver = Solver(model, content_var, style_var, num_iters=args.num_iters, gpu=gpu)
    styled_image = solver.train()
    styled_image.save(args.output_path)


if __name__ == '__main__':
    main()
