from io import BytesIO

import requests
import torch
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
from torch.autograd import Variable


def load_image(path):
    return Image.open(path)


def load_image_from_url(url):
    response = requests.get(url)
    return Image.open(BytesIO(response.content))


def load_features_extractor(model_name):
    net = getattr(models, model_name)(pretrained=True)
    for param in net.parameters():
        param.requires_grad = False
    return net.features


def get_resize_transform(im_size):
    transform_op = getattr(transforms, 'Resize', None)
    if transform_op is None:
        transform_op = transforms.Scale
    return transform_op((im_size, im_size))


def load_variable(pil_image, gpu=False):
    im_size = 512 if gpu else 224
    dtype = torch.cuda.FloatTensor if gpu else torch.FloatTensor
    tsfm = transforms.Compose([
        get_resize_transform(im_size),
        transforms.ToTensor()
    ])
    tensor = tsfm(pil_image)
    if gpu:
        tensor = tensor.cuda()
    tensor = tensor.type(dtype)
    return Variable(tensor, requires_grad=False).unsqueeze(0)
