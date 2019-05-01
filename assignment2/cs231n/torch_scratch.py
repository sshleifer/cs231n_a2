import torch.nn.functional as F
import torch
from torch import nn
import numpy as np


def three_layer_convnet(x, params):
    """
    Performs the forward pass of a three-layer convolutional network with the
    architecture defined above.

    Inputs:
    - x: A PyTorch Tensor of shape (N, 3, H, W) giving a minibatch of images
    - params: A list of PyTorch Tensors giving the weights and biases for the
      network; should contain the following:
      - conv_w1: PyTorch Tensor of shape (channel_1, 3, KH1, KW1) giving weights
        for the first convolutional layer
      - conv_b1: PyTorch Tensor of shape (channel_1,) giving biases for the first
        convolutional layer
      - conv_w2: PyTorch Tensor of shape (channel_2, channel_1, KH2, KW2) giving
        weights for the second convolutional layer
      - conv_b2: PyTorch Tensor of shape (channel_2,) giving biases for the second
        convolutional layer
      - fc_w: PyTorch Tensor giving weights for the fully-connected layer. Can you
        figure out what the shape should be?
      - fc_b: PyTorch Tensor giving biases for the fully-connected layer. Can you
        figure out what the shape should be?

    Returns:
    - scores: PyTorch Tensor of shape (N, C) giving classification scores for x
    """
    conv_w1, conv_b1, conv_w2, conv_b2, fc_w, fc_b = params
    ################################################################################
    # TODO: Implement the forward pass for the three-layer ConvNet.                #
    ################################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    x = F.relu(F.conv2d(x, conv_w1, bias=conv_b1, padding=2))
    x = F.relu(F.conv2d(x, conv_w2, bias=conv_b2, padding=1))
    print(x.shape)
    x = torch.flatten(x, 1, -1)
    x = F.linear(x, fc_w, bias=fc_b)
    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    ################################################################################
    #                                 END OF YOUR CODE                             #
    ################################################################################
    return x

dtype = None


def init_default(m: nn.Module, func=nn.init.kaiming_normal_) -> None:
    "Initialize `m` weights with `func` and set `bias` to 0."
    if func:
        if hasattr(m, 'weight'): func(m.weight)
        if hasattr(m, 'bias') and hasattr(m.bias, 'data'): m.bias.data.fill_(0.)
    return m


def conv_layer(in_channels, nf, ks,  padding, **kwargs):
    c1 = init_default(
        nn.Conv2d(in_channels, nf, ks, padding=padding, **kwargs)
    )
    model = nn.Sequential(c1, nn.LeakyReLU(inplace=True, negative_slope=.1), nn.BatchNorm2d(nf))
    model.outshape = c1.outshape
    return model


class ConvFun(nn.Module):
    def __init__(self, in_channel, channel_1, channel_2, num_classes):
        super().__init__()
        self.conv1 = conv_layer(in_channel, channel_1, 5, padding=2, bias=True)
        self.conv2 = conv_layer(channel_1, channel_2, 3, padding=1, bias=True)
        self.lin = nn.Linear(np.prod(self.conv2.outshape), num_classes)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        scores = self.lin(Flatten()(x))
        return scores

def compute_outshape(c, H=32, W=32):
    nf, _, HH, WW = c.weight.shape
    P1, P2, stride1, stride2 = c.padding + c.stride
    Hp = int(1 + (H + 2 * P1 - HH) / stride1)
    Wp = int(1 + (W + 2 * P2 - WW) / stride2)
    out = (nf, Hp, Wp)
    return out
