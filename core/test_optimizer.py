# -*- coding: utf-8 -*- 
# Time: 2022-01-29 17:11
# Copyright (c) 2022
# author: Euraxluo


import numpy as np
import torch
from torch.nn import functional
from unittest import TestCase
from .tensor import Tensor
from .optimizer import Adam, SGD, RMSprop

x_init = np.random.randn(1, 3).astype(np.float32)
W_init = np.random.randn(3, 3).astype(np.float32)
m_init = np.random.randn(1, 3).astype(np.float32)


def step_test_grad(optimizer, kwargs={}):
    net = SimpleNet()
    optimizer = optimizer([net.x, net.W], **kwargs)
    out = net.forward()
    out.backward()
    optimizer.step()
    return net.x.data, net.W.data


def step_pytorch(optimizer, kwargs={}):
    net = TorchNet()
    optimizer = optimizer([net.x, net.W], **kwargs)
    out = net.forward()
    out.backward()
    optimizer.step()
    return net.x.detach().numpy(), net.W.detach().numpy()


class SimpleNet:
    def __init__(self):
        self.x = Tensor(x_init.copy())
        self.W = Tensor(W_init.copy())
        self.m = Tensor(m_init.copy())

    def forward(self):
        out = self.x.dot(self.W).relu()
        out = out.logsoftmax()
        out = out.mul(self.m).add(self.m).sum()
        return out


class TorchNet:
    def __init__(self):
        self.x = torch.tensor(x_init.copy(), requires_grad=True)
        self.W = torch.tensor(W_init.copy(), requires_grad=True)
        self.m = torch.tensor(m_init.copy())

    def forward(self):
        out = self.x.matmul(self.W).relu()
        out = torch.nn.functional.log_softmax(out, dim=1)
        out = out.mul(self.m).add(self.m).sum()
        return out


class TestOptimizer(TestCase):

    def test_adam(self):
        for x, y in zip(step_test_grad(Adam),
                        step_pytorch(torch.optim.Adam)):
            np.testing.assert_allclose(x, y, atol=1e-4)

    def test_sgd(self):
        for x, y in zip(step_test_grad(SGD, kwargs={'lr': 0.001}),
                        step_pytorch(torch.optim.SGD, kwargs={'lr': 0.001})):
            np.testing.assert_allclose(x, y, atol=1e-5)

    def test_rms_prop(self):
        for x, y in zip(step_test_grad(RMSprop, kwargs={'lr': 0.001, 'decay': 0.99}),
                        step_pytorch(torch.optim.RMSprop,
                                     kwargs={'lr': 0.001, 'alpha': 0.99})):
            np.testing.assert_allclose(x, y, atol=1e-5)
