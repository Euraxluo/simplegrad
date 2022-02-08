# -*- coding: utf-8 -*- 
# Time: 2022-01-20 10:00
# Copyright (c) 2022
# author: Euraxluo

import os

os.environ['GRAPH'] = "true"

from core.tensor import *
from core import optimizer as opt
from core.loss import *
from core.helper import *

import unittest
import numpy as np
from dataset.minist import fetch_mnist
from tqdm import trange


class SimpleNet:

    def __init__(self):
        self.l1 = Tensor.uniform(784, 128)
        self.l2 = Tensor.uniform(128, 10)

    def forward(self, x: Any):
        return x.dot(self.l1).relu().dot(self.l2).logsoftmax()


class Linear:
    def __init__(self, in_dim, out_dim, bias=True):
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.use_bias = bias
        self.weight = Tensor.uniform(in_dim, out_dim)
        if self.use_bias:
            self.bias = Tensor.zeros(1, out_dim)

    def __call__(self, x):
        B, *dims, D = x.shape
        x = x.reshape(shape=(B * np.prod(dims).astype(np.int32), D))
        x = x.dot(self.weight)
        if self.use_bias:
            tmp = self.bias.reshape(shape=(1, -1))
            x = x.add(tmp)
        x = x.reshape(shape=(B, *dims, -1))
        return x


class LinearNet(Forward):
    def __init__(self):
        self.fc1 = Linear(28 * 28, 64)
        # self.fc2 = Linear(256, 64)
        self.fc3 = Linear(64, 10)

    def forward(self, x):
        x = self.fc1(x).relu()
        # x = self.fc2(x).relu()
        x = self.fc3(x).logsoftmax()
        return x


class SimpleConvNet:
    def __init__(self):
        # https://keras.io/examples/vision/mnist_convnet/
        conv = 3
        # inter_chan, out_chan = 32, 64
        inter_chan, out_chan = 8, 16  # for speed
        self.c1 = Tensor.uniform(inter_chan, 1, conv, conv)
        self.c2 = Tensor.uniform(out_chan, inter_chan, conv, conv)
        self.l1 = Tensor.uniform(out_chan * 5 * 5, 10)

    def forward(self, x):
        x = x.reshape(shape=(-1, 1, 28, 28))  # hacks
        x = x.conv2d(self.c1).relu().max_pool2d()
        x = x.conv2d(self.c2).relu().max_pool2d()
        x = x.reshape(shape=[x.shape[0], -1])
        return x.dot(self.l1).logsoftmax()


X_train, Y_train, X_test, Y_test = fetch_mnist()


class TestMNIST(unittest.TestCase):
    def test_conv_adam(self):
        np.random.seed(1)
        model = SimpleConvNet()
        optimizer = opt.Adam(model, lr=0.0002)
        train(model, X_train, Y_train, optimizer, steps=1000, loss_function=sparse_categorical_cross_entropy)
        assert evaluate(model, X_test, Y_test) > 0.95

    def test_simple_adam(self):
        np.random.seed(1)
        model = SimpleNet()
        optimizer = opt.Adam(model, lr=0.0002)
        train(model, X_train, Y_train, optimizer, steps=1000, loss_function=sparse_categorical_cross_entropy)
        assert evaluate(model, X_test, Y_test) > 0.95

    def test_simple_sgd(self):
        np.random.seed(1)
        model = SimpleNet()
        optimizer = opt.SGD(model, lr=0.001)
        train(model, X_train, Y_train, optimizer, steps=1000, loss_function=sparse_categorical_cross_entropy)
        assert evaluate(model, X_test, Y_test) > 0.95

    def test_simple_rmsprop(self):
        np.random.seed(1)
        model = SimpleNet()
        optimizer = opt.RMSprop(model, lr=0.0002)
        train(model, X_train, Y_train, optimizer, steps=1000, loss_function=sparse_categorical_cross_entropy)
        assert evaluate(model, X_test, Y_test) > 0.95

    def test_linear_adam(self):
        np.random.seed(1)
        model = LinearNet()
        optimizer = opt.Adam(model, lr=0.001)
        train(model, X_train, Y_train, optimizer, steps=1000, loss_function=sparse_categorical_cross_entropy)
        assert evaluate(model, X_test, Y_test) > 0.95

    def test_linear_sgd(self):
        np.random.seed(1)
        model = LinearNet()
        optimizer = opt.SGD(model, lr=0.001)
        train(model, X_train, Y_train, optimizer, steps=1000, loss_function=sparse_categorical_cross_entropy)
        assert evaluate(model, X_test, Y_test) > 0.95

    def test_linear_rmsprop(self):
        np.random.seed(1)
        model = LinearNet()
        optimizer = opt.RMSprop(model, lr=0.001)
        train(model, X_train, Y_train, optimizer, steps=1000, loss_function=sparse_categorical_cross_entropy)
        assert evaluate(model, X_test, Y_test) > 0.95
