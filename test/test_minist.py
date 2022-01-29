# -*- coding: utf-8 -*- 
# Time: 2022-01-20 10:00
# Copyright (c) 2022
# author: Euraxluo


from core.tensor import Tensor
from core import optimizer as opt

import unittest
import numpy as np
from dataset.minist import fetch_mnist
from tqdm import trange

# load the mnist dataset
X_train, Y_train, X_test, Y_test = fetch_mnist()


# create a model
class TinyBobNet:

    def __init__(self):
        self.l1 = Tensor.uniform(784, 128)
        self.l2 = Tensor.uniform(128, 10)

    def parameters(self):
        return get_parameters(self)

    def forward(self, x):
        return x.dot(self.l1).relu().dot(self.l2).logsoftmax()


# create a model with a conv layer
class TinyConvNet:
    def __init__(self):
        # https://keras.io/examples/vision/mnist_convnet/
        conv = 3
        # inter_chan, out_chan = 32, 64
        inter_chan, out_chan = 8, 16  # for speed
        self.c1 = Tensor.uniform(inter_chan, 1, conv, conv)
        self.c2 = Tensor.uniform(out_chan, inter_chan, conv, conv)
        self.l1 = Tensor.uniform(out_chan * 5 * 5, 10)

    def parameters(self):
        return get_parameters(self)

    def forward(self, x):
        x = x.reshape(shape=(-1, 1, 28, 28))  # hacks
        x = x.conv2d(self.c1).relu().max_pool2d()
        x = x.conv2d(self.c2).relu().max_pool2d()
        x = x.reshape(shape=[x.shape[0], -1])
        return x.dot(self.l1).logsoftmax()


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


# create a model
class TinyMyNet:
    def __init__(self):
        self.fc1 = Linear(28 * 28, 64)
        # self.fc2 = Linear(256, 64)
        self.fc3 = Linear(64, 10)

        # self.w1 = Tensor(layer_init_uniform(28 * 28, 128))
        # self.b1 = Tensor.zeros(128)
        # self.w2 = Tensor(layer_init_uniform(128, 10))
        # self.b2 = Tensor.zeros(10)

    def forward(self, x):
        x = self.fc1(x).relu()
        # x = self.fc2(x).relu()
        x = self.fc3(x).logsoftmax()
        return x


def get_parameters(obj):
    parameters = []
    if isinstance(obj, Tensor):
        parameters.append(obj)
    elif isinstance(obj, list):
        for x in obj:
            parameters.extend(get_parameters(x))
    elif hasattr(obj, '__dict__'):
        for v in obj.__dict__.values():
            parameters.extend(get_parameters(v))
    return parameters


def sparse_categorical_crossentropy(out, Y):
    num_classes = out.shape[-1]
    YY = Y.flatten()
    y = np.zeros((YY.shape[0], num_classes), np.float32)
    # correct loss for NLL, torch NLL loss returns one per row
    y[range(y.shape[0]), YY] = -1.0 * num_classes
    y = y.reshape(list(Y.shape) + [num_classes])
    y = Tensor(y)
    return out.mul(y).mean()


def train(model, X_train, Y_train, optim, steps, BS=128, lossfn=sparse_categorical_crossentropy,
          transform=lambda x: x, target_transform=lambda x: x):
    import time
    losses, accuracies = [], []
    start = time.perf_counter()
    Tensor.training = True
    losses, accuracies = [], []
    for i in (t := trange(steps)):
        samp = np.random.randint(0, X_train.shape[0], size=(BS))
        x = Tensor(transform(X_train[samp]))
        y = target_transform(Y_train[samp])

        # network
        out = model.forward(x)

        loss = lossfn(out, y)
        optim.zero_grad()
        loss.backward()
        optim.step()

        cat = np.argmax(out.data, axis=-1)
        accuracy = (cat == y).mean()

        # printing
        loss = loss.data
        losses.append(loss)
        accuracies.append(accuracy)
        t.set_description("loss %.2f accuracy %.2f" % (loss, accuracy))
        t.set_description("model:%s step %d loss %.2f accuracy %.2f" % (str(model), i, loss, accuracy))
    print(time.perf_counter() - start)


def evaluate(model, X_test, Y_test, num_classes=None, BS=128, return_predict=False, transform=lambda x: x,
             target_transform=lambda y: y):
    Tensor.training = False

    def numpy_eval(Y_test, num_classes):
        Y_test_preds_out = np.zeros(list(Y_test.shape) + [num_classes])
        for i in trange((len(Y_test) - 1) // BS + 1):
            x = Tensor(transform(X_test[i * BS:(i + 1) * BS]))
            Y_test_preds_out[i * BS:(i + 1) * BS] = model.forward(x).data
        Y_test_preds = np.argmax(Y_test_preds_out, axis=-1)
        Y_test = target_transform(Y_test)
        return (Y_test == Y_test_preds).mean(), Y_test_preds

    if num_classes is None: num_classes = Y_test.max().astype(int) + 1
    acc, Y_test_pred = numpy_eval(Y_test, num_classes)
    print("test set accuracy is %f" % acc)
    return (acc, Y_test_pred) if return_predict else acc


class TestMNIST(unittest.TestCase):
    def test_conv(self):
        np.random.seed(1337)
        model = TinyConvNet()
        optimizer = opt.Adam(model.parameters(), lr=0.001)
        train(model, X_train, Y_train, optimizer, steps=200)
        assert evaluate(model, X_test, Y_test) > 0.95

    def test_sgd(self):
        np.random.seed(1337)
        model = TinyBobNet()
        optimizer = opt.SGD(model.parameters(), lr=0.001)
        train(model, X_train, Y_train, optimizer, steps=1000)
        assert evaluate(model, X_test, Y_test) > 0.95

    def test_rmsprop(self):
        np.random.seed(1337)
        model = TinyBobNet()
        optimizer = opt.RMSprop(model.parameters(), lr=0.0002)
        train(model, X_train, Y_train, optimizer, steps=1000)
        assert evaluate(model, X_test, Y_test) > 0.95

    # # @unittest.skip(reason="mad slow")
    # def test_conv(self):
    #     np.random.seed(1337)
    #     model = TinyConvNet()
    #     optimizer = opt.Adam([model.c1, model.l1, model.l2], lr=0.001)
    #     train(model, optimizer, steps=1000)
    #     evaluate(model)
    #
    # # @unittest.skip(reason="mad slow")
    # def test_sgd(self):
    #     np.random.seed(1337)
    #     model = TinyBobNet()
    #     optimizer = opt.SGD([model.l1, model.l2], lr=0.001)
    #     train(model, optimizer, steps=1000)
    #     evaluate(model)
    #
    # # @unittest.skip(reason="mad slow")
    # def test_cov_sgd(self):
    #     np.random.seed(1337)
    #     model = TinyConvNet()
    #     optimizer = opt.SGD([model.l1, model.l2], lr=0.001)
    #     train(model, optimizer, steps=1000)
    #     evaluate(model)
    #
    # @unittest.skip(reason="mad slow")
    def test_rmsprop(self):
        np.random.seed(1337)
        model = TinyBobNet()
        optimizer = opt.RMSprop([model.l1, model.l2], lr=0.0002)
        # train(model, optimizer, steps=1000)
        # evaluate(model)
        train(model, X_train, Y_train, optimizer, steps=1000)
        assert evaluate(model, X_test, Y_test) > 0.95

    # # @unittest.skip(reason="mad slow")
    def test_my_net(self):
        np.random.seed(1337)
        model = TinyMyNet()
        optimizer = opt.SGD(get_parameters(model), lr=0.001)
        # train(model, optimizer, steps=1000)
        # evaluate(model)
        train(model, X_train, Y_train, optimizer, steps=1000)
        assert evaluate(model, X_test, Y_test) > 0.95


if __name__ == '__main__':
    unittest.main()
