# -*- coding: utf-8 -*- 
# Time: 2022-01-29 17:01
# Copyright (c) 2022
# author: Euraxluo
from abc import abstractmethod
from .tensor import Tensor, Module, Context, Forward, Backward, Layer
from typing import *


# https://www.cnblogs.com/guoyaohua/p/8542554.html

class Optimizer:
    def __init__(self, params: Union[list, Tensor, Backward, Forward, Layer, Context, Module]):
        self.params = Optimizer.get_parameters(params)

    def zero_grad(self):
        for param in self.params:
            param.grad = None

    @staticmethod
    def get_parameters(params):
        parameters = []
        if isinstance(params, Tensor) and params.requires_grad:
            parameters.append(params)
        elif isinstance(params, list):
            for x in params:
                parameters.extend(Optimizer.get_parameters(x))
        elif hasattr(params, '__dict__') and isinstance(params, Module):
            for v in params.__dict__.values():
                parameters.extend(Optimizer.get_parameters(v))
        return parameters

    @abstractmethod
    def step(self):
        """
        使用梯度下降法利用梯度更新进行优化
        :return:
        """
        ...


class SGD(Optimizer):
    def __init__(self, params, lr=0.001):
        super().__init__(params)
        self.lr = lr

    def step(self):
        for t in self.params:
            t -= t.grad * self.lr


class RMSprop(Optimizer):
    def __init__(self, params, lr=0.001, decay=0.9, eps=1e-8):
        super().__init__(params)
        self.lr, self.decay, self.eps = lr, decay, eps

        self.v = [Tensor.zeros(*t.shape, requires_grad=False) for t in self.params]

    def step(self):
        for i, t in enumerate(self.params):
            self.v[i] = self.decay * self.v[i] + (1.0 - self.decay) * t.grad * t.grad
            t -= (t.grad * self.lr).div(self.v[i].sqrt() + self.eps)


class Adam(Optimizer):
    def __init__(self, params, lr=0.001, b1=0.9, b2=0.999, eps=1e-8):
        super().__init__(params)
        self.lr, self.b1, self.b2, self.eps, self.t = lr, b1, b2, eps, 0

        self.m = [Tensor.zeros(*t.shape, requires_grad=False) for t in self.params]
        self.v = [Tensor.zeros(*t.shape, requires_grad=False) for t in self.params]

    def step(self):
        self.t = self.t + 1
        a = self.lr * ((1.0 - self.b2 ** self.t) ** 0.5) / (1.0 - self.b1 ** self.t)
        for i, t in enumerate(self.params):
            self.m[i] = self.b1 * self.m[i] + (1.0 - self.b1) * t.grad
            self.v[i] = self.b2 * self.v[i] + (1.0 - self.b2) * t.grad * t.grad
            t -= a * self.m[i].div(self.v[i].sqrt() + self.eps)
