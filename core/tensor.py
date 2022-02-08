# -*- coding: utf-8 -*- 
# Time: 2022-01-20 17:27
# Copyright (c) 2022
# author: Euraxluo


import abc
import builtins
import numpy as np
import inspect
import functools
from abc import ABCMeta, abstractmethod
from typing import *
from .logger import *

Number = Union[builtins.int, builtins.float, builtins.bool]


@runtime_checkable
class Module(Protocol):
    """
    一种协议,即实现了"forward", "backward", "__call__" 任意一个函数的类,可视为Module类
    Module类,主要用于Optimizer 的 params解析
    """

    __slots__ = ()

    def forward(self, *args, **kwargs):
        ...

    def backward(self, *args, **kwargs):
        ...

    def __call__(self, *args, **kwargs):
        ...

    @staticmethod
    def _check_methods(C, *methods):
        mro = C.__mro__
        for B in mro:
            for method in methods:
                if method in B.__dict__:
                    if B.__dict__[method] is None:
                        return NotImplemented
                    break
            else:
                break
        else:
            return NotImplemented
        return True

    @classmethod
    def __subclasshook__(cls, C):
        if cls is Module:
            return Module._check_methods(C, "forward", "backward", "__call__")
        return NotImplemented


class Forward(Protocol):
    """
    一种协议,即实现了"forward" 函数的类,可视为Forward,多为Network类
    """

    def forward(self, *args, **kwargs):
        ...


class Backward(Protocol):
    """
    一种协议,即实现了"backward" 函数的类,可视为Backward,多为Tensor类
    """

    def backward(self, *args, **kwargs):
        ...


class Context(Protocol):
    """
    一种协议,即实现了"forward" 和 "backward"类,即Function的实例化对象,可视为Context类
    """

    def forward(self, *args, **kwargs):
        ...

    def backward(self, *args, **kwargs):
        ...


class Layer(Protocol):
    """
    一种协议,即实现了"__call__" 函数的类,可视为Layer类
    """

    def __call__(self, *args, **kwargs):
        ...


class Function:
    """
    An instantiation of the Function is the `Context` class
    """
    name: str = 'Function'

    def __init__(self, *tensors: 'Tensor'):
        self.parents = tensors
        self.saved_tensors = []
        self.requires_grad = any([t.requires_grad for t in tensors])

    def save_for_backward(self, *x):
        if self.requires_grad:
            self.saved_tensors.extend(x)

    @classmethod
    def register(cls, name: str, context: Type['Function']):
        if cls == Function:
            # 启动调用逻辑
            for subclass in cls.__subclasses__():
                subclass.register(subclass.name.lower(), subclass)

        if issubclass(cls, Function):
            # 启动子类逻辑
            if getattr(Tensor, name, None) is not None:
                setattr(Tensor, "_" + name, context.apply())
            else:
                setattr(Tensor, name, context.apply())

    @classmethod
    def apply(cls: Type['Function']) -> Callable:
        def wrapper(*x: 'Tensor', **kwargs):
            tensor_args = [arg for arg in x if isinstance(arg, Tensor)]
            x = [Tensor(np.array([arg], dtype=tensor_args[0].dtype if len(tensor_args) > 0 else None), requires_grad=False) if not isinstance(arg, Tensor) else arg for arg in x]
            ctx = cls(*x)
            # 先使用默认参数
            params = inspect.signature(cls.forward).parameters
            for p in params.values():
                if p.default is not p.empty:
                    setattr(ctx, p.name, p.default)

            # 覆盖默认参数
            for k, v in kwargs.items():
                setattr(ctx, k, v)

            # forward
            with profile.log(ctx, ctx.name, x) as p:
                tensor_forward = Tensor(cls.forward(ctx, *[t.data for t in x], **kwargs), requires_grad=any([t.requires_grad for t in x]))
                p.output = [tensor_forward]
            if tensor_forward.requires_grad:
                tensor_forward._ctx = ctx
            return tensor_forward

        return wrapper

    @abc.abstractmethod
    def forward(self: 'Function', *arg, **kwargs):
        ...

    @abc.abstractmethod
    def backward(self: 'Function', *arg, **kwargs):
        ...

    def __init_subclass__(cls, **kwargs):
        cls.name = cls.__name__
        if hasattr(cls, 'backward'):
            cls.forward = staticmethod(cls.forward)
        if hasattr(cls, 'backward'):
            cls.backward = staticmethod(cls.backward)


from .ops import *


class Tensor(object):
    training = False

    def __init__(self, data: Union[np.ndarray, np.float], requires_grad=True):
        self.data: Union[np.ndarray, np.float] = data
        self.grad = None
        self.requires_grad = requires_grad
        self._ctx = None
        Function.register(name=Function.name.lower(), context=Function)

    def __repr__(self):
        return f"<Tensor {self.data!r} with grad {(self.grad.data if self.grad else None)!r}>"

    @property
    def ctx(self):
        return self._ctx

    @property
    def shape(self):
        return self.data.shape

    @property
    def dtype(self):
        return self.data.dtype

    def astype(self, t) -> 'Tensor':
        return Tensor(self.data.astype(t))

    def assign(self, v):
        if not isinstance(v, Tensor):
            v = Tensor(v)
        self.data = v.data

    def detach(self):
        return Tensor(self.data, requires_grad=False)

    def cat(self, y, dim=0):
        if self.shape != y.shape:
            raise ValueError(f'shape must match self shape in {self.data!r}, {self.shape!r} != {y.shape!r}')
        dim = (dim + len(self.shape)) if dim < 0 else dim
        s1, s2 = [], []
        for i in range(len(self.shape)):
            if i != dim:
                if self.shape[i] != y.shape[i]:
                    raise ValueError(f'shape must equal {y.shape[i]!r} != {y.shape[i]!r}')
                s1.append((0, self.shape[i]))
                s2.append((0, self.shape[i]))
            else:
                s1.append((0, self.shape[i] + y.shape[i]))
                s2.append((-self.shape[i], y.shape[i]))
        return self.slice(arg=s1) + y.slice(arg=s2)

    def reshape(self, shape: Tuple):
        return self._reshape(shape=shape)

    def transpose(self, axes: Union[Tuple, List]):
        return self._transpose(axes=axes)

    def slice(self, arg):
        return self._slice(arg=arg)

    # getitem
    def __getitem__(self, val):
        """
        具备自动填充功能
        :param val:
        :return:
        """
        arg = []
        new_shape = []
        if val is not None:
            for i, s in enumerate(val if isinstance(val, (list, tuple)) else [val]):
                if isinstance(s, int):
                    arg.append((s, s + 1))
                else:
                    arg.append((s.start if s.start is not None else 0,
                                (s.stop if s.stop >= 0 else self.shape[i] + s.stop) if s.stop is not None else self.shape[i]))
                    new_shape.append(arg[-1][1] - arg[-1][0])
        new_shape += self.shape[len(arg):]
        return self.slice(arg=arg + [(0, self.shape[i]) for i in range(len(arg), len(self.shape))]).reshape(shape=new_shape)

    # backward
    def deep_walk(self: 'Tensor') -> List['Tensor']:
        """
        网络结构获取

        :return:
        """

        def _deep_walk(node: 'Tensor', visited: set, nodes: List['Tensor']) -> List['Tensor']:
            visited.add(node)
            if node.ctx:
                [_deep_walk(i, visited, nodes) for i in node.ctx.parents if i not in visited]
                nodes.append(node)
            return nodes

        return _deep_walk(self, set(), [])

    def backward(self):
        """
        隐式创建梯度

        :return: self.grad
        """
        self.grad = Tensor(np.ones(self.shape, dtype=self.dtype), requires_grad=False)
        for tensor in reversed(self.deep_walk()):
            if not any([x.requires_grad for x in tensor._ctx.parents]):
                continue
            with profile.log(tensor._ctx, tensor._ctx.__class__.__name__, [tensor.grad], backward=True) as p:
                grads = tensor._ctx.backward(tensor._ctx, tensor.grad.data)
                new_grads = []
                for g in ([grads] if len(tensor._ctx.parents) == 1 else grads):
                    if g is not None:
                        new_grads.append(Tensor(g, requires_grad=False))
                    else:
                        new_grads.append(None)
                p.output = grads = new_grads

            for t, g in zip(tensor._ctx.parents, grads):
                if g is not None and t.requires_grad:
                    if g.shape != t.shape:
                        raise ValueError(f'grad shape must match self shape in {self._ctx!r}, {g.shape!r} != {t.shape!r}')
                    t.grad = g if t.grad is None else (t.grad + g)
        return self.grad

    # helpers

    @classmethod
    def zeros(cls, *shape, **kwargs):
        return cls(np.zeros(shape, dtype=np.float32), **kwargs)

    @classmethod
    def ones(cls, *shape, **kwargs):
        return cls(np.ones(shape, dtype=np.float32), **kwargs)

    @classmethod
    def randn(cls, *shape, **kwargs):
        return cls(np.random.randn(*shape).astype(np.float32), **kwargs)

    @classmethod
    def uniform(cls, *shape, **kwargs):
        return cls((np.random.uniform(-1., 1., size=shape) / np.sqrt(np.prod(shape))).astype(np.float32), **kwargs)

    @classmethod
    def eye(cls, *shape, **kwargs):
        return cls(np.eye(*shape, dtype=np.float32), **kwargs)

    @classmethod
    def arange(cls, stop, start: Union[int, float, complex, None] = 0, **kwargs):
        return cls(np.arange(start=start, stop=stop).astype(np.float32), **kwargs)

    # add
    def add(self, other):
        return self._add(other)

    def __add__(self, other):
        return self.add(other)

    def __iadd__(self, other):
        return self.assign(self.add(other))

    def __radd__(self, other):
        return Tensor._add(other, self)

    # sub
    def sub(self, other):
        return self._sub(other)

    def __sub__(self, other):
        return self.sub(other)

    def __isub__(self, other):
        return self.assign(self.sub(other))

    def __rsub__(self, other):
        return Tensor._sub(other, self)

    # mul
    def mul(self, other):
        return self._mul(other)

    def __mul__(self, other):
        return self.mul(other)

    def __imul__(self, other):
        return self.assign(self.mul(other))

    def __rmul__(self, other):
        return Tensor._mul(other, self)

    # div
    def div(self, other):
        return self * (other ** -1.0)

    def __truediv__(self, other):
        return self.div(other)

    def __idiv__(self, other):
        return self.assign(self.div(other))

    def __rtruediv__(self, other):
        return Tensor.div(other, self)

    # pow
    def pow(self, other):
        return self._pow(other)

    def __pow__(self, other):
        return self.pow(other)

    def __ipow__(self, other):
        return self.assign(self.pow(other))

    def __rpow__(self, other):
        return Tensor._pow(other, self)

    # dot
    def matmul(self, other):
        return self._matmul(other)

    def dot(self, other):
        return self.matmul(other)

    def __matmul__(self, other):
        return self.matmul(other)

    def __imatmul__(self, other):
        return self.assign(self.matmul(other))

    def __rmatmul__(self, other):
        return Tensor._matmul(other, self)

    # reduce
    def _canonicalize_reduce_axis(self, axis):
        if axis is None: axis = range(len(self.shape))
        if isinstance(axis, int): axis = [axis]
        axis = tuple([x if x >= 0 else x + len(self.shape) for x in axis])
        shape = [self.shape[i] for i in range(len(self.shape)) if i not in axis]
        shape = [1] if shape == [] else shape
        return axis, shape

    def sum(self, axis=None, keepdim=False):
        axis, out_shape = self._canonicalize_reduce_axis(axis)
        ret = self._sum(axis=axis)
        return ret if keepdim or ret.shape == out_shape else ret.reshape(shape=out_shape)

    def max(self, axis=None, keepdim=False):
        axis, out_shape = self._canonicalize_reduce_axis(axis)
        ret = self._max(axis=axis)
        return ret if keepdim or ret.shape == out_shape else ret.reshape(shape=out_shape)

    def mean(self, axis=None, keepdim=False):
        out = self.sum(axis=axis, keepdim=keepdim)
        return out * (np.prod(out.shape) / np.prod(self.shape))

    # 数学一元操作
    def log(self):
        return self._log()

    def exp(self):
        return self._exp()

    def relu(self):
        return self._relu()

    def abs(self):
        return self.relu() + (-1.0 * self).relu()

    def sign(self):
        return self / (self.abs() + 1e-10)

    def sqrt(self):
        return self.pow(0.5)

    def sigmoid(self):
        return (1.0 + (0.0 - self).exp()) ** -1.0

    def swish(self):
        return self * self.sigmoid()

    def leaky_relu(self, neg_slope=0.01):
        return self.relu() - (-neg_slope * self).relu()

    def relu6(self):
        return self.relu() - (self - 6).relu()

    def hardswish(self):
        return self * (self + 3).relu6() * (1 / 6)

    def tanh(self):
        return 2.0 * ((2.0 * self).sigmoid()) - 1.0

    def gelu(x):
        return 0.5 * x * (1.0 + (x * 0.7978845608 * (1.0 + 0.044715 * x * x)).tanh())

    def softplus(self, limit=20, beta=1):
        eb = (self * beta).exp()
        ret = (1 + eb).log()
        return (1 / beta) * ret

    def mish(self):
        return self * (self.softplus().tanh())

    def softmax(self):
        ns = list(self.shape)[:-1] + [1]
        m = self.max(axis=len(self.shape) - 1).reshape(shape=ns)
        e = (self - m).exp()
        ss = e.sum(axis=len(self.shape) - 1).reshape(shape=ns)
        return e.div(ss)

    def logsoftmax(self):
        ns = list(self.shape)[:-1] + [1]
        m = self.max(axis=len(self.shape) - 1).reshape(shape=ns)
        ss = m + (self - m).exp().sum(axis=len(self.shape) - 1).reshape(shape=ns).log()
        return self - ss

    # 数据操作
    def dropout(self, p=0.5):
        if Tensor.training:
            _mask = np.asarray(np.random.binomial(1, 1.0 - p, size=self.shape), dtype=self.dtype)
            return self * Tensor(_mask, requires_grad=False) * (1 / (1.0 - p))
        else:
            return self

    def pad(self, padding: Union[Tuple, List, int]):
        """
        pad

        :param padding: [pad_left, pad_right, pad_top, pad_bottom]
        :return:
        """
        if isinstance(padding, int):
            padding = [padding] * 4
        if len(self.shape) < 2:
            raise ValueError("shape dimension must greater than 1")
        slice_args = [slice(-padding[2], self.shape[-2] + padding[3], None), slice(-padding[0], self.shape[-1] + padding[1], None)]
        for i in range(2, len(self.shape)):
            slice_args.insert(0, slice(None, None, None))
        return self[slice_args]

    def _pool2d(self, *size):
        if len(self.shape) < 2:
            raise ValueError("shape dimension must greater than 1")
        slice_args = []

        for i, s in enumerate(size[::-1]):
            slice_args.insert(0, slice(None, self.shape[-(i + 1)] - self.shape[-(i + 1)] % s, None))
        for i in range(0, len(self.shape) - len(size)):
            slice_args.insert(0, slice(None, None, None))

        xup = self[slice_args]

        reshape_args = []
        for i, s in enumerate(size[::-1]):
            reshape_args.insert(0, s)
            reshape_args.insert(0, xup.shape[-(i + 1)] // s)
        for i in range(0, len(xup.shape) - len(size)):
            reshape_args.insert(i, xup.shape[i])
        return xup.reshape(shape=reshape_args)

    def avg_pool2d(self, kernel_size=(2, 2)):
        return self._pool2d(*kernel_size).mean(axis=(3, 5))

    def max_pool2d(self, kernel_size=(2, 2)):
        return self._pool2d(*kernel_size).max(axis=(3, 5))

    def conv2d(self, weight, bias=None, stride=1, groups=1):
        ret = self._conv2d(weight, stride=stride, groups=groups)
        return ret if bias is None else ret.add(bias.reshape(shape=[1, -1, 1, 1]))

    # ***** 神经网络操作 *****

    def linear(self, weight, bias):
        shp = [1] * (len(self.shape) - 1) + [-1]
        ret = self.mul(weight.reshape(shape=shp)) if len(weight.shape) == 1 else self.dot(weight)
        return ret.add(bias.reshape(shape=shp))

    def sequential(self, ll):
        for l in ll: self = l(self)
        return self

    def layernorm(x, eps=1e-5):
        y = (x - x.mean(axis=-1, keepdim=True))
        return y.div((y * y).mean(axis=-1, keepdim=True).add(eps).sqrt())
