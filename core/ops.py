# -*- coding: utf-8 -*- 
# Time: 2022-01-21 14:45
# Copyright (c) 2022
# author: Euraxluo

import numpy as np
from typing import *
from .tensor import *


# ************* 简单操作 *************

def unbroadcast(out, in_sh):
    """
    Un Broadcast
    :param out:
    :param in_sh:
    :return:
    """
    if in_sh == (1,):
        return out.sum().reshape((1,))
    else:
        sum_axis = tuple([i for i in range(len(in_sh)) if in_sh[i] == 1 and out.shape[i] > 1])
        return out.sum(axis=sum_axis).reshape(in_sh) if len(sum_axis) > 0 else out


class Add(Function):
    def forward(self, x, y):
        self.save_for_backward(x.shape, y.shape)
        return x + y

    def backward(self, grad_output):
        shape_x, shape_y = self.saved_tensors
        return unbroadcast(grad_output, shape_x), unbroadcast(grad_output, shape_y)


class Sub(Function):
    def forward(self, x, y):
        self.save_for_backward(x.shape, y.shape)
        return x - y

    def backward(self, grad_output):
        shape_x, shape_y = self.saved_tensors
        return unbroadcast(grad_output, shape_x), unbroadcast(-grad_output, shape_y)


class Mul(Function):
    def forward(self, x, y):
        self.save_for_backward(x, y)
        return x * y

    def backward(self, grad_output):
        x, y = self.saved_tensors
        return unbroadcast(y * grad_output, x.shape), unbroadcast(x * grad_output, y.shape)


class Pow(Function):
    def forward(self, x, y):
        self.save_for_backward(x, y)
        return x ** y

    def backward(self, grad_output):
        x, y = self.saved_tensors
        return unbroadcast(y * (x ** (y - 1.0)) * grad_output, x.shape), unbroadcast((x ** y) * np.log(x) * grad_output, y.shape)


# ************* 变换操作 *************

class Reshape(Function):
    def forward(self, x: np.ndarray, shape):
        self.save_for_backward(x.shape)
        return x.reshape(shape)

    def backward(self, grad_output):
        in_shape, = self.saved_tensors
        return grad_output.reshape(in_shape)


class Transpose(Function):
    def forward(self, x: np.ndarray, axes: Union[Tuple, List]):
        self.save_for_backward(axes)
        return np.transpose(x, axes=axes)

    def backward(self, x):
        axes, = self.saved_tensors
        return x.permute(tuple(np.argsort(axes)))


def inner_slice(x, arg):
    padding = [(max(0, -p[0]), max(0, p[1] - x.shape[i])) for i, p in enumerate(arg)]
    x = np.pad(x, padding)
    slice_index = [(p[0] + padding[i][0], p[1] + padding[i][0]) for i, p in enumerate(arg)]
    return x[tuple([slice(i[0], i[1], None) for i in slice_index])]


class Slice(Function):
    def forward(self, x, arg=None):
        self.save_for_backward(x.shape)
        return inner_slice(x, arg)

    def backward(self, grad_output):
        shape, = self.saved_tensors
        narg = [(0 - p[0], grad_output.shape[i] + (shape[i] - p[1])) for i, p in enumerate(self.arg)]
        return inner_slice(grad_output, narg)


# ************* unary ops *************

class ReLU(Function):
    def forward(self, input: np.ndarray):
        self.save_for_backward(input)
        return np.maximum(input.data, 0)

    def backward(self, grad_output):
        input, = self.saved_tensors
        return grad_output * (input >= 0)


class Log(Function):
    def forward(self, input: np.ndarray):
        self.save_for_backward(input)
        return np.log(input)

    def backward(self, grad_output):
        input, = self.saved_tensors
        return grad_output / input


class Exp(Function):
    def forward(self, input: np.ndarray):
        ret = np.exp(input.clip(-88, 88))
        self.save_for_backward(ret)
        return ret

    def backward(self, grad_output):
        ret, = self.saved_tensors
        return grad_output * ret


# ************* reduce ops (with keepdims=True) *************

class Sum(Function):
    def forward(self, input: np.ndarray, axis):
        self.save_for_backward(input.shape)
        return input.sum(axis, keepdims=True)

    def backward(self, grad_output):
        shape_input, = self.saved_tensors
        return np.broadcast_to(grad_output, shape_input)


class Max(Function):
    def forward(self, input: np.ndarray, axis):
        ret = np.amax(input, axis=axis, keepdims=True)
        self.save_for_backward(input, axis, ret)
        return ret

    def backward(self, grad_output):
        input, axis, ret = self.saved_tensors
        ret2 = (input == ret)
        return ret2 * grad_output / np.sum(ret2, axis=tuple(axis), keepdims=True).astype(input.dtype)


# ************* processing ops *************

class Matmul(Function):
    def forward(self, input: np.ndarray, weight):
        self.save_for_backward(input, weight)
        return input @ weight

    def backward(self, grad_output):
        input, weight = self.saved_tensors
        grad_input = grad_output @ weight.swapaxes(-2, -1)
        grad_weight = input.swapaxes(-2, -1) @ grad_output
        return grad_input, grad_weight


# TODO   File "/home/yons/Project/jupyter/NN/simplegrad/core/ops.py", line 201, in forward
#     ret[:, g] += np.tensordot(tx[:, g], tw[g], ((1, 4, 5), (1, 2, 3)))
# numpy.core._exceptions.UFuncTypeError: Cannot cast ufunc 'add' output from dtype('float32') to dtype('uint8') with casting rule 'same_kind'

class Conv2D(Function):
    def forward(self, x, w, stride=1, groups=1):
        if isinstance(self.stride, int):
            self.stride = (self.stride, self.stride)
        cout, cin, H, W = w.shape
        ys, xs = self.stride
        bs, cin_ = x.shape[0], x.shape[1]
        oy, ox = (x.shape[2] - (H - ys)) // ys, (x.shape[3] - (W - xs)) // xs
        assert cin * self.groups == cin_
        assert cout % self.groups == 0
        rcout = cout // self.groups

        gx = x.reshape(bs, self.groups, cin, x.shape[2], x.shape[3])
        tx = np.lib.stride_tricks.as_strided(gx,
                                             shape=(bs, self.groups, cin, oy, ox, H, W),
                                             strides=(*gx.strides[0:3], gx.strides[3] * ys, gx.strides[4] * xs, *gx.strides[3:5]),
                                             writeable=False,
                                             )
        tw = w.reshape(self.groups, rcout, cin, H, W)
        self.save_for_backward(tx, tw, x.shape)

        ret = np.zeros((bs, self.groups, oy, ox, rcout), dtype=x.dtype)
        for g in range(self.groups):
            # ijYXyx,kjyx -> iYXk ->ikYX
            ret[:, g] = ret[:, g] + np.tensordot(tx[:, g], tw[g], ((1, 4, 5), (1, 2, 3)))
        return np.moveaxis(ret, 4, 2).reshape(bs, cout, oy, ox)

    def backward(self, grad_output):
        bs, _, oy, ox = grad_output.shape
        tx, tw, x_shape = self.saved_tensors
        _, rcout, cin, H, W = tw.shape
        ys, xs = self.stride
        OY, OX = x_shape[2:4]

        ggg = grad_output.reshape(bs, self.groups, rcout, oy, ox)

        gdw = np.zeros((self.groups, rcout, cin, H, W), dtype=tx.dtype)
        for g in range(self.groups):
            # 'ikYX,ijYXyx -> kjyx'
            gdw[g] = gdw[g] + np.tensordot(ggg[:, g], tx[:, g], ((0, 2, 3), (0, 2, 3)))

        # needs to be optimized
        gdx = np.zeros((bs, self.groups, cin, OY, OX), dtype=tx.dtype)
        for k in range(oy * ox):
            Y, X = k // ox, k % ox
            iY, iX = Y * ys, X * xs
            # gdx[:,:,: , iY:iY+H, iX:iX+W] += np.einsum('igk,gkjyx->igjyx', ggg[:,:,:,Y,X], tw)
            for g in range(self.groups):
                tg = np.dot(ggg[:, g, :, Y, X].reshape(bs, -1), tw[g].reshape(rcout, -1))
                gdx[:, g, :, iY:iY + H, iX:iX + W] = gdx[:, g, :, iY:iY + H, iX:iX + W] + tg.reshape((bs, cin, H, W))

        return gdx.reshape((bs, self.groups * cin, OY, OX)), gdw.reshape((self.groups * rcout, cin, H, W))
