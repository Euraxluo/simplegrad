# -*- coding: utf-8 -*- 
# Time: 2022-01-22 13:43
# Copyright (c) 2022
# author: Euraxluo


import timeit
import torch
from torch.nn import functional
import itertools
from unittest import skip
from unittest import TestCase
import os

os.environ['GRAPH'] = "true"

from .tensor import *


def op_test(shapes, torch_function, test_function, values=None, atol=1e-6, rtol=1e-6, backward=True, grad_atol=1e-6, grad_rtol=1e-6, a=-0.5, b=20):
    torch.manual_seed(0)
    if shapes is None:
        tensor_list = [torch.tensor(x, requires_grad=True) for x in values]
    else:
        tensor_list = [torch.tensor((np.random.random(size=x).astype(np.float32) + a) * b, requires_grad=True) for x in shapes]

    my_tensor_list = [Tensor(x.detach().numpy()) for x in tensor_list]
    # 公差比较
    standard = torch_function(*tensor_list)
    test_result = test_function(*my_tensor_list)
    np.testing.assert_allclose(test_result.data, standard.detach().numpy(), atol=atol, rtol=rtol)
    # 反向传播公差比较
    if backward:
        standard.mean().backward()
        test_result.mean().backward()

        for tensor, my_tensor in zip(tensor_list, my_tensor_list):
            np.testing.assert_allclose(tensor.grad, my_tensor.grad.data, atol=grad_atol, rtol=grad_rtol)
    # 时间比较
    standard = timeit.Timer(functools.partial(torch_function, *tensor_list)).timeit(5) * 1000 / 5
    test_result = timeit.Timer(functools.partial(test_function, *my_tensor_list)).timeit(5) * 1000 / 5
    print(f"testing {shapes}   standard/test  {standard} /{test_result} ms")
    # 反向传播时间比较
    if backward:
        standard = timeit.Timer(functools.partial(lambda f, x: f(*x).mean().backward(), torch_function, tensor_list)).timeit(5) * 1000 / 5
        test_result = timeit.Timer(functools.partial(lambda f, x: f(*x).mean().backward(), test_function, my_tensor_list)).timeit(5) * 1000 / 5
        print(f"testing backward {shapes}   standard/test  {standard} /{test_result} ms")


class TestOps(TestCase):
    def test_assert_all_close(self):
        np.testing.assert_allclose(1, 1.000001, atol=1e-6, rtol=1e-6)

    def test_dtype(self):
        x = Tensor.uniform(4, 3, 6, 6)
        assert x.dtype == np.float32
        x.astype(np.float16)
        assert x.dtype == np.float16

    def test_detach(self):
        op_test([(4, 3, 6, 6)], lambda x: x.detach(), lambda x: x.detach(), backward=False)

    def test_cat(self):
        for dim in range(-1, 2):
            op_test([(45, 65), (45, 65)], lambda x, y: torch.cat((x, y), dim), lambda x, y: x.cat(y, dim))

    def test_reshape(self):
        op_test([(4, 3, 6, 6)], lambda x: torch.reshape(x, (-1, 3, 6, 6)), lambda x: x.reshape(shape=(-1, 3, 6, 6)))
        op_test([(4, 3, 6, 6)], lambda x: torch.reshape(x, (-1, 1, 6, 6)), lambda x: x.reshape(shape=(-1, 1, 6, 6)))

    def test_transpose(self):
        op_test([(3, 3, 3)], lambda x: x.transpose(1, 2), lambda x: x.transpose(axes=(0, 2, 1)))
        op_test([(21, 22, 23, 24)], lambda x: x.movedim((3, 0, 2, 1), (0, 1, 2, 3)), lambda x: x.transpose(axes=(3, 0, 2, 1)))
        op_test([(3, 4, 5, 6)], lambda x: x.movedim((3, 2, 1, 0), (0, 1, 2, 3)), lambda x: x.transpose(axes=(3, 2, 1, 0)))

    def test_slice(self):
        op_test([(3, 3, 3, 3)], lambda x: x[1:2], lambda x: x[1:2])
        op_test([(3, 3, 3, 3)], lambda x: x[1:2, 1:2], lambda x: x[1:2, 1:2])
        op_test([(3, 3, 3, 3)], lambda x: x[1:2, 1:2, 0:-1], lambda x: x[1:2, 1:2, 0:-1])

    def test_add(self):
        op_test([(45, 65), (45, 65)], lambda x, y: x + y, Tensor.add)
        op_test([(45, 65), (45, 65)], lambda x, y: x + y + 0.7978845608, lambda x, y: x + y + 0.7978845608)
        op_test([(45, 65), (45, 65)], lambda x, y: 0.7978845608 + y + x, lambda x, y: 0.7978845608 + y + x)

    def test_sub(self):
        op_test([(45, 65), (45, 65)], lambda x, y: x - y, Tensor.sub)
        op_test([(45, 65), (45, 65)], lambda x, y: x - y - 0.7978845608, lambda x, y: x - y - 0.7978845608)
        op_test([(45, 65), (45, 65)], lambda x, y: 0.7978845608 - y, lambda x, y: 0.7978845608 - y)
        op_test([(45, 65)], lambda x: x - 2, lambda x: x - 2)
        op_test([(45, 65)], lambda x: 2 - x, lambda x: 2 - x)

    def test_mul(self):
        op_test([(45, 65), (45, 65)], lambda x, y: x * y, Tensor.mul)
        op_test([(45, 65), (45, 65)], lambda x, y: x * y * 0.7978845608, lambda x, y: x * y * 0.7978845608)
        op_test([(45, 65), (45, 65)], lambda x, y: 0.7978845608 * y * x, lambda x, y: 0.7978845608 * y * x)
        op_test([(45, 65)], lambda x: x * 2, lambda x: x * 2)
        op_test([(45, 65)], lambda x: 2 * x, lambda x: 2 * x)

    def test_div(self):
        op_test([(45, 65), (45, 65)], lambda x, y: x / y, Tensor.div)
        op_test([(45, 65), (45, 65)], lambda x, y: x / y / 0.7978845608, lambda x, y: x / y / 0.7978845608)
        op_test([(45, 65), (45, 65)], lambda x, y: 0.7978845608 / x / y, lambda x, y: 0.7978845608 / x / y)

    def test_pow(self):
        op_test([(45, 65), (45, 65)], lambda x, y: x ** y, Tensor.pow, a=0)
        op_test([(45, 65), (45, 65)], lambda x, y: torch.abs(x) ** torch.abs(y) ** 0.7978845608, lambda x, y: x.abs() ** y.abs() ** 0.7978845608, rtol=1e-5)
        op_test([(45, 65), (45, 65)], lambda x, y: 0.7978845608 ** torch.abs(y) ** torch.abs(x), lambda x, y: 0.7978845608 ** y.abs() ** x.abs())

    def test_dot(self):
        op_test([(45, 65), (65, 100)], lambda x, y: x.matmul(y), Tensor.matmul, atol=1e-4, rtol=1e-4)
        op_test([(10, 45, 65), (65, 45)], lambda x, y: x @ y, lambda x, y: x @ y, atol=1e-4, rtol=1e-4)
        op_test([(3, 3, 45, 65), (3, 3, 65, 45)], lambda x, y: x @ y, lambda x, y: x @ y, atol=1e-4, rtol=1e-4)
        op_test([(10, 45, 65), (10, 65, 45)], lambda x, y: x @ y, lambda x, y: x @ y, atol=1e-4, rtol=1e-4)

    def test_mixed_operation(self):
        op_test([(45, 65)], lambda x: (x + x) * x, lambda x: x.add(x).mul(x))

        for torch_function, test_function in [(torch.add, Tensor.add),
                                              (torch.sub, Tensor.sub),
                                              (torch.mul, Tensor.mul),
                                              (torch.div, Tensor.div),
                                              (torch.pow, Tensor.pow)]:
            for shapes in [((1, 32, 32, 32), (1, 32, 1, 1)), ((5, 13, 24, 16, 2), (1, 13, 24, 1, 1)),
                           ((4, 1), (4, 5)), ((1, 4), (5, 4)), ((5, 13, 24, 16), (5, 1, 24, 1)), ((1, 3, 1, 7, 1), (2, 1, 5, 1, 8))]:
                with self.subTest(op=torch_function.__name__, shapes=shapes):
                    op_test(shapes, torch_function, test_function, a=-0.5 if test_function != Tensor.pow else 0.0)

    def test_sum(self):
        op_test([(45, 3)], lambda x: x.sum(), Tensor.sum)
        op_test([(3, 4, 5, 6)], lambda x: x.sum(axis=3), lambda x: Tensor.sum(x, axis=3))
        op_test([(3, 4, 5, 6)], lambda x: x.sum(axis=(1, 3)), lambda x: Tensor.sum(x, axis=(1, 3)))
        op_test([(3, 4, 5, 6)], lambda x: x.sum(axis=(0, 2)), lambda x: Tensor.sum(x, axis=(0, 2)))
        op_test([(3, 4, 5, 6)], lambda x: x.sum(axis=(1, 2)), lambda x: Tensor.sum(x, axis=(1, 2)))
        op_test([(3, 4, 5, 6)], lambda x: x.sum(axis=1), lambda x: Tensor.sum(x, axis=1))

    def test_max(self):
        op_test([(45, 3)], lambda x: x.max(), Tensor.max)
        op_test([(45, 3)], lambda x: x.max().mul(0.5), lambda x: Tensor.max(x).mul(0.5))
        op_test([(3, 4, 5, 6)], lambda x: x.max(axis=1)[0], lambda x: Tensor.max(x, axis=1))
        op_test(None, lambda x: x.max().mul(0.5), lambda x: Tensor.max(x).mul(0.5), values=[[[1.0, 1.0, 0.0, 1.0]]])

    def test_mean(self):
        op_test([(3, 4, 5, 6)], lambda x: x.mean(axis=(1, 2)), lambda x: Tensor.mean(x, axis=(1, 2)))

    def test_log(self):
        op_test([(45, 65)], lambda x: torch.log(x.abs()), lambda x: x.abs().log())

    def test_exp(self):
        op_test([(45, 65)], lambda x: torch.exp(x), Tensor.exp)

    def test_relu(self):
        op_test([(45, 65)], lambda x: x.relu(), Tensor.relu)
        op_test([(3, 3, 45, 65)], lambda x: x.relu(), Tensor.relu)

    def test_abs(self):
        op_test([(45, 65)], lambda x: torch.abs(x), Tensor.abs)

    def test_sign(self):
        op_test([(45, 65)], lambda x: torch.sign(x), Tensor.sign)

    def test_sqrt(self):
        op_test([(45, 65)], lambda x: x.sqrt(), Tensor.sqrt, a=0)

    def test_sigmoid(self):
        op_test([(45, 65)], lambda x: x.sigmoid(), Tensor.sigmoid)

    def test_swish(self):
        op_test([(45, 65)], lambda x: x * x.sigmoid(), Tensor.swish)

    def test_leaky_relu(self):
        op_test([(45, 65)], lambda x: torch.nn.functional.leaky_relu(x, negative_slope=0.01), Tensor.leaky_relu)

    def test_relu6(self):
        op_test([(45, 65)], lambda x: torch.nn.functional.relu6(x), Tensor.relu6)

    def test_hardswish(self):
        op_test([(45, 65)], lambda x: torch.nn.functional.hardswish(x), Tensor.hardswish)

    def test_tanh(self):
        op_test([(45, 65)], lambda x: x.tanh(), Tensor.tanh)

    def test_gelu(self):
        op_test([(45, 65)], lambda x: 0.5 * x * (1.0 + torch.tanh(x * 0.7978845608 * (1.0 + 0.044715 * x * x))), Tensor.gelu)

    def test_softplus(self):
        op_test([(45, 65)], lambda x: torch.nn.functional.softplus(x), Tensor.softplus)

    def test_mish(self):
        op_test([(45, 65)], lambda x: x * torch.tanh(torch.nn.functional.softplus(x)), Tensor.mish, atol=1e-4, rtol=1e-4)

    def test_softmax(self):
        op_test([(45, 65)], lambda x: torch.nn.functional.softmax(x, dim=1), Tensor.softmax)

    def test_logsoftmax(self):
        op_test([(45, 65)], lambda x: torch.nn.functional.log_softmax(x, dim=1), Tensor.logsoftmax)

    def test_dropout(self):
        Tensor.training = True
        n, rate = 1_000_000, 0.1
        w = Tensor.ones(n).dropout(rate)
        non_zeros = np.count_nonzero(w.data)
        expected = n * (1 - rate)
        np.testing.assert_allclose(non_zeros, expected, rtol=1e-4)

    def test_pad(self):
        op_test([(3, 3, 3, 3)], lambda x: torch.nn.functional.pad(x, (1, 2, 3, 4)), lambda x: x.pad(padding=(1, 2, 3, 4)))
        op_test([(2, 3, 1)], lambda x: torch.nn.functional.pad(x, (1, 1, 1, 1)), lambda x: x.pad(padding=(1, 1, 1, 1)))
        op_test([(1, 1)], lambda x: torch.nn.functional.pad(x, (1, 1, 1, 1)), lambda x: x.pad(padding=(1, 1, 1, 1)))

    def test_maxpool2d(self):
        for ksz in [(2, 2), (3, 3), (3, 2), (5, 5), (5, 1)]:
            with self.subTest(kernel_size=ksz):
                op_test([(32, 2, 110, 28)],
                        lambda x: torch.nn.functional.max_pool2d(x, kernel_size=ksz),
                        lambda x: Tensor.max_pool2d(x, kernel_size=ksz))

    def test_avgpool2d(self):
        shape = (32, 2, 111, 28)
        for ksz in [(2, 2), (3, 3), (3, 2), (5, 5), (5, 1), shape[2:]]:
            with self.subTest(kernel_size=ksz):
                op_test([shape],
                        lambda x: torch.nn.functional.avg_pool2d(x, kernel_size=ksz),
                        lambda x: Tensor.avg_pool2d(x, kernel_size=ksz), rtol=1e-5)

    def test_conv2d(self):
        for bs in [1, 8]:
            for cin in [1, 3]:
                for groups in [1, 3] if cin == 3 else [1]:
                    for H in [1, 2, 5]:
                        for W in [1, 2, 3, 5]:
                            with self.subTest(batch_size=bs, channels=cin, groups=groups, height=H, width=W):
                                op_test([(bs, cin, 11, 28), (6, cin // groups, H, W)],
                                        lambda x, w: torch.nn.functional.conv2d(x, w, groups=groups).relu(),
                                        lambda x, w: Tensor.conv2d(x, w, groups=groups).relu(), atol=1e-4, grad_rtol=1e-5)

    def test_large_input_conv2d(self):
        bs = 4
        cin = 16
        groups = 1
        H = 5
        W = 2
        op_test([(bs, cin, 64, 64), (6, cin // groups, H, W)],
                lambda x, w: torch.nn.functional.conv2d(x, w, groups=groups).relu(),
                lambda x, w: Tensor.conv2d(x, w, groups=groups).relu(), atol=1e-4, grad_rtol=1e-4)

    def test_grouped_conv2d(self):
        groups = 2
        op_test([(1, 2, 5, 5), (groups, 1, 3, 3)],
                lambda x, w: torch.nn.functional.conv2d(x, w, groups=groups).relu(),
                lambda x, w: Tensor.conv2d(x, w, groups=groups).relu(), atol=1e-4, grad_rtol=1e-5, backward=False)

    def test_fancy_conv2d(self):
        bs = 2
        cin = 3
        cout = 1
        groups = 3
        H, W = 3, 3
        op_test([(bs, cin, 11, 28), (groups * cout, cin // groups, H, W)],
                lambda x, w: torch.nn.functional.conv2d(x, w, groups=groups).relu(),
                lambda x, w: Tensor.conv2d(x, w, groups=groups).relu(), atol=1e-4, grad_rtol=1e-5)

    def test_strided_conv2d(self):
        bs = 4
        cin = 3
        H, W = 3, 3
        with self.subTest(stride := 2):
            op_test([(bs, cin, 11, 28), (4, cin, H, W)],
                    lambda x, w: torch.nn.functional.conv2d(x, w, stride=2).relu(),
                    lambda x, w: Tensor.conv2d(x, w, stride=stride).relu(), atol=1e-4)
        with self.subTest(stride := (2, 1)):
            op_test([(bs, cin, 11, 28), (4, cin, H, W)],
                    lambda x, w: torch.nn.functional.conv2d(x, w, stride=stride).relu(),
                    lambda x, w: Tensor.conv2d(x, w, stride=(2, 1)).relu(), atol=1e-4)
