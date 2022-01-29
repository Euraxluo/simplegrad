# -*- coding: utf-8 -*- 
# Time: 2022-01-21 11:18
# Copyright (c) 2022
# author: Euraxluo

from unittest import TestCase
from .tensor import *


class TestTensor(TestCase):
    def test_tensor(self):
        t = Tensor.uniform(1)
        print(t)
        x = t.data
        print(x)
        t.data = Tensor.uniform(2)
        print(t.data)
