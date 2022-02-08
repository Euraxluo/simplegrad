# -*- coding: utf-8 -*- 
# Time: 2022-02-08 19:19
# Copyright (c) 2022
# author: Euraxluo

from core.tensor import Tensor
import numpy as np


def sparse_categorical_cross_entropy(out, Y):
    """
    sparse categorical cross entropy
    稀疏分类交叉熵
    :param out:
    :param Y:
    :return:
    """
    num_classes = out.shape[-1]
    YY = Y.flatten()
    y = np.zeros((YY.shape[0], num_classes), np.float32)
    # correct loss for NLL, torch NLL loss returns one per row
    y[range(y.shape[0]), YY] = -1.0 * num_classes
    y = y.reshape(list(Y.shape) + [num_classes])
    y = Tensor(y)
    return out.mul(y).mean()
