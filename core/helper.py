# -*- coding: utf-8 -*- 
# Time: 2022-02-08 19:17
# Copyright (c) 2022
# author: Euraxluo

from core.tensor import *
import numpy as np
from tqdm import trange


def train(model, X, Y, optimizer, steps, loss_function, batch_size=128, transform=lambda x: x, target_transform=lambda x: x):
    import time
    start = time.perf_counter()
    Tensor.training = True
    losses, accuracies = [], []
    for i in (t := trange(steps)):
        samp = np.random.randint(0, X.shape[0], size=(batch_size,))
        x = Tensor(transform(X[samp]))
        y = target_transform(Y[samp])

        # 前向传播
        out = model.forward(x)
        # 梯度计算
        loss = loss_function(out, y)
        # 梯度归零
        optimizer.zero_grad()
        # 反向传播,计算梯度
        loss.backward()
        # 梯度下降
        optimizer.step()

        loss = loss.data
        accuracy = (np.argmax(out.data, axis=-1) == y).mean()

        losses.append(loss)
        accuracies.append(accuracy)
        t.set_description("model:%s optimizer:%s time:%.2f step %d loss %.2f accuracy %.2f" % (str(model), str(optimizer), time.perf_counter() - start, i, loss, accuracy))


def evaluate(model, X, Y, num_classes=None, batch_size=128, return_predict=False, transform=lambda x: x,
             target_transform=lambda y: y):
    Tensor.training = False

    def numpy_eval(Y, num_classes):
        Y_test_preds_out = np.zeros(list(Y.shape) + [num_classes])

        for i in trange((len(Y) - 1) // batch_size + 1):
            x = Tensor(transform(X[i * batch_size:(i + 1) * batch_size]))
            Y_test_preds_out[i * batch_size:(i + 1) * batch_size] = model.forward(x).data
        Y_test_preds = np.argmax(Y_test_preds_out, axis=-1)
        Y_test = target_transform(Y)
        return (Y_test == Y_test_preds).mean(), Y_test_preds

    if num_classes is None:
        num_classes = Y.max().astype(int) + 1
    acc, Y_test_pred = numpy_eval(Y, num_classes)
    return (acc, Y_test_pred) if return_predict else acc
