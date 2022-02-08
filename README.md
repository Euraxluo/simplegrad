Simplegrad
```python
from core.tensor import *
from core import optimizer as opt
from core.loss import *
from core.helper import *

import unittest
import numpy as np
from dataset.minist import fetch_mnist
from tqdm import trange

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



X_train, Y_train, X_test, Y_test = fetch_mnist()


class TestMNIST(unittest.TestCase):
    def test_linear_adam(self):
        np.random.seed(1)
        model = LinearNet()
        optimizer = opt.Adam(model, lr=0.001)
        train(model, X_train, Y_train, optimizer, steps=1000, loss_function=sparse_categorical_cross_entropy)
        assert evaluate(model, X_test, Y_test) > 0.95

```
```bash
out>>>
model:<test_minist.LinearNet object at 0x7fda938220d0> optimizer:<core.optimizer.Adam object at 0x7fdab86e2580> time:213.68 step 999 loss 0.15 accuracy 0.97: 100%|██████████| 1000/1000 [03:33<00:00,  4.68it/s]
100%|██████████| 79/79 [00:09<00:00,  8.49it/s]
```