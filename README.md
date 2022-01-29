Simplegrad
```python
from core.tensor import Tensor
from core import optimizer as opt
import numpy as np
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

def test_my_net(self):
    np.random.seed(1337)
    model = TinyMyNet()
    optimizer = opt.SGD(get_parameters(model), lr=0.001)
    # train(model, optimizer, steps=1000)
    # evaluate(model)
    train(model, X_train, Y_train, optimizer, steps=1000)
    assert evaluate(model, X_test, Y_test) > 0.95
```
```bash
out>>>
model:<test_minist.TinyMyNet object at 0x7fbaadead0d0> step 999 loss 0.14 accuracy 0.95: 100%|██████████| 1000/1000 [00:04<00:00, 208.72it/s]
```