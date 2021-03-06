# -*- coding: utf-8 -*- 
# Time: 2022-01-20 17:25
# Copyright (c) 2022
# author: Euraxluo

import requests, gzip, os, hashlib, numpy


def fetch_mnist():
    def fetch(url):
        fp = os.path.join("/tmp", hashlib.md5(url.encode('utf-8')).hexdigest())
        if os.path.isfile(fp):
            with open(fp, "rb") as f:
                dat = f.read()
        else:
            with open(fp, "wb") as f:
                dat = requests.get(url).content
                f.write(dat)
        return numpy.frombuffer(gzip.decompress(dat), dtype=numpy.uint8).copy()

    X_train = fetch("http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz")[0x10:].reshape((-1, 28*28))
    Y_train = fetch("http://yann.lecun.com/exdb/mnist/train-labels-idx1-ubyte.gz")[8:]
    X_test = fetch("http://yann.lecun.com/exdb/mnist/t10k-images-idx3-ubyte.gz")[0x10:].reshape((-1, 28*28))
    Y_test = fetch("http://yann.lecun.com/exdb/mnist/t10k-labels-idx1-ubyte.gz")[8:]
    return X_train, Y_train, X_test, Y_test
