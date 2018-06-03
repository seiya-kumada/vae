#!/usr/bin/env python
# -*- coding: utf-8 -*-
import chainer
import os
import numpy as np

OUTPUT_DIR = '/root/data/binarized_mnist'


def convert_to_binary(dataset):
    imgs = []
    lbls = []
    for v in dataset:
        imgs.append((v[0] >= 0.5).astype(np.float32))
        lbls.append(v[1])
    return chainer.datasets.TupleDataset(imgs, lbls)


def make_binarized_mnist():

    train, test = chainer.datasets.get_mnist(withlabel=True)
    binarized_train = convert_to_binary(train)
    binarized_test = convert_to_binary(test)

    train_path = os.path.join(OUTPUT_DIR, 'train.npy')
    np.save(train_path, binarized_train)

    test_path = os.path.join(OUTPUT_DIR, 'test.npy')
    np.save(test_path, binarized_test)


if __name__ == '__main__':

    make_binarized_mnist()
