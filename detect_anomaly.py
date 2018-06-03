#!/usr/bin/env python
# -*- coding: utf-8 -*-

import chainer
import net_2
import chainer.functions as F
from train_vae_with_specified_label import *  # noqa
from sklearn import svm
from sklearn.manifold import TSNE
from scipy.spatial.distance import mahalanobis


MODEL_PATH = './result/model_100.npz'
DIMZ = 100
NU = 0.01
GAMMA = 0.005
REDUCED_DATASET_PATH = './reduced_dataset.npy'
THRESHOLD = 1.3
N_INLIERS = 1000
N_OUTLIERS = 100
F_VALUE_PATH = './f_value.txt'

TRAIN_PATH = '/root/data/binarized_mnist/train.npy'
TEST_PATH = '/root/data/binarized_mnist/test.npy'


# encode images into (mu,sigma)
def encode(xs):
    xs = chainer.Variable(xs)
    with chainer.using_config('train', False), chainer.no_backprop_mode():
            mus, ln_vars = model.encode(xs)
    return mus, ln_vars


def extract_and_encode_images_with(label, number, dataset):
    xs = extract_specified_labels(dataset, label)
    mus, ln_vars = encode(xs[:number])
    return mus, ln_vars


def display(scores, num):
    for i, score in enumerate(scores):
        print(i, score)
        if num == i:
            break


def detect_outliers_with_oneclass_svm(inners, outliers, nu=0.1, gamma=0.1):
    detector = svm.OneClassSVM(nu=nu, gamma=gamma, kernel='rbf')

    # train a model
    detector.fit(inners)

    # predict inners
    inner_predictions = detector.predict(inners)

    # predict outliers
    outlier_predictions = detector.predict(outliers)

    n_error_inners = inner_predictions[inner_predictions == -1].size
    n_error_outliers = outlier_predictions[outlier_predictions == -1].size

    print('n_error_inners={}/{}'.format(n_error_inners, inners.shape[0]))
    print('n_error_outliers={}/{}'.format(n_error_outliers, outliers.shape[0]))

    # calculate inner scores
    # inner_scores = detector.decision_function(inners)
    # error_inner_scores = inner_scores[inner_predictions == -1]

    # # predict outliers
    # outlier_scores = detector.decision_function(outliers)
    # error_outlier_scores = outlier_scores[outlier_predictions == -1]

    # print('type(inner_scores)', type(inner_scores), inner_scores.shape)
    # display(error_inner_scores, 10)

    # print('type(outlier_scores))', type(outlier_scores), outlier_scores.shape)
    # display(error_outlier_scores, 10)


def detect_outliers_with_tsne(inliers, outliers, reuses=True):
    dataset = np.concatenate([inliers, outliers], axis=0)
    reduced_dataset = None

    if reuses is False:
        reduced_dataset = TSNE(n_components=2, random_state=0).fit_transform(dataset)
        np.save(REDUCED_DATASET_PATH, reduced_dataset)
    else:
        reduced_dataset = np.load(REDUCED_DATASET_PATH)

    reduced_inliers = reduced_dataset[:inliers.shape[0]]
    reduced_outliers = reduced_dataset[inliers.shape[0]:]

    # calculate a covariance matrix using reduced_inners
    inv_sigma = np.linalg.inv(np.cov(reduced_inliers, rowvar=False))
    print('inv_sigma', inv_sigma.shape)

    # calculate a mean using reduced_inners
    mean = np.mean(reduced_inliers, axis=0)

    with open(F_VALUE_PATH, 'w') as fout:
        for i in range(20):
            threshold = THRESHOLD + 0.1 * i
            r = sum(1 for outlier in reduced_outliers if mahalanobis(mean, outlier, inv_sigma) > threshold)
            b = sum(1 for inlier in reduced_inliers if mahalanobis(mean, inlier, inv_sigma) > threshold)

            # the number of predicted outliers
            n = r + b

            precision = r / n
            recall = r / N_OUTLIERS
            f = 2 * precision * recall / (precision + recall)
            # print('thr={},f={},p={},r={}'.format(threshold, f, precision, recall))
            fout.write('{} {} {} {}\n'.format(threshold, f, precision, recall))


if __name__ == '__main__':

    # load the trained  model
    model = net_2.VAE(784, DIMZ, 500, F.softplus)
    chainer.serializers.load_npz(MODEL_PATH, model, strict=True)
    model.to_cpu()

    # load dataset
    # use binarized mnist
    # src_train, src_test = chainer.datasets.get_mnist(withlabel=True)
    src_train = np.load(TRAIN_PATH)
    src_test = np.load(TEST_PATH)

    # extract zero images
    zero_mus, zero_ln_vars = extract_and_encode_images_with(label=0, number=N_INLIERS, dataset=src_train)

    # extract six images
    six_mus, six_ln_vars = extract_and_encode_images_with(label=6, number=N_OUTLIERS, dataset=src_train)

    detect_outliers_with_tsne(inliers=zero_mus.data, outliers=six_mus.data, reuses=False)
