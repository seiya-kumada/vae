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


def detect_outliers_with_tsne(inners, outliers):
    dataset = np.concatenate([inners, outliers], axis=0)
    reduced_dataset = TSNE(n_components=2, random_state=0).fit_transform(dataset)
    reduced_inners = reduced_dataset[:inners.shape[0]]
    reduced_outliers = reduced_dataset[inners.shape[0]:]

    # calculate a covariance matrix using reduced_inners
    inv_sigma = np.linalg.inv(np.cov(reduced_inners, rowvar=False))
    print('inv_sigma', inv_sigma.shape)

    # calculate a mean using reduced_inners
    mean = np.mean(reduced_inners, axis=0)

    outlier_dists = [mahalanobis(mean, outlier, inv_sigma) for outlier in reduced_outliers]
    print(outlier_dists.shape)


if __name__ == '__main__':

    # load the trained  model
    model = net_2.VAE(784, DIMZ, 500, F.softplus)
    chainer.serializers.load_npz(MODEL_PATH, model, strict=True)
    model.to_cpu()

    # load dataset
    src_train, src_test = chainer.datasets.get_mnist(withlabel=True)

    # extract zero images
    zero_mus, zero_ln_vars = extract_and_encode_images_with(label=0, number=1000, dataset=src_train)

    # extract six images
    six_mus, six_ln_vars = extract_and_encode_images_with(label=6, number=100, dataset=src_train)

    # detect_outliers(inners=zero_mus.data, outliers=six_mus.data, nu=NU, gamma=GAMMA)
    detect_outliers_with_tsne(inners=zero_mus.data, outliers=six_mus.data)
