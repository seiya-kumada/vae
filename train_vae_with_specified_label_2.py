#!/usr/bin/env python
"""Chainer example: train a VAE on MNIST
"""
import argparse
import os
import numpy as np
import chainer
from chainer import training
from chainer.training import extensions
import net_4
import chainer.functions as F

# see make_binarized_mnist.py
TRAIN_PATH = '/root/data/binarized_mnist/train.npy'
TEST_PATH = '/root/data/binarized_mnist/test.npy'
np.random.seed(1)
chainer.cuda.cupy.random.seed(1)


def extract_specified_labels(dataset, n):
    return np.array([img for (img, label) in dataset if label == n])


def main():
    parser = argparse.ArgumentParser(description='Chainer example: VAE')
    parser.add_argument('--initmodel', '-m', default='',
                        help='Initialize the model from given file')
    parser.add_argument('--resume', '-r', default='',
                        help='Resume the optimization from snapshot')
    parser.add_argument('--gpu', '-g', default=-1, type=int,
                        help='GPU ID (negative value indicates CPU)')
    parser.add_argument('--out', '-o', default='result',
                        help='Directory to output the result')
    parser.add_argument('--epoch', '-e', default=100, type=int,
                        help='number of epochs to learn')
    parser.add_argument('--dimz', '-z', default=20, type=int,
                        help='dimention of encoded vector')
    parser.add_argument('--batchsize', '-b', type=int, default=100,
                        help='learning minibatch size')
    parser.add_argument('--test', action='store_true',
                        help='Use tiny datasets for quick tests')
    parser.add_argument('--label', '-l', type=int, default=5,
                        help='number when training')
    args = parser.parse_args()

    print('GPU: {}'.format(args.gpu))
    print('# dim z: {}'.format(args.dimz))
    print('# Minibatch-size: {}'.format(args.batchsize))
    print('# epoch: {}'.format(args.epoch))
    print('')

    # Prepare VAE model, defined in net.py
    model = net_4.VAE(784, n_latent=args.dimz, n_h=500, activation=F.softplus)
    # model = net.VAE(784, n_latent=args.dimz, n_h=500)

    # Setup an optimizer
    optimizer = chainer.optimizers.Adam()
    optimizer.setup(model)

    # Initialize
    if args.initmodel:
        chainer.serializers.load_npz(args.initmodel, model)

    # Load the MNIST dataset
    train, test = chainer.datasets.get_mnist(withlabel=True)

    # Extract images which have the specified labels
    train = extract_specified_labels(train, args.label)
    test = extract_specified_labels(test, args.label)

    if args.test:
        train, _ = chainer.datasets.split_dataset(train, 100)
        test, _ = chainer.datasets.split_dataset(test, 100)

    train_iter = chainer.iterators.SerialIterator(train, args.batchsize)
    test_iter = chainer.iterators.SerialIterator(test, args.batchsize,
                                                 repeat=False, shuffle=False)

    # Set up an updater. StandardUpdater can explicitly specify a loss function
    # used in the training with 'loss_func' option
    updater = training.StandardUpdater(
        train_iter, optimizer,
        device=args.gpu, loss_func=model.get_loss_func())

    trainer = training.Trainer(updater, (args.epoch, 'epoch'), out=args.out)
    trainer.extend(extensions.Evaluator(test_iter, model, device=args.gpu,
                                        eval_func=model.get_loss_func(k=10)))
    trainer.extend(extensions.dump_graph('main/loss'))
    trainer.extend(extensions.snapshot(), trigger=(args.epoch, 'epoch'))
    trainer.extend(extensions.LogReport())
    trainer.extend(extensions.PrintReport(
        [
            'epoch',
            'main/loss',
            'validation/main/loss',
            'main/rec_loss',
            'validation/main/rec_loss',
            'main/kl',
            'validation/main/kl',
            'main/mu',
            'validation/main/mu',
            'main/sigma',
            'validation/main/sigma',
            'elapsed_time',
        ]))
    trainer.extend(extensions.ProgressBar())

    if args.resume:
        chainer.serializers.load_npz(args.resume, trainer)

    # Run the training
    trainer.run()

    # Save the model
    chainer.serializers.save_npz(os.path.join(args.out, 'model_{}.npz'.format(args.epoch)), model, compression=True)


if __name__ == '__main__':
    main()
