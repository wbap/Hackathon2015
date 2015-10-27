#!/usr/bin/env python
"""Chainer example: train a multi-layer perceptron on MNIST

This is a minimal example to write a feed-forward net. It requires scikit-learn
to load MNIST dataset.

"""
import argparse

import numpy as np
import six

import chainer
from chainer import computational_graph as c
from chainer import cuda
from chainer import Variable, FunctionSet
import chainer.functions as F
from chainer import optimizers

from dataset import load_dataset
import logging
import os
import time
import pickle


def create_result_dir(args):
    if args.restart_from is None:
        result_dir = 'results/' + os.path.basename("Developmental_train").split('.')[0]
        result_dir += '_' + time.strftime('%Y-%m-%d_%H-%M-%S_')
        result_dir += str(time.time()).replace('.', '')
        if not os.path.exists(result_dir):
            os.makedirs(result_dir)
        log_fn = '%s/log.txt' % result_dir
        logging.basicConfig(
            format='%(asctime)s [%(levelname)s] %(message)s',
            filename=log_fn, level=logging.DEBUG)
        logging.info(args)
    else:
        result_dir = '.'
        log_fn = 'log.txt'
        logging.basicConfig(
            format='%(asctime)s [%(levelname)s] %(message)s',
            filename=log_fn, level=logging.DEBUG)
        logging.info(args)

    return log_fn, result_dir
    
class VGG_mini(FunctionSet):
    def __init__(self):
        super(VGG_mini, self).__init__(
            conv1_1=F.Convolution2D(3, 64, 3, stride=1, pad=1),
            conv1_2=F.Convolution2D(64, 256, 3, stride=1, pad=1),

            fc=F.Linear(16384, 10)
        )

    def forward(self, x_data, y_data, train=True, models=None):
        x = Variable(x_data, volatile=not train)
        t = Variable(y_data, volatile=not train)

        h = F.relu(self.conv1_1(x))
        h = F.max_pooling_2d(h, 2, stride=2)
        h = F.dropout(h, ratio=0.25, train=train)
        
        h = F.relu(self.conv1_2(h))
        h = F.max_pooling_2d(h, 2, stride=2)
        h = F.dropout(h, ratio=0.25, train=train)

        h = self.fc(h)

        if train:
            return F.softmax_cross_entropy(h, t), F.accuracy(h, t)
        else:
            # return F.softmax_cross_entropy(h, t), F.accuracy(h, t), h
            return F.softmax_cross_entropy(h, t), F.accuracy(h, t)
            
            
class VGG_mini2(FunctionSet):
    def __init__(self):
        super(VGG_mini2, self).__init__(
            conv2_1=F.Convolution2D(3, 64, 3, stride=1, pad=1),
            conv2_2=F.Convolution2D(64, 64, 3, stride=1, pad=1),
        )

    def forward(self, x_data, y_data, train=True, models=None):
        VGG_mini = models["VGG_mini"]
        
        x = Variable(x_data, volatile=not train)
        t = Variable(y_data, volatile=not train)

        h = F.relu(self.conv2_1(x))
        h = F.relu(self.conv2_2(h))
        h = F.max_pooling_2d(h, 2, stride=2)
        h = F.dropout(h, ratio=0.25, train=train)
        
        h = F.relu(VGG_mini.conv1_2(h))
        h = F.max_pooling_2d(h, 2, stride=2)
        h = F.dropout(h, ratio=0.25, train=train)
        
        h = VGG_mini.fc(h)

        if train:
            return F.softmax_cross_entropy(h, t), F.accuracy(h, t)
        else:
            # return F.softmax_cross_entropy(h, t), F.accuracy(h, t), h
            return F.softmax_cross_entropy(h, t), F.accuracy(h, t)
            
            
class VGG_mini3(FunctionSet):
    def __init__(self):
        super(VGG_mini3, self).__init__(
            conv3_1=F.Convolution2D(3, 64, 3, stride=1, pad=1),
            conv3_2=F.Convolution2D(64, 64, 3, stride=1, pad=1),
        )

    def forward(self, x_data, y_data, train=True, models=None):
        VGG_mini = models["VGG_mini"]
        VGG_mini2 = models["VGG_mini2"]
        
        x = Variable(x_data, volatile=not train)
        t = Variable(y_data, volatile=not train)

        h = F.relu(self.conv3_1(x))
        h = F.relu(self.conv3_2(h))
        h = F.relu(VGG_mini2.conv2_2(h))
        h = F.max_pooling_2d(h, 2, stride=2)
        h = F.dropout(h, ratio=0.25, train=train)
        
        h = F.relu(VGG_mini.conv1_2(h))
        h = F.max_pooling_2d(h, 2, stride=2)
        h = F.dropout(h, ratio=0.25, train=train)
        
        h = VGG_mini.fc(h)

        if train:
            return F.softmax_cross_entropy(h, t), F.accuracy(h, t)
        else:
            # return F.softmax_cross_entropy(h, t), F.accuracy(h, t), h
            return F.softmax_cross_entropy(h, t), F.accuracy(h, t)
            

class VGG_mini4(FunctionSet):
    def __init__(self):
        super(VGG_mini4, self).__init__(
            conv4_1=F.Convolution2D(3, 64, 3, stride=1, pad=1),
            conv4_2=F.Convolution2D(64, 64, 3, stride=1, pad=1),
        )

    def forward(self, x_data, y_data, train=True, models=None):
        VGG_mini = models["VGG_mini"]
        VGG_mini2 = models["VGG_mini2"]
        VGG_mini3 = models["VGG_mini3"]
        
        x = Variable(x_data, volatile=not train)
        t = Variable(y_data, volatile=not train)

        h = F.relu(self.conv4_1(x))
        h = F.relu(self.conv4_2(h))
        h = F.relu(VGG_mini3.conv3_2(h))
        h = F.relu(VGG_mini2.conv2_2(h))
        h = F.max_pooling_2d(h, 2, stride=2)
        h = F.dropout(h, ratio=0.25, train=train)
        
        h = F.relu(VGG_mini.conv1_2(h))
        h = F.max_pooling_2d(h, 2, stride=2)
        h = F.dropout(h, ratio=0.25, train=train)
        
        h = VGG_mini.fc(h)

        if train:
            return F.softmax_cross_entropy(h, t), F.accuracy(h, t)
        else:
            # return F.softmax_cross_entropy(h, t), F.accuracy(h, t), h
            return F.softmax_cross_entropy(h, t), F.accuracy(h, t)
            
            
def train(n_epoch, model, N, N_test, batchsize, x_train, y_train, x_test, y_test, **kwargs):
    logging.info('start training...')
    time = 0.0
    for epoch in six.moves.range(1, n_epoch + 1):
        print('epoch', epoch)

        # training
        perm = np.random.permutation(N)
        sum_accuracy = 0
        sum_loss = 0
        for i in six.moves.range(0, N, batchsize):
            x_batch = xp.asarray(x_train[perm[i:i + batchsize]])
            y_batch = xp.asarray(y_train[perm[i:i + batchsize]])

            optimizer.zero_grads()
            loss, acc = model.forward(x_batch, y_batch, **kwargs)
            loss.backward()
            optimizer.update()

            sum_loss += float(loss.data) * len(y_batch)
            sum_accuracy += float(acc.data) * len(y_batch)
            time += 1.0

        msg = 'epoch:{:02d}\ttrain mean loss={}, accuracy={}'.format(
            epoch + args.epoch_offset, sum_loss / N, sum_accuracy / N)
        logging.info(msg)
        print('\n%s' % msg)

        # evaluation
        sum_accuracy = 0
        sum_loss = 0
        for i in six.moves.range(0, N_test, batchsize):
            x_batch = xp.asarray(x_test[i:i + batchsize])
            y_batch = xp.asarray(y_test[i:i + batchsize])
            
            loss, acc = model.forward(x_batch, y_batch, train=False, **kwargs)

            sum_loss += float(loss.data) * len(y_batch)
            sum_accuracy += float(acc.data) * len(y_batch)
            
        msg = 'epoch:{:02d}\ttest mean loss={}, accuracy={}'.format(
            epoch + args.epoch_offset, sum_loss / N_test, sum_accuracy / N_test)
        logging.info(msg)
        print('\n%s' % msg)
         
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Chainer example: MNIST')
    parser.add_argument('--gpu', '-g', default=-1, type=int,
                        help='GPU ID (negative value indicates CPU)')
    parser.add_argument('--restart_from', type=str)
    parser.add_argument('--epoch_offset', type=int, default=0)
    parser.add_argument('--epoch', type=int, default=10)
    parser.add_argument('--datadir', type=str, default='data')
    args = parser.parse_args()
    if args.gpu >= 0:
        cuda.check_cuda_available()
    xp = cuda.cupy if args.gpu >= 0 else np

    batchsize = 100
    n_epoch = args.epoch
    n_units = 1000

    # create result dir
    log_fn, result_dir = create_result_dir(args)

    # Prepare dataset
    print('load CIFAR10 dataset')
    dataset = load_dataset(args.datadir)
    x_train, y_train, x_test, y_test = dataset
    x_train = x_train.astype(np.float32) / 255.0
    y_train = y_train.astype(np.int32)
    x_test = x_test.astype(np.float32) / 255.0
    y_test = y_test.astype(np.int32)
    N = x_train.shape[0]
    N_test = x_test.shape[0]
    
    models = []
    model = VGG_mini()
    
    if args.gpu >= 0:
        cuda.get_device(args.gpu).use()
        model.to_gpu()

    # Setup optimizer
    optimizer = optimizers.Adam()
    optimizer.setup(model)
    
    train(n_epoch, model, N, N_test, batchsize, x_train, y_train, x_test, y_test)
    
    model2 = VGG_mini2()
    
    if args.gpu >= 0:
        cuda.get_device(args.gpu).use()
        model2.to_gpu()

    # Setup optimizer
    optimizer = optimizers.Adam()
    optimizer.setup(model2)
    
    train(n_epoch, model2, N, N_test, batchsize, x_train, y_train, x_test, y_test, models={"VGG_mini": model})
    

    model3 = VGG_mini3()
    
    if args.gpu >= 0:
        cuda.get_device(args.gpu).use()
        model3.to_gpu()

    # Setup optimizer
    optimizer = optimizers.Adam()
    optimizer.setup(model3)
    
    train(n_epoch, model3, N, N_test, batchsize, x_train, y_train, x_test, y_test, models={"VGG_mini": model, "VGG_mini2": model2})
    

    model4 = VGG_mini4()
    
    if args.gpu >= 0:
        cuda.get_device(args.gpu).use()
        model4.to_gpu()

    # Setup optimizer
    optimizer = optimizers.Adam()
    optimizer.setup(model4)
    
    train(n_epoch, model4, N, N_test, batchsize, x_train, y_train, x_test, y_test, models={"VGG_mini": model, "VGG_mini2": model2, "VGG_mini3": model3})
    
