import math
import pickle

import numpy as np

# import pycuda.autoinit

from chainer import cuda, Function, FunctionSet, gradient_check, Variable, optimizers
import chainer.functions as F

class SuperClass():
    def __init__(self, n_epoch=20, batchsize=100, use_cuda=False):
        self.n_epoch = n_epoch
        self.batchsize = batchsize
        self.use_cuda = use_cuda
        
        self.model = None
        
    # to setup Model params, should call this method
    def registModel(self):
        if self.use_cuda:
            self.model.to_gpu()

        self.optimizer = optimizers.Adam()
        self.optimizer.setup(self.model.collect_parameters())
        return
        
    # if used cuda, turn to GPU value
    def procInput(self, x_data):
        if self.use_cuda and not isinstance(x_data, cuda.cupy.ndarray):
            return cuda.to_gpu(x_data)
        else:
            return x_data
        
    # if used cuda, turn to GPU value
    def procOutput(self, t_data):
        if self.use_cuda and isinstance(t_data, cuda.cupy.ndarray):
            return cuda.to_cpu(t_data)
        else:
            return t_data

    # for Unsupervised
    def encode(self, x_var):
        assert(False)
        return None

    # for Unsupervised
    def decode(self, x_var):
        assert(False)
        return None

    # calc output with no training
    def predict(self, x_data):
        assert(False)
        return None

    # this method will be used in training
    def cost(self, x_data):
        assert(False)
        return -1
    
    # sometimes cost & loss have other process(such as dropout)
    def test(self, x_data):
        assert(False)
        return -1

    # calc cost and do backpropagation
    def train(self, x_data):
        for epoch in range(self.n_epoch):
            sum_loss = 0
            indexes = np.random.permutation(x_data.shape[0])
            for i in range(0, x_data.shape[0], self.batchsize):
                self.optimizer.zero_grads()
                x_batch = self.procInput( x_data[indexes[i : i + self.batchsize]] )
                cost = self.cost(x_batch)
                cost.backward()
                self.optimizer.update()
                sum_loss += self.procOutput(cost.data) * x_batch.shape[0]
            print(self.__class__.__name__+' epoch:'+str(epoch)+' loss:' + str(sum_loss/x_data.shape[0]))
