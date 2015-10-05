import math

import numpy as np

# import pycuda.autoinit

from chainer import cuda, Function, FunctionSet, gradient_check, Variable, optimizers
import chainer.functions as F

class ConvolutionalDenoisingAutoencoder():
    def __init__(self, imgsize, n_in_channels, n_out_channels, ksize, stride=1, pad=0, use_cuda=False):
        self.model = FunctionSet(
            encode=F.Convolution2D(n_in_channels, n_out_channels, ksize, stride, pad),
            decode=F.Linear(n_out_channels*(math.floor((imgsize+2*pad-ksize)/stride)+1)**2, n_in_channels*imgsize**2)
        )
        self.use_cuda = use_cuda

        if self.use_cuda:
            self.model.to_gpu()

        self.optimizer = optimizers.Adam()
        self.optimizer.setup(self.model.collect_parameters())

    def encode(self, x_var):
        return F.sigmoid(self.model.encode(x_var))

    def decode(self, x_var):
        return self.model.decode(x_var)

    def predict(self, x_data):
        if self.use_cuda:
            x_data = cuda.to_gpu(x_data)
        x = Variable(x_data)
        p = self.encode(x)
        if self.use_cuda:
            return cuda.to_cpu(p.data)
        else:
            return p.data

    def cost(self, x_data):
        x = Variable(x_data)
        t = Variable(x_data.reshape(x_data.shape[0], x_data.shape[1]*x_data.shape[2]*x_data.shape[3]))
        h = F.dropout(x)
        h = self.encode(h)
        y = self.decode(h)
        return F.mean_squared_error(y, t)

    def train(self, x_data):
        if self.use_cuda:
            x_data = cuda.to_gpu(x_data)
        self.optimizer.zero_grads()
        loss = self.cost(x_data)
        loss.backward()
        self.optimizer.update()
        if self.use_cuda:
            return float(cuda.to_cpu(loss.data))
        else:
            return loss.data

    def test(self, x_data):
        if self.use_cuda:
            x_data = cuda.to_gpu(x_data)
        loss = self.cost(x_data)
        return float(cuda.to_cpu(loss.data))
