import pickle

import numpy as np

# import pycuda.autoinit

from SuperClass import SuperClass

from chainer import cuda, Function, FunctionSet, gradient_check, Variable, optimizers
import chainer.functions as F

## dA (https://gist.github.com/ktnyt/455694506ee6595c92e4)
class DenoisingAutoencoder(SuperClass):
    def __init__(self, n_in, n_hidden, n_epoch=20, batchsize=100, use_cuda=False):
        super().__init__(n_epoch, batchsize, use_cuda)
        
        self.model = FunctionSet(
            encode=F.Linear(n_in, n_hidden),
            decode=F.Linear(n_hidden, n_in)
        )
        self.registModel()

    def encode(self, x_var):
        return F.sigmoid(self.model.encode(x_var))

    def decode(self, x_var):
        return self.model.decode(x_var)

    def predict(self, x_data):
        x_data = self.procInput(x_data)
        x = Variable(x_data)
        p = self.encode(x)
        return self.procOutput(p.data)

    def cost(self, x_data):
        x_data = self.procInput(x_data)
        x = Variable(x_data)
        t = Variable(x_data)
        h = self.encode(F.dropout(t))
        y = self.decode(h)
        return self.procOutput(F.mean_squared_error(y, x))

    def test(self, x_data):
        x_data = self.procInput(x_data)
        x = Variable(x_data)
        t = Variable(x_data)
        h = self.encode(t)
        y = self.decode(h)
        return self.procOutput(F.mean_squared_error(y, x))

    def save(self, filedir, n_hidden, n_epoch, batchsize):
        name = "SdA_"+ "layer"+str(n_hidden) + "_epoch"+str(n_epoch)
        param = {}
        param['W'] = self.model.encode.parameters[0]
        param['b'] = self.model.encode.parameters[1]
        pickle.dump(param, open(filedir+'/'+name+'.pkl', 'wb'), pickle.HIGHEST_PROTOCOL)
        return
    
    def load(self, filename):
        if filename.find('.pkl')==-1:
            filename = filename + '.pkl'
        param = pickle.load(open(filename, 'rb'))
        self.model.encode.parameters = (param['W'], param['b'])
        return