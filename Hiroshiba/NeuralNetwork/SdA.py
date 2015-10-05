import pickle

import numpy as np

# import pycuda.autoinit

from chainer import cuda, Function, FunctionSet, gradient_check, Variable, optimizers
import chainer.functions as F

from SuperClass import SuperClass
from dA import DenoisingAutoencoder

class StackedDenoisingAutoencoder(SuperClass):
    def __init__(self, n_in, n_hiddens, n_epoch=20, batchsize=100, use_cuda=False):
        super().__init__(n_epoch, batchsize, use_cuda)
        
        self.SdA = []
        self.n_nodes = (n_in,) + n_hiddens
        
        self.num_layer = len(n_hiddens)
        for i in range(self.num_layer):
            dA = DenoisingAutoencoder(self.n_nodes[i], self.n_nodes[i+1], n_epoch, batchsize, use_cuda=use_cuda)
            self.SdA.append(dA)
            
        # self.registModel()

    def predict(self, x_data, bAllLayer=False):
        x_data = self.procInput(x_data)
        
        x_eachlayer = []
        for i in range(self.num_layer):
            x_data = self.SdA[i].predict(x_data)
            x_eachlayer.append(self.procOutput(x_data))
        p = x_data
        
        if bAllLayer:
            return x_eachlayer
        else:
            return p

#     def cost(self, x_data):
#         return F.mean_squared_error(y, t)

    def train(self, x_data):
        for i_layer in range(self.num_layer):
            self.SdA[i_layer].train(x_data)
            x_data = self.SdA[i_layer].predict(x_data)

    def save(self, filedir, n_hiddens, n_epoch, batchsize):
        name = "SdA_"+ "layer"+str(n_hiddens) + "_epoch"+str(n_epoch)
        params = []
        for i in range(len(self.SdA)):
            dic = {}
            dic['W'] = self.SdA[i].model.encode.parameters[0]
            dic['b'] = self.SdA[i].model.encode.parameters[1]
            params.append(dic)
        pickle.dump(params, open(filedir+'/'+name+'.pkl', 'wb'), pickle.HIGHEST_PROTOCOL)
        return
    
    def load(self, filename):
        if filename.find('.pkl')==-1:
            filename = filename + '.pkl'
        params = pickle.load(open(filename, 'rb'))
        for i in range(len(self.SdA)):
            dic = params[i]
            self.SdA[i].model.encode.parameters = (dic['W'], dic['b'])
        return