
# coding: utf-8

# In[ ]:

import os
import sys
import re # for regex
import math
import json
import pickle
import importlib

from PIL import Image
import numpy as np

from sklearn.datasets import fetch_mldata
import matplotlib.pyplot as plt
get_ipython().magic('matplotlib inline')

# import pycuda.autoinit

from chainer import cuda, Function, FunctionSet, gradient_check, Variable, optimizers
import chainer.functions as F

import SuperClass
import dA
import SdA
import CdA
import sAE

from utils import *


# In[ ]:

mnist = fetch_mldata('MNIST original', data_home="~/Hevy/")
mnistdata = mnist.data[0:10000,:].astype(np.float32) / 255.0


# In[ ]:

importlib.reload(SuperClass)
importlib.reload(sAE)

n_in = mnistdata.shape[1]
n_hidden = 50**2

sae = sAE.SparseAutoEncoder(n_in, n_hidden, n_epoch=500)


# In[ ]:

sae.train(mnistdata)


# In[ ]:

W = sae.ae.W.get_value()

draw_weight(W[:,9], (28, 28))


# In[ ]:



