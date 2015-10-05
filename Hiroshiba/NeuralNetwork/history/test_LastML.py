
# coding: utf-8

# In[ ]:

import os
import sys
import re # for regex
import math
import json
import pickle

from PIL import Image
import numpy as np

from sklearn.datasets import fetch_mldata
# import matplotlib.pyplot as plt
# get_ipython().magic('matplotlib inline')

from chainer import cuda, Function, FunctionSet, gradient_check, Variable, optimizers
import chainer.functions as F

from dA import DenoisingAutoencoder
from SdA import StackedDenoisingAutoencoder

import utils


# In[ ]:

## Params
use_cuda = True

batchsize = 100
if use_cuda:
    n_epoch_SdA = 50
    n_epoch_fA = 50
    n_epoch_last = 30
else:
    n_epoch_SdA = 3
    n_epoch_fA = 10
    n_epoch_last = 30

if use_cuda:
    path_imagedir = {'self':os.environ['HOME'] + '/Hevy/wba_hackathon/self_mit_fp_50x1000/', 'other':os.environ['HOME'] + '/Hevy/wba_hackathon/other_mit_fp_50x100/'}
    n_dataset_self = 1000
    n_dataset_other = 100
else:
    path_imagedir = {'self':os.environ['HOME'] + '/Hevy/wba_hackathon/self_mit_fp_50x100/', 'other':os.environ['HOME'] + '/Hevy/wba_hackathon/other_mit_fp_50x100/'}
    n_dataset_self = 100
    n_dataset_other = 100
n_dataset = n_dataset_self + n_dataset_other

size_image = [64, 64]
scale = 4
size_image[0]=size_image[0]/scale
size_image[1]=size_image[1]/scale

n_hold = 10
n_cross = 1

n_fA_node = 32

n_moveframe = 50
n_oneframe = 5
n_onemovie = int(n_moveframe / n_oneframe)
if use_cuda:
    n_hiddens = (24**2, 12**2, 4**2)
else:
    n_hiddens = (4**2, 2**2)

num_images = n_dataset * n_moveframe
num_movie = num_images / n_oneframe

num_test_dataset = n_dataset // n_hold
num_train_dataset = n_dataset - num_test_dataset
num_test_movie =  num_movie // n_hold
num_train_movie = num_movie - num_test_movie

if use_cuda:
    cuda.check_cuda_available()


# In[ ]:

## load images
size = size_image[0]
num_pximage = size**2
num_pxmovie = n_oneframe * num_pximage

# load images
movies = np.zeros((n_dataset, n_moveframe, num_pximage), dtype=np.float32)
i = 0
for label in {'self', 'other'}:
    for name in os.listdir(path_imagedir[label]):
        if re.match( '.*png$', name ):
            img = Image.open( os.path.join(path_imagedir[label], name) )
            img.thumbnail( (size_image[0], size_image[1]) )
            img = np.asarray(img, dtype=np.float32).mean(axis=2).T
            movies[i//n_moveframe, i%n_moveframe, :] = np.reshape( img / 255.0, (1, -1) )
            i = i+1

## load json files
joint_angles = [{}] * num_images
i = 0
for label in {'self', 'other'}:
    for name in os.listdir(path_imagedir[label]):
        if re.match( '.*json$', name ):
            j = json.load( open(os.path.join(path_imagedir[label], name)) )
            joint_angles[i] = j['joint_angle']
            i = i+1


# In[ ]:

## setup ML values
v_all = np.reshape(movies, (n_dataset, -1))
v_all = utils.splitInputs(v_all, n_moveframe/n_oneframe)

num_node_x = 8
x_all = np.zeros((num_images, num_node_x), dtype=np.float32)
for i in range(len(joint_angles)):
    x_all[i][0:3] = [joint_angles[i]['left_shoulder']['y'],                     joint_angles[i]['left_shoulder']['p'],                     joint_angles[i]['left_shoulder']['r']]
    x_all[i][3]   =  joint_angles[i]['left_elbow']['p']
    x_all[i][4:7] = [joint_angles[i]['right_shoulder']['y'],                     joint_angles[i]['right_shoulder']['p'],                     joint_angles[i]['right_shoulder']['r']]
    x_all[i][7]   =  joint_angles[i]['right_elbow']['p']

x_all = x_all/180
x_all = utils.bindInputs(x_all, n_moveframe)
x_all = utils.splitInputs(x_all, n_moveframe/n_oneframe)


# In[ ]:

# label 0:other, 1:self
label_x = np.append( np.ones((n_dataset_self), dtype=np.int32), np.zeros((n_dataset_other), dtype=np.int32) )

# shuffle all data
rng = np.random.RandomState(1234)
indices = np.arange(n_dataset, dtype=np.int32)
rng.shuffle(indices)
v_all   = v_all[indices]
x_all   = x_all[indices]
label_x = label_x[indices]

n_set = n_dataset / n_hold
# split each data into 10 block
v_s = np.split(v_all, n_set*np.r_[1:n_hold])
x_s = np.split(x_all, n_set*np.r_[1:n_hold])
label_x_s = np.split(label_x, n_set*np.r_[1:n_hold])

num_layers= len(n_hiddens)


# In[ ]:

def forward(x_data, y_data):
    x = Variable(x_data); t = Variable(y_data)
    h = model.l1(x)
    y = F.sigmoid(model.l2(h))
    return F.mean_squared_error(y, t), y

def forwardLastML(x_data, y_data):
    x = Variable(x_data); t = Variable(y_data)
    h = F.sigmoid(model.l1(x))
    y = model.l2(h)
    return F.softmax_cross_entropy(y, t), y

list_cross = []
for i in range(n_cross):
    # split test and train data
    set_l = list(set(range(n_hold)).difference([i]))
    v_train = np.empty(0, dtype=np.float32)
    x_train = np.empty(0, dtype=np.float32)
    label_train = np.empty(0, dtype=np.int32)
    for i_set in range(n_hold-1):
        v_train = utils.vstack_(v_train, v_s[set_l[i_set]])
        x_train = utils.vstack_(x_train, x_s[set_l[i_set]])
        label_train = utils.vstack_(label_train, label_x_s[set_l[i_set]])
    
    v_train = np.reshape(v_train, (num_train_movie, -1))
    x_train = np.reshape(x_train, (num_train_movie, -1))
    label_train = np.reshape(label_train, (num_train_dataset, -1))
    v_test = np.reshape(v_s[i], (num_test_movie, -1))
    x_test = np.reshape(x_s[i], (num_test_movie, -1))
    label_test = label_x_s[i]
    
    # create SdA
    sda = StackedDenoisingAutoencoder(num_pxmovie, n_hiddens, n_epoch=n_epoch_SdA, use_cuda=use_cuda)
    sda.train(v_train)
    
    # split test and train data
    y_train_each = sda.predict(v_train, bAllLayer=True)
    y_test_each = sda.predict(v_test, bAllLayer=True)
    
    list_layer = []
    for j in range(num_layers):
        y_train  = y_train_each[j]
        y_test   = y_test_each[j]
        # separate x&y into other and self
        x_test_split = [np.empty(0,dtype=np.float32), np.empty(0,dtype=np.float32)]
        y_test_split = [np.empty(0,dtype=np.float32), np.empty(0,dtype=np.float32)]
        for i_test in range(int(num_test_movie)):
            label = label_test[i_test//n_onemovie]
            x_test_split[label] = utils.vstack_(x_test_split[label], x_test[i_test])
            y_test_split[label] = utils.vstack_(y_test_split[label], y_test[i_test])
        
        # f(x->y)
        model = FunctionSet(
            l1 = F.Linear(num_node_x*n_oneframe, n_fA_node),
            l2 = F.Linear(n_fA_node, n_hiddens[j])
        )
        optimizer = optimizers.SGD()
        optimizer.setup(model.collect_parameters())
        
        dic = {'loss':{}, 'hist':{}, 'lastpredict':{}}
        dic['loss'] = {'self':np.empty(0,dtype=np.float32), 'other':np.empty(0,dtype=np.float32)}
        for epoch in range(n_epoch_fA):
            indexes = np.random.permutation(int(num_train_movie))
            sum_loss = 0
            for k in range(0, int(num_train_movie), batchsize):
                x_batch = x_train[indexes[k : k + batchsize]]
                y_batch = y_train[indexes[k : k + batchsize]]
                optimizer.zero_grads()
                loss, output = forward(x_batch, y_batch)
                loss.backward()
                optimizer.update()
                sum_loss = sum_loss+loss.data*batchsize
            print('fA: epoch:'+str(epoch)+' loss:' + str(sum_loss/num_train_movie))
            
            # test
            loss, output = forward(x_test_split[1], y_test_split[1])
            dic['loss']['self'] = utils.vstack_(dic['loss']['self'], loss.data)
            loss, output = forward(x_test_split[0], y_test_split[0])
            dic['loss']['other'] = utils.vstack_(dic['loss']['other'], loss.data)
            print('test loss:' + str(loss.data))
        
        dic['hist'] = {'self':np.empty(0, dtype=np.float32), 'other':np.empty(0, dtype=np.float32)}
        for i_test in range((x_test_split[1].shape[0])):
            loss, output = forward(x_test_split[1][i_test][None], y_test_split[1][i_test][None]) # [8,][None] -> [1,8]
            dic['hist']['self'] = utils.vstack_(dic['hist']['self'], loss.data)
        for i_test in range(x_test_split[0].shape[0]):
            loss, output = forward(x_test_split[0][i_test][None], y_test_split[0][i_test][None])
            dic['hist']['other'] = utils.vstack_(dic['hist']['other'], loss.data)
        
        # loss => self or other
        loss_train = np.zeros((num_train_dataset, n_onemovie), dtype=np.float32)
        for i_train in range(num_train_dataset):
            for i_movie in range(n_onemovie):
                loss, output = forward(x_train[i_train*n_onemovie+i_movie][None], y_train[i_train*n_onemovie+i_movie][None])
                loss_train[i_train, i_movie] = loss.data
        loss_test = np.zeros((num_test_dataset, n_onemovie), dtype=np.float32)
        for i_test in range(num_test_dataset):
            for i_movie in range(n_onemovie):
                loss, output = forward(x_test[i_test*n_onemovie+i_movie][None], y_test[i_test*n_onemovie+i_movie][None])
                loss_test[i_test, i_movie] = loss.data
        
        model = FunctionSet(
            l1 = F.Linear(n_onemovie, n_onemovie//2),
            l2 = F.Linear(n_onemovie//2, 2)
        )
        optimizer = optimizers.SGD()
        optimizer.setup(model.collect_parameters())
        
        for epoch in range(n_epoch_last):
            indexes = np.random.permutation(int(num_train_dataset))
            sum_loss = 0
            for k in range(0, int(num_train_dataset), batchsize):
                x_batch = loss_train[indexes[k : k + batchsize]]
                y_batch = label_train[indexes[k : k + batchsize]].ravel()
                optimizer.zero_grads()
                loss, output = forwardLastML(x_batch, y_batch)
                loss.backward()
                optimizer.update()
                sum_loss = sum_loss+loss.data*batchsize
            print('LastML: epoch:'+str(epoch)+' loss:' + str(sum_loss/num_train_dataset))
        
        dic['lastpredict']['label'] = label_test
        dic['lastpredict']['pedict'] = np.empty(0, dtype=np.int)
        dic['lastpredict']['output'] = np.empty(0, dtype=np.float)
        for i_test in range(num_test_dataset):
            loss, output = forwardLastML(loss_test[i_test][None], label_test[i_test].ravel())
            dic['lastpredict']['output'] = utils.vstack_(dic['lastpredict']['output'], output.data)
            if output.data[0,0] > output.data[0,1]:
                dic['lastpredict']['pedict'] = utils.vstack_(dic['lastpredict']['pedict'], 0)
            else:
                dic['lastpredict']['pedict'] = utils.vstack_(dic['lastpredict']['pedict'], 1)
        
        list_layer.append(dic)
        
    list_cross.append(list_layer)


# In[ ]:

# save data
f = open('save.dump', 'wb')
pickle.dump(list_cross, f)

print('finish!')

