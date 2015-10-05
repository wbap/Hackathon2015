 
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
import matplotlib.pyplot as plt
get_ipython().magic('matplotlib inline')

# import pycuda.autoinit

from chainer import cuda, Function, FunctionSet, gradient_check, Variable, optimizers
import chainer.functions as F

from dA import DenoisingAutoencoder
from SdA import StackedDenoisingAutoencoder
from CdA import ConvolutionalDenoisingAutoencoder


# In[ ]:

## load images
path_imagedir = '/Users/Hiho/Downloads/mit_body_v2'

# count up
num_images = 0
for name in os.listdir(path_imagedir):
    if re.match( '.*png$', name ):
        num_images = num_images+1
        
# get image size
for name in os.listdir(path_imagedir):
    if re.match( '.*png$', name ):
        img = Image.open( os.path.join(path_imagedir, name) )
        size_image = img.size
        break
num_pximage = size_image[0]*size_image[1]

# laod images
imgs = np.zeros((num_images, num_pximage), dtype=np.float32)
i=0
for name in os.listdir(path_imagedir):
    if re.match( '.*png$', name ):
        img = Image.open( os.path.join(path_imagedir, name) )
        img = np.asarray(img, dtype=np.uint8).T
        imgs[i,:] = np.reshape( np.mean(img, axis=0), (1, -1) ).astype(np.float32) / 255
        i=i+1


# In[ ]:

## make movie
num_frame = 5
num_movies = num_images - num_frame + 1
num_pxmovie = num_pximage*num_frame
movies = np.zeros((num_movies, num_pxmovie), dtype=np.float32)
for i in range(num_movies):
    movies[i,:] = np.reshape( imgs[i:i+5,:], (1, -1) )


# In[ ]:

## load json files
i=0
true_poses = [{}] * num_images
joint_angles = [{}] * num_images
for name in os.listdir(path_imagedir):
    if re.match( '.*json$', name ):
        j = json.load( open(os.path.join(path_imagedir, name)) )
        true_poses[i] = j['true_position']
        joint_angles[i] = j['joint_angle']
        i = i+1


# In[ ]:

## setup ML values
num_test =  num_movies // 40
num_train = num_movies - num_test
v_all = movies.copy()

num_node_tp = 9
tp_all = np.zeros((num_movies, num_node_tp), dtype=np.float32)
for i in range(num_movies):
    tp = true_poses[i+num_frame-1]
    tp_all[i][0:3] = [tp['right_elbow']['x'], tp['right_elbow']['y'], tp['right_elbow']['z']]
    tp_all[i][3:6] = [tp['right_shoulder']['x'], tp['right_shoulder']['y'], tp['right_shoulder']['z']]
    tp_all[i][6:9] = [tp['right_hand']['x'], tp['right_hand']['y'], tp['right_hand']['z']]
    
num_node_xA = 4
xA_all = np.zeros((num_movies, num_node_xA), dtype=np.float32)
for i in range(num_movies):
    xA = joint_angles[i+num_frame-1]
    xA_all[i][0:3] = [xA['right_shoulder']['y'], xA['right_shoulder']['p'], xA['right_shoulder']['r']]
    xA_all[i][3] = xA['right_elbow']['p']
xA_all = xA_all/360

# shuffle all data
rng = np.random.RandomState(1234)
indices = np.arange(num_movies)
rng.shuffle(indices)
v_all = v_all[indices]
tp_all = tp_all[indices]

# split test and train data
v_train, v_test = np.split(v_all, [num_train])
tp_train, tp_test = np.split(tp_all, [num_train])
xA_train, xA_test = np.split(xA_all, [num_train])

batchsize = 100
n_epoch = 300


# In[ ]:

# create SdA
n_hiddens = (12**2*num_frame, 6**2*num_frame)
sda = StackedDenoisingAutoencoder(num_pxmovie, n_hiddens)
sda.train(v_all, n_epoch=n_epoch)
sda.save('history', n_hiddens, n_epoch, batchsize)
# sda.load('history/SdA_layer(576, 64)_epoch300.pkl')

# split test and train data
yA_each = sda.predict(v_all, bAllLayer=True)
yA_all = yA_each[-1]
# yA_hidden1_all = yA_each[0]
yA_train, yA_test = np.split(yA_all, [num_train])

# check output histgram
dummy = plt.hist(np.reshape(yA_all, (-1, 1)), 50)


# In[ ]:

## draw weight
def draw_weight(data, size):
    Z = data.reshape(size).T
    plt.imshow(Z, interpolation='none')
    plt.xlim(0,size[0])
    plt.ylim(0,size[1])
    plt.gray()
    plt.tick_params(labelbottom="off")
    plt.tick_params(labelleft="off")

num_show = 4
for i_layer in range(len(n_hiddens)):
    for i in range(num_show):
        for i_frame in range(num_frame):
            plt.subplot(len(n_hiddens)*num_frame, num_show, num_show*(num_frame*i_layer+i_frame)+i+1)
            iw_s = num_pximage*i_frame
            iw_e = num_pximage*(i_frame+1)
            draw_weight( sda.SdA[i_layer].model.encode.W[i][iw_s:iw_e], (math.sqrt(sda.n_nodes[i_layer]/num_frame), math.sqrt(sda.n_nodes[i_layer]/num_frame)) )


# In[ ]:

# check true position
model = FunctionSet(
#     l1 = F.Linear(n_hiddens[-1], 50),
#     l2 = F.Linear(50, num_node_tp),
    l = F.Linear(n_hiddens[-1], num_node_tp),
)
optimizer = optimizers.SGD()
optimizer.setup(model.collect_parameters())

def forward(x_data, y_data):
    x = Variable(x_data); t = Variable(y_data)
#     h = F.relu(model.l1(x))
    y = model.l(x)
    return F.mean_squared_error(y, t), y

for epoch in range(n_epoch):
    indexes = np.random.permutation(num_train)
    sum_loss = 0
    for i in range(0, num_train, batchsize):
        x_batch = yA_train[indexes[i : i + batchsize]]
        y_batch = tp_train[indexes[i : i + batchsize]]
        optimizer.zero_grads()
        loss, output = forward(x_batch, y_batch)
        loss.backward()
        optimizer.update()
        sum_loss = sum_loss+loss.data*batchsize
    print('epoch:'+str(epoch)+' loss:' + str(sum_loss/num_train))
    
# test
loss, output = forward(yA_test, tp_test)
print('test loss:' + str(loss.data))

for i_check in range(0, num_test, math.floor(num_test/8)):
    print(i_check)
    print( "true : %.2f %.2f %.2f %.2f %.2f %.2f %.2f %.2f %.2f" % (tp_test[i_check][0], tp_test[i_check][1], tp_test[i_check][2], tp_test[i_check][3], tp_test[i_check][4], tp_test[i_check][5], tp_test[i_check][6], tp_test[i_check][7], tp_test[i_check][8] ))
    print( "predicted : %.2f %.2f %.2f %.2f %.2f %.2f %.2f %.2f %.2f" % (output.data[i_check][0], output.data[i_check][1], output.data[i_check][2], output.data[i_check][3], output.data[i_check][4], output.data[i_check][5], output.data[i_check][6], output.data[i_check][7], output.data[i_check][8] ))


# In[ ]:

# fA(xA->yA)
model = FunctionSet(
    l1 = F.Linear(num_node_xA, 50),
    l2 = F.Linear(50, n_hiddens[-1]),
)
optimizer = optimizers.SGD()
optimizer.setup(model.collect_parameters())

def forward(x_data, y_data):
    x = Variable(x_data); t = Variable(y_data)
    h = F.sigmoid(model.l1(x))
    y = model.l2(h)
    return F.mean_squared_error(y, t), y

for epoch in range(n_epoch):
    indexes = np.random.permutation(num_images)
    sum_loss = 0
    for i in range(0, num_train, batchsize):
        x_batch = xA_all[indexes[i : i + batchsize]]
        y_batch = yA_all[indexes[i : i + batchsize]]
        optimizer.zero_grads()
        loss, output = forward(x_batch, y_batch)
        loss.backward()
        optimizer.update()
        sum_loss = sum_loss+loss.data*batchsize
    print('epoch:'+str(epoch)+' loss:' + str(sum_loss/num_train))
    
# test
loss, output = forward(xA_test, yA_test)
print('test loss:' + str(loss.data))

