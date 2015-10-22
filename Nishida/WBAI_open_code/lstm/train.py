#coding:utf-8
import sys
import argparse
import cPickle as pickle
import copy
import os
import time
import math
import numpy as np
import csv
from chainer import cuda, Variable, FunctionSet, optimizers
import chainer.functions as F
from lstm import LSTM, make_initial_state

# input data
def load_data():
    ft = open('train_feature.dump','rb')
    train_data = pickle.load(ft)
    ft.close()
    fv = open('val_feature.dump','rb')
    val_data = pickle.load(fv)[:160]
    fv.close()
    return train_data,val_data

# arguments
'''
実行する時の引数を指定する
(ex)
parser.add_argument('-d','--data_dir',type=str,default='data/tinyshakespeare')
→オプションのパスを-dまたは--data_dirとし，データ型はstring，defaultの値はdata/tinyshakespeare
'''
parser = argparse.ArgumentParser()
parser.add_argument('--gpu',                        type=int,   default=0)
parser.add_argument('--lstm_size',                  type=int,   default=1045)
parser.add_argument('--learning_rate',              type=float, default=2e-3)
parser.add_argument('--learning_rate_decay',        type=float, default=0.97)
parser.add_argument('--learning_rate_decay_after',  type=int,   default=10)
parser.add_argument('--decay_rate',                 type=float, default=0.95)
parser.add_argument('--dropout',                    type=float, default=0.0)
parser.add_argument('--epochs',                     type=int,   default=1)
parser.add_argument('--grad_clip',                  type=int,   default=5)
parser.add_argument('--init_from',                  type=str,   default='')

args = parser.parse_args()

train_data,val_data = load_data()

n_epochs    = args.epochs
n_units     = args.lstm_size
grad_clip   = args.grad_clip

#LSTMを初期化
model = LSTM(3136, n_units)
if args.gpu >= 0:
    cuda.get_device(args.gpu).use()
    model.to_gpu()

#学習アルゴリズムのセットアップ
optimizer = optimizers.RMSprop(lr=args.learning_rate, alpha=args.decay_rate, eps=1e-8)
optimizer.setup(model.collect_parameters())

whole_len          = len(train_data)
whole_val_len      = len(val_data)
epoch              = 0
start_at           = time.time()
cur_at             = start_at
end_time           = 0
state              = make_initial_state(n_units)
train_loss_all     = []
val_loss_all       = []
iterations_count   = 0

if args.gpu >= 0:
    loss   = Variable(cuda.zeros(()))
    val_loss = Variable(cuda.zeros(()))
    for key, value in state.items():
        value.data = cuda.to_gpu(value.data)
else:
    loss   = Variable(np.zeros((), dtype=np.float32))
    val_loss   = Variable(np.zeros((), dtype=np.float32))

for i in xrange(whole_len*n_epochs):
    for j in xrange(0,len(train_data[i%whole_len])-1):
        x_t = np.array([train_data[i%whole_len][j]])
        y_t = np.array([train_data[i%whole_len][j+1]])
        if args.gpu >=0:
            x_t = cuda.to_gpu(x_t)
            y_t = cuda.to_gpu(y_t)
        state, loss_i = model.forward_one_step(x_t, y_t, state, dropout_ratio=args.dropout)        
        loss +=loss_i
    now = time.time()
    end_time += now-cur_at
    iterations_count +=1
    print 'loss_all='+str(loss.data)
    print '{}, train_loss = {}, time = {:.4f}'.format(iterations_count,loss.data / (len(train_data[i%whole_len])-1), now-cur_at)
    cur_at = now
    optimizer.zero_grads()
    loss.backward()
    loss.unchain_backward() 
    optimizer.clip_grads(grad_clip)
    optimizer.update()
    if (i+1)==(whole_len*n_epochs):
        cuda.cupy.save( 'l1_x_W.npy', model.l1_x.W )
        cuda.cupy.save( 'l1_x_b.npy', model.l1_x.b )
    	cuda.cupy.save( 'l1_h_W.npy', model.l1_h.W )
        cuda.cupy.save( 'l1_h_b.npy', model.l1_h.b )
        cuda.cupy.save( 'l6_W.npy', model.l6.W )
        cuda.cupy.save( 'l6_b.npy', model.l6.b )
    if ((i+1)%whole_len)==0:
        epoch += 1
        train_loss_all.append(loss.data.get()/len(train_data[i%whole_len]))
        for k in xrange(whole_val_len):
            val_state=make_initial_state(n_units)
            for key, value in val_state.items():
                value.data = cuda.to_gpu(value.data)
            for l in xrange(0,len(val_data[k])-1):
                x_t = np.array([val_data[k][l]])
                y_t = np.array([val_data[k][l+1]])
                if args.gpu >=0:
                    x_t = cuda.to_gpu(x_t)
                    y_t = cuda.to_gpu(y_t)
                val_state, loss_i = model.predict(x_t, y_t,val_state)
                val_loss +=loss_i
            val_loss_all.append(val_loss.data.get()/len(val_data[k]))
            if args.gpu >= 0:
                val_loss = Variable(cuda.zeros(()))
            else:
                val_loss = Variable(np.zeros((), dtype=np.float32)) 
        if epoch >= args.learning_rate_decay_after:
            optimizer.lr *= args.learning_rate_decay
            print 'decayed learning rate by a factor {} to {}'.format(args.learning_rate_decay, optimizer.lr)
    if args.gpu >= 0:
        loss = Variable(cuda.zeros(()))
    else:
        loss = Variable(np.zeros((), dtype=np.float32))
print 'train {} iterations'.format(iterations_count)
print 'all_train_time is {:.4f}'.format(end_time)
cuda.cupy.save('train_loss_all.npy',train_loss_all)
cuda.cupy.save('val_loss_all.npy',val_loss_all)
