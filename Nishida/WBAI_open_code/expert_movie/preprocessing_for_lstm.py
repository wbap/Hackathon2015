# -*- coding: utf-8 -*-  

import copy

import pickle
import numpy as np
import scipy.misc as spm

from chainer import cuda, FunctionSet, Variable, optimizers
import chainer.functions as F

from rlglue.agent.Agent import Agent
from rlglue.agent import AgentLoader as AgentLoader
from rlglue.types import Action


class CNN_class:
    def __init__(self):
        self.model = FunctionSet(
            l1 = F.Convolution2D(4, 32, ksize=8, stride=4, nobias=False, wscale=np.sqrt(2)),
            l2 = F.Convolution2D(32, 64, ksize=4, stride=2, nobias=False, wscale=np.sqrt(2)),
            l3 = F.Convolution2D(64, 64, ksize=3, stride=1, nobias=False, wscale=np.sqrt(2))
)
        
        self.model.l1.W = np.load('elite/l1_W.npy')
        self.model.l1.b = np.load('elite/l1_b.npy')
        self.model.l2.W = np.load('elite/l2_W.npy')
        self.model.l2.b = np.load('elite/l2_b.npy')
        self.model.l3.W = np.load('elite/l3_W.npy')
        self.model.l3.b = np.load('elite/l3_b.npy')


    def CNN_forward(self, state):
        h1 = F.relu(self.model.l1(state / 254.0))
        h2 = F.relu(self.model.l2(h1))
        h3 = F.relu(self.model.l3(h2))

        return h3

if __name__ == '__main__':
    dqn = CNN_class()
    R = np.load('Reward/Reward_test.npy')
    s = np.load('State/Stock_test.npy')
    print 'Shape[s,R] = [', s.shape, ' ',R.shape,']' 
    counter1 = 0
    counter2 = 0
    featureList = []
    for i,r in enumerate(R):
        if r != 0:
            print i
            counter1 = counter2
            counter2 = i
            s_now = s[counter1:counter2]
            List = []
            for S in s_now:

                S = np.asanyarray(S.reshape(1,4,84,84),dtype=np.float32)
                state = Variable(S)
                f = dqn.CNN_forward(state)
                List.append(f.data)
            featureList.append(List)
    counter1 = counter2
    counter2 = len(s)
    s_now = s[counter1:counter2]
    List = []
    for S in s_now:
        S = np.asanyarray(S.reshape(1,4,84,84),dtype=np.float32)
        state = Variable(S)
        f = dqn.CNN_forward(state)
        List.append(f.data)
    featureList.append(List)

    files = open('feature_test.dump','w')
    pickle.dump(featureList, files)
    files.close()
