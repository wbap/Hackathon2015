#!/usr/bin/env python
# -*- coding: utf-8 -*-

from chainer import Variable, FunctionSet
import chainer.functions as F

"""
手話認識用CNN-LSTM
可視CNN、DepthCNNの2種類のCNNの出力をLSTMの入力とする
"""
class syuwa_cnn_lstm(FunctionSet):

    def __init__(self, num_iunits, num_hunits, num_labels):
        super(syuwa_cnn_lstm, self).__init__(
            conv11=F.Convolution2D(1, 16, 6, stride=2),
            bn11=F.BatchNormalization(16),
            conv12=F.Convolution2D(16, 32, 5, stride=1),
            bn12=F.BatchNormalization(32),
            conv13=F.Convolution2D(32, 48, 3, stride=1),
            fc14=F.Linear(1728, 512),
            fc15=F.Linear(512, num_iunits / 2),
            conv21=F.Convolution2D(1, 16, 6, stride=2),
            bn21=F.BatchNormalization(16),
            conv22=F.Convolution2D(16, 32, 5, stride=1),
            bn22=F.BatchNormalization(32),
            conv23=F.Convolution2D(32, 48, 3, stride=1),
            fc24=F.Linear(1728, 512),
            fc25=F.Linear(512, num_iunits / 2),
            i2h = F.Linear(num_iunits, num_hunits * 4),
            h2h = F.Linear(num_hunits, num_hunits * 4),
            h2y = F.Linear(num_hunits, num_labels),
        )

    """
    pool1に適当な画像を入れたときの出力を返す
    出力：Variable
    """
    def extract_pool11(self, x_vis):
        x1 = Variable(x_vis.reshape(1, 1, x_vis.shape[0], x_vis.shape[1]), volatile=True)
        h1 = F.max_pooling_2d(F.relu(self.bn11(self.conv11(x1))), 2, stride=2)

        return h1

    def extract_pool21(self, x_dep):
        x2 = Variable(x_dep.reshape(1, 1, x_dep.shape[0], x_dep.shape[1]), volatile=True)
        h2 = F.max_pooling_2d(F.relu(self.bn21(self.conv21(x2))), 2, stride=2)

        return h2

    """
    CNN-LSTMのComputational Graphを記述
    入力：x_vis(width, height)、x_dep(width, height)
    """
    def forward_one_step(self, x_vis, x_dep, train_label, c, h, volatile=False):
        x1 = Variable(x_vis.reshape(1, 1, x_vis.shape[0], x_vis.shape[1]), volatile=volatile)
        h1 = F.max_pooling_2d(F.relu(self.bn11(self.conv11(x1))), 2, stride=2)
        h1 = F.max_pooling_2d(F.relu(self.bn12(self.conv12(h1))), 2, stride=2)
        h1 = F.max_pooling_2d(F.relu(self.conv13(h1)), 2, stride=2)
        h1 = self.fc14(h1)
        h1 = self.fc15(h1)

        x2 = Variable(x_dep.reshape(1, 1, x_dep.shape[0], x_dep.shape[1]), volatile=volatile)
        h2 = F.max_pooling_2d(F.relu(self.bn21(self.conv21(x2))), 2, stride=2)
        h2 = F.max_pooling_2d(F.relu(self.bn22(self.conv22(h2))), 2, stride=2)
        h2 = F.max_pooling_2d(F.relu(self.conv23(h2)), 2, stride=2)
        h2 = self.fc24(h2)
        h2 = self.fc25(h2)
        
        # 可視CNNとDepthCNNの出力を連結
        lstm_input = F.concat((h1, h2), axis=1)
        t = Variable(train_label, volatile=volatile)

        h_in = self.i2h(F.dropout(lstm_input, train=not volatile)) + self.h2h(h)
        c, h = F.lstm(c, h_in)

        y = self.h2y(F.dropout(h, train=not volatile))
        return F.softmax_cross_entropy(y, t), y, c, h
