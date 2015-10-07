#/usr/bin/env python
#-*- coding: utf-8 -*-
from chainer import Variable, FunctionSet
import chainer.functions as F


"""
ベースラインCNN
CNN-LSTMのLSTM層のみをカット、最終層のFull Connectionで連結
"""
class syuwa_cnn(FunctionSet):

    def __init__(self, num_labels):
        super(syuwa_cnn, self).__init__(
            conv11=F.Convolution2D(1, 16, 6, stride=2),
            bn11=F.BatchNormalization(16),
            conv12=F.Convolution2D(16, 32, 5, stride=1),
            bn12=F.BatchNormalization(32),
            conv13=F.Convolution2D(32, 48, 3, stride=1),
            fc14=F.Linear(1728, 512),
            fc15=F.Linear(512, 200),
            conv21=F.Convolution2D(1, 16, 6, stride=2),
            bn21=F.BatchNormalization(16),
            conv22=F.Convolution2D(16, 32, 5, stride=1),
            bn22=F.BatchNormalization(32),
            conv23=F.Convolution2D(32, 48, 3, stride=1),
            fc24=F.Linear(1728, 512),
            fc25=F.Linear(512, 200),
            fc_comb =F.Linear(400, num_labels),
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


    def forward(self, x_vis, x_dep, train_label, volatile=False):
        
        x1 = Variable(x_vis.reshape(x_vis.shape[0], 1, x_vis.shape[1], x_vis.shape[2]), volatile=volatile)
        h1 = F.max_pooling_2d(F.relu(self.bn11(self.conv11(x1))), 2, stride=2)
        h1 = F.max_pooling_2d(F.relu(self.bn12(self.conv12(h1))), 2, stride=2)
        h1 = F.max_pooling_2d(F.relu(self.conv13(h1)), 2, stride=2)
        h1 = self.fc14(h1)
        h1 = self.fc15(h1)

        x2 = Variable(x_dep.reshape(x_vis.shape[0], 1, x_dep.shape[1], x_dep.shape[2]), volatile=volatile)
        h2 = F.max_pooling_2d(F.relu(self.bn21(self.conv21(x2))), 2, stride=2)
        h2 = F.max_pooling_2d(F.relu(self.bn22(self.conv22(h2))), 2, stride=2)
        h2 = F.max_pooling_2d(F.relu(self.conv23(h2)), 2, stride=2)
        h2 = self.fc24(h2)
        h2 = self.fc25(h2)
        
        # 可視CNNとDepthCNNの出力を連結
        last_input = F.concat((h1, h2), axis=1)
        t = Variable(train_label, volatile=volatile)

        y = self.fc_comb(last_input)
        
        return F.softmax_cross_entropy(y, t), y
