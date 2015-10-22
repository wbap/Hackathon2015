#coding:utf-8
import numpy as np
from chainer import Variable, FunctionSet
import chainer.functions as F

class LSTM(FunctionSet):
 
    def __init__(self,f_n_units, n_units):
        super(LSTM, self).__init__(
            l1_x = F.Linear(f_n_units, 4*n_units),
            l1_h = F.Linear(n_units, 4*n_units),
            l6   = F.Linear(n_units, f_n_units)
        )
        # パラメータの値を-0.08~0.08の範囲で初期化
        for param in self.parameters:
            param[:] = np.random.uniform(-0.08, 0.08, param.shape)
    
    def forward_one_step(self, x_data, y_data, state, train=True,dropout_ratio=0.0):
        x ,t = Variable(x_data,volatile=not train),Variable(y_data,volatile=not train)
        h1_in   = self.l1_x(F.dropout(x, ratio=dropout_ratio, train=train)) + self.l1_h(state['h1'])
        c1, h1  = F.lstm(state['c1'], h1_in)
        y       = self.l6(F.dropout(h1, ratio=dropout_ratio, train=train))
        state   = {'c1': c1, 'h1': h1}
        return state, F.mean_squared_error(y, t)
 
    def predict(self, x_data, y_data, state):
        x ,t = Variable(x_data,volatile=False),Variable(y_data,volatile=False)
        h1_in   = self.l1_x(x) + self.l1_h(state['h1'])
        c1, h1  = F.lstm(state['c1'], h1_in)
        y       = self.l6(h1)
        state   = {'c1': c1, 'h1': h1}
        return state,F.mean_squared_error(y,t)
 
def make_initial_state(n_units,train = True):
    return {name: Variable(np.zeros((1,n_units), dtype=np.float32),
            volatile=not train)
            for name in ('c1', 'h1')}
            #for name in ('c1', 'h1', 'c2', 'h2', 'c3', 'h3','c4','h4','c5','h5')}
