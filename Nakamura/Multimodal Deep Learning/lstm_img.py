import numpy as np
import chainer
import chainer.functions as F
from chainer import Variable,FunctionSet,optimizers,cuda
import argparse
import convert
import csv

class LSTMmodel(FunctionSet):
    def __init__(self,n_input,n_hidden,n_output):
        super(LSTMmodel,self).__init__(
                                 l1_x = F.Linear(n_input ,  4*n_hidden),
                                 l1_h = F.Linear(n_hidden , 4*n_hidden),
                                 l2_x = F.Linear(n_hidden , 4*n_hidden),
                                 l2_h = F.Linear(n_hidden , 4*n_hidden),
                                 l3 = F.Linear(n_hidden , n_output)
                                 )
    

    def forward(self,x_data,y_data,state,train=True):
        x = Variable(x_data, volatile = not train)
        t = Variable(y_data)
        h1_in = self.l1_x(F.dropout(x,train=train)) + self.l1_h(state['h1'])
        c1,h1 = F.lstm(state['c1'],h1_in)
        h2_in = self.l2_x(F.dropout(h1,train=train)) + self.l2_h(state['h2'])
        c2,h2 = F.lstm(state['c2'],h2_in)
        y = self.l3(F.dropout(h2,train=train))
        state = {'c1':c1, 'h1':h1, 'c2':c2, 'h2':h2 }
        Loss = F.softmax_cross_entropy(y,t)
        accuracy = F.accuracy(y,t)
        
        return state,Loss,accuracy,y.data,t.data

class Autoencoder(FunctionSet):
    def __init__(self,n_input,n_output):
        super(Autoencoder,self).__init__(
                                         encoder = F.Linear(n_input , n_output),
                                         decoder = F.Linear(n_output , n_input)
                                         )
    
    def forward(self,x_data):
        x = Variable(x_data)
        x = F.dropout(x)
        y = F.sigmoid(self.encoder(x))
        y_hat = F.sigmoid(self.decoder(y))
        Loss = F.mean_squared_error(y_hat,x)
        
        return Loss

def make_initial_state(batchsize,n_units,train=True):
    return {name: chainer.Variable(xp.zeros((batchsize,n_units),dtype=np.float32),volatile=not train) for name in ('c1','h1','c2','h2')}

def evaluate(ydata,tdata):
    while True:
        if isinstance(ydata[0],int):
            break
                
        t = 15
        num = len(ydata)*len(ydata[0])/t
        ydata = xp.array(ydata)
        tdata = xp.array(tdata)
        
        ydata_reshape = xp.array(ydata).reshape((num,t))
        tdata_reshape = xp.array(tdata).reshape((num,t))

        tdata_last_all = tdata_reshape[:,14]
        ydata_last_all = ydata_reshape[:,14]
        
        allmatched_list = []
        for i in range(len(ydata_last_all)):
            if ydata_last_all[i] == tdata_last_all[i]:
                allmatched_list.append(ydata_last_all[i])
 
        size = 200
        start = tdata_reshape.shape[0] - size

        tdata_last = tdata_reshape[start:start+size,14]
        ydata_last = ydata_reshape[start:start+size,14]
    
        matched_list = []
        for i in range(len(ydata_last)):
            if ydata_last[i] == tdata_last[i]:
                matched_list.append(ydata_last[i])



        accuracy_last_all = float(len(allmatched_list))/float(len(ydata_last_all))
        accuracy_last = float(len(matched_list))/float(len(ydata_last))
        return accuracy_last_all,accuracy_last



if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu", "-g", default=-1, type=int, help="GPU ID")
    
    args = parser.parse_args()
    
    xp = cuda.cupy if args.gpu >= 0 else np
    
    use_gpu = False
    if args.gpu >= 0:
        use_gpu = True
        cuda.check_cuda_available()
        cuda.get_device(args.gpu).use()
    

    batchsize = 30
    n_epoch = 100
    n_units = 400

    #loding dataset

    imgdata = np.loadtxt("./img.csv",delimiter=",")
    
    img = {}
    img['data'] = imgdata
    img['data'] = img['data'].astype(np.float32)
    img['data'] /= 255
    img['target'] = convert.gen_target()
    img['target'] = img['target'].astype(np.int32)
    
    
    N_train = 1500
    x_train, x_test = np.split(img['data'],   [N_train])
    y_train, y_test = np.split(img['target'], [N_train])
    N_test = y_test.size
    
    
    
    state = make_initial_state(batchsize,n_units)

    model = LSTMmodel(len(x_train[0]),n_units,10)

    if use_gpu:
        model.to_gpu()
        x_train = cuda.to_gpu(x_train)
        x_test  = cuda.to_gpu(x_test)
        y_train = cuda.to_gpu(y_train)
        y_test  = cuda.to_gpu(y_test)

    optimizer = optimizers.Adam()
    optimizer.setup(model.collect_parameters())

    l1_x_W = []
    l1_h_W = []
    l2_x_W = []
    l2_h_W = []
    l3_W = []

    y_array = []
    y_each = []
    t_array = []
        
    y_trainbackup = []
    t_trainbackup = []
    y_testbackup = []
    t_testbackup = []

    trainloss = []
    trainaccuracy = []
    trainaccuracylast = []

    testloss = []
    testaccuracy = []
    testaccuracylast = []

    for epoch in xrange(n_epoch):
        #perm = np.random.permutation(N_train)
        sum_accuracy = 0
        sum_Loss = 0

        for batchnum in xrange(0 , N_train, batchsize):
            x_batch = xp.array(x_train[batchnum:batchnum+batchsize])
            y_batch = xp.array(y_train[batchnum:batchnum+batchsize])
            

            if use_gpu:
                x_batch = cuda.to_gpu(x_batch)
                y_batch  = cuda.to_gpu(y_batch)
            
            optimizer.zero_grads()
            state, Loss, accuracy, y, t = model.forward(x_batch, y_batch, state)
            Loss.backward()
            optimizer.update()

            sum_accuracy += accuracy.data * batchsize
            sum_Loss += Loss.data * batchsize
        
            t_array.append(t)
            t_trainbackup = xp.array(t_array)
            
            for i in range(batchsize):
                y_each.append(int(np.argmax(y[i])))
            
            y_array.append(y_each)
            y_trainbackup = xp.array(y_array)
            y_each = []

            with open('./imgdatalog3/tdata_train.csv','a') as f:
                writer = csv.writer(f)
                writer.writerows(t_array)
                t_array = []
            
            with open('./imgdatalog3/ydata_train.csv','a') as f:
                writer = csv.writer(f)
                writer.writerows(y_array)
                y_array = []

        mean_accuracy = sum_accuracy/N_train
        mean_Loss = sum_Loss / N_train
        #accuracy_last = evaluate(y_trainbackup,t_trainbackup)
        accuracy_last = None


        if use_gpu:
            mean_accuracy = cuda.to_cpu(mean_accuracy)
            mean_Loss = cuda.to_cpu(mean_Loss)
            accuracy_last = cuda.to_cpu(accuracy_last)


        print "Train Epoch {} : Loss {} : Accuracy {}  accuracy at t=15 : {}".format(epoch,mean_Loss,mean_accuracy,accuracy_last)

        trainloss.append(mean_Loss)
        trainaccuracy.append(mean_accuracy)
        trainaccuracylast.append(accuracy_last)

        with open('./imgdatalog3/train_loss.csv','w') as f:
            writer = csv.writer(f)
            writer.writerow(trainloss)

        with open('./imgdatalog3/train_accuracy.csv','w') as f:
            writer = csv.writer(f)
            writer.writerow(trainaccuracy)


        sum_accuracy = 0
        sum_Loss = 0

        for batchnum in xrange(0 , N_test, batchsize):
            x_batch = xp.array(x_test[batchnum:batchnum+batchsize])
            y_batch = xp.array(y_test[batchnum:batchnum+batchsize])
            
            if use_gpu:
                x_batch = cuda.to_gpu(x_batch)
                y_batch  = cuda.to_gpu(y_batch)
        
            state, Loss, accuracy, y, t = model.forward(x_batch,y_batch,state)
            
            sum_accuracy += accuracy.data * batchsize
            sum_Loss += Loss.data * batchsize
            
            t_array.append(t)
            t_testbackup = xp.array(t_array)
            
            for i in range(batchsize):
                y_each.append(int(np.argmax(y[i])))

            y_array.append(y_each)
            y_testbackup = xp.array(y_array)
            y_each = []

            with open('./imgdatalog3/tdata_test.csv','a') as f:
                writer = csv.writer(f)
                writer.writerows(t_array)
                t_array = []
            
            with open('./imgdatalog3/ydata_test.csv','a') as f:
                writer = csv.writer(f)
                writer.writerows(y_array)
                y_array = []


        mean_accuracy = sum_accuracy/N_test
        mean_Loss = sum_Loss / N_test
        accuracy_last = None


        if use_gpu:
            mean_accuracy = cuda.to_cpu(mean_accuracy)
            mean_Loss = cuda.to_cpu(mean_Loss)
            accuracy_last = cuda.to_cpu(accuracy_last)

        l1_x_W.append(model.l1_x.W)
        l1_h_W.append(model.l1_h.W)
        l2_x_W.append(model.l2_x.W)
        l2_h_W.append(model.l2_h.W)
        l3_W.append(model.l3.W)
        testloss.append(mean_Loss)
        testaccuracy.append(mean_accuracy)
        testaccuracylast.append(accuracy_last)



        with open('./imgdatalog3/l1_x_W.csv','a') as f:
            writer = csv.writer(f)
            writer.writerows(l1_x_W)
            l1_x_W = []

        with open('./imgdatalog3/l1_h_W.csv','a') as f:
            writer = csv.writer(f)
            writer.writerows(l1_h_W)
            l1_h_W = []

        with open('./imgdatalog3/l2_x_W.csv','a') as f:
            writer = csv.writer(f)
            writer.writerows(l2_x_W)
            l2_x_W = []

        with open('./imgdatalog3/l2_h_W.csv','a') as f:
            writer = csv.writer(f)
            writer.writerows(l2_h_W)
            l2_h_W = []

        with open('./imgdatalog3/l3_W.csv','a') as f:
            writer = csv.writer(f)
            writer.writerows(l3_W)
            l3_W = []

        with open('./imgdatalog3/test_loss.csv','w') as f:
            writer = csv.writer(f)
            writer.writerow(testloss)

        with open('./imgdatalog3/test_accuracy.csv','w') as f:
            writer = csv.writer(f)
            writer.writerow(testaccuracy)


        print "Test Epoch {} : Loss {} : Accuracy {}  accuracy at t=15 : {}".format(epoch,mean_Loss,mean_accuracy,accuracy_last)








