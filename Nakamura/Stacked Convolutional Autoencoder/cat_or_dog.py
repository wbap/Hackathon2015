#coding: utf-8
import argparse
import numpy as np
from chainer import Variable, FunctionSet, optimizers, cuda
import chainer.functions  as F
from dataset_cat_or_dog import load_dataset

import logging
import os
import time
import brica1

'''
referebce
https://gist.github.com/ktnyt/9a61e3d5b74722824309
'''

def create_result_dir(args):
    if args.restart_from is None:
        result_dir = 'results/' + "original_VGG_mini"
        result_dir += '_' + time.strftime('%Y-%m-%d_%H-%M-%S_')
        result_dir += str(time.time()).replace('.', '')
        if not os.path.exists(result_dir):
            os.makedirs(result_dir)
        log_fn = '%s/log.txt' % result_dir
        logging.basicConfig(
            format='%(asctime)s [%(levelname)s] %(message)s',
            filename=log_fn, level=logging.DEBUG)
        logging.info(args)
    else:
        result_dir = '.'
        log_fn = 'log.txt'
        logging.basicConfig(
            format='%(asctime)s [%(levelname)s] %(message)s',
            filename=log_fn, level=logging.DEBUG)
        logging.info(args)

    return log_fn, result_dir

def _as_mat(x):
    if x.ndim == 2:
        return x
    return x.reshape(len(x), -1)

class SLP(FunctionSet):
    def __init__(self, n_input, n_output):
        super(SLP, self).__init__(
            transform=F.Linear(n_input, n_output)
        )

    def forward(self, x_data, y_data):
        x = Variable(x_data)
        t = Variable(y_data)
        y = F.sigmoid(self.transform(x))
        loss = F.softmax_cross_entropy(y, t)
        accuracy = F.accuracy(y, t)
        return loss, accuracy

    def predict(self, x_data):
        x = Variable(x_data)
        y = F.sigmoid(self.transform(x))
        return y.data


class Autoencoder(FunctionSet):
    def __init__(self, n_input, n_output):
        super(Autoencoder, self).__init__(
            encoder=F.Linear(n_input, n_output),
            decoder=F.Linear(n_output, n_input)
        )

    def forward(self, x_data):
        x = Variable(_as_mat(x_data))
        t = Variable(_as_mat(x_data))
        x = F.dropout(x)
        h = F.sigmoid(self.encoder(x))
        y = F.sigmoid(self.decoder(h))
        loss = F.mean_squared_error(y, t)
        return loss

    def encode(self, x_data):
        x = Variable(_as_mat(x_data))
        h = F.sigmoid(self.encoder(x))
        return h.data


class ConvolutionalAutoencoder(FunctionSet):
    def __init__(self, n_in, n_out, ksize, stride=1, pad=0, wscale=1, bias=0, nobias=False):
        super(ConvolutionalAutoencoder, self).__init__(
            encoder=F.Convolution2D(n_in, n_out, ksize, stride=stride, pad=pad, wscale=wscale, bias=bias, nobias=nobias),
            decoder=F.Convolution2D(n_out, n_in, ksize, stride=stride, pad=pad, wscale=wscale, bias=bias, nobias=nobias)
        )

    def forward(self, x_data, train=True):
        x = Variable(x_data)
        t = Variable(x_data)
        if train:
            x = F.dropout(x)
        h = F.sigmoid(self.encoder(x))
        y = F.sigmoid(self.decoder(h))
        return F.mean_squared_error(y, t)

    def encode(self, x_data):
        x = Variable(x_data)
        h = F.sigmoid(self.encoder(x))
        return h.data

    def decode(self, h_data):
        h = Variable(h_data)
        y = F.sigmoid(self.decoder(h))
        return y.data


class SLPComponent(brica1.Component):
    def __init__(self, n_input, n_output, use_gpu=False):
        super(SLPComponent, self).__init__()
        self.model = SLP(n_input, n_output)
        self.optimizer = optimizers.Adam()

        self.make_in_port("input", n_input)
        self.make_in_port("target", 1)
        self.make_out_port("output", n_output)
        self.make_out_port("loss", 1)
        self.make_out_port("accuracy", 1)

        self.use_gpu = use_gpu

        if self.use_gpu:
            self.model.to_gpu()

        self.optimizer.setup(self.model)
        # self.optimizer.setup(self.model.collect_parameters())


    def fire(self):
        x_data = self.inputs["input"].astype(np.float32)
        t_data = self.inputs["target"].astype(np.int32)

        if self.use_gpu:
            x_data = cuda.to_gpu(x_data)
            t_data = cuda.to_gpu(t_data)

        self.optimizer.zero_grads()
        loss, accuracy = self.model.forward(x_data, t_data)
        loss.backward()
        self.optimizer.update()

        y_data = self.model.predict(x_data)

        self.results["loss"] = cuda.to_cpu(loss.data)
        self.results["accuracy"] = cuda.to_cpu(accuracy.data)
        self.results["output"] = cuda.to_cpu(y_data)


class AutoencoderComponent(brica1.Component):
    def __init__(self, n_input, n_output, use_gpu=False):
        super(AutoencoderComponent, self).__init__()
        self.model = Autoencoder(n_input, n_output)
        self.optimizer = optimizers.Adam()

        self.make_in_port("input", n_input)
        self.make_out_port("output", n_output)
        self.make_out_port("loss", 1)

        self.use_gpu = use_gpu

        if self.use_gpu:
            self.model.to_gpu()

        self.optimizer.setup(self.model.collect_parameters())

    def fire(self):
        x_data = self.inputs["input"].astype(np.float32)

        if self.use_gpu:
            x_data = cuda.to_gpu(x_data)

        self.optimizer.zero_grads()
        loss = self.model.forward(x_data)
        loss.backward()
        self.optimizer.update()

        y_data = self.model.encode(x_data)

        self.results["loss"] = cuda.to_cpu(loss.data)
        self.results["output"] = cuda.to_cpu(y_data)


class ConvolutionalAutoencoderComponent(brica1.Component):
    def __init__(self, n_input, n_output, ksize, stride=1, pad=0, wscale=1, bias=0, nobias=False, use_gpu=False):
        super(ConvolutionalAutoencoderComponent, self).__init__()
        self.model = ConvolutionalAutoencoder(n_input, n_output, ksize, stride, pad, wscale, bias, nobias) #stride=1, pad=1?
        self.optimizer = optimizers.Adam()

        self.make_in_port("input", n_input)
        self.make_out_port("output", n_output)
        self.make_out_port("loss", 1)

        self.use_gpu = use_gpu

        if self.use_gpu:
            self.model.to_gpu()

        self.optimizer.setup(self.model.collect_parameters())

    def fire(self):
        x_data = self.inputs["input"].astype(np.float32)

        if self.use_gpu:
            x_data = cuda.to_gpu(x_data)

        self.optimizer.zero_grads()
        loss = self.model.forward(x_data)
        loss.backward()
        self.optimizer.update()

        y_data = self.model.encode(x_data)

        self.results["loss"] = cuda.to_cpu(loss.data)
        self.results["output"] = cuda.to_cpu(y_data)


class PoolingAndDropout(brica1.Component):
    def __init__(self, n_input, n_output):
        super(PoolingAndDropout, self).__init__()
        self.make_in_port("input", n_input)
        self.make_out_port("output", n_output)

    def fire(self):
        x_data = self.inputs["input"].astype(np.float32)
        x = Variable(x_data)

        y = F.max_pooling_2d(x, 2, stride=2)
        y = F.dropout(y, ratio=0.25, train=True)

        self.results["output"] = cuda.to_cpu(y.data)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Chainer-BriCa integration")
    parser.add_argument("--gpu", "-g", default=-1, type=int, help="GPU ID")
    # parser.add_argument('--model', type=str, default='models/VGG_mini_ABN.py')
    parser.add_argument('--epoch', type=int, default=100)
    parser.add_argument('--batchsize', type=int, default=128)
    parser.add_argument('--prefix', type=str,
                        default='VGG_mini_ABN_Adam')
    parser.add_argument('--snapshot', type=int, default=10)
    parser.add_argument('--restart_from', type=str)
    parser.add_argument('--epoch_offset', type=int, default=0)
    parser.add_argument('--datadir', type=str, default='data')
    parser.add_argument('--flip', type=int, default=1)
    parser.add_argument('--shift', type=int, default=10)
    parser.add_argument('--size', type=int, default=28)
    parser.add_argument('--norm', type=int, default=0)
    parser.add_argument('--opt', type=str, default='Adam',
                        choices=['MomentumSGD', 'Adam', 'AdaGrad'])
    parser.add_argument('--weight_decay', type=float, default=0.0005)
    parser.add_argument('--alpha', type=float, default=0.001)
    parser.add_argument('--lr', type=float, default=0.01)
    parser.add_argument('--lr_decay_freq', type=int, default=100)
    parser.add_argument('--lr_decay_ratio', type=float, default=0.1)
    parser.add_argument('--seed', type=int, default=1701)

    args = parser.parse_args()

    batchsize = args.batchsize

    # create result dir
    log_fn, result_dir = create_result_dir(args)

    use_gpu=False
    if args.gpu >= 0:
        use_gpu = True
        cuda.get_device(args.gpu).use()

    # batchsize = 100
    # n_epoch = 1

    # mnist = data.load_mnist_data()
    # mnist['data'] = mnist['data'].astype(np.float32)
    # mnist['data'] /= 255
    # mnist['target'] = mnist['target'].astype(np.int32)
    #
    # N_train = 60000
    # x_train, x_test = np.split(mnist['data'],   [N_train])
    # y_train, y_test = np.split(mnist['target'], [N_train])
    # N_test = y_test.size

    # create model and optimizer
    dataset = load_dataset(args.datadir)
    train_data, train_labels, test_data, test_labels = dataset
    train_data = train_data.astype(np.float32) / 255.0
    train_labels = train_labels.astype(np.int32)
    test_data = test_data.astype(np.float32) / 255.0
    test_labels = test_labels.astype(np.int32)
    N_train = train_data.shape[0]
    N_test = test_data.shape[0]

    convolutional1 = ConvolutionalAutoencoderComponent(3,   64,  3, stride=1, pad=1, use_gpu=use_gpu)
    convolutional2 = ConvolutionalAutoencoderComponent(64,  64,  3, stride=1, pad=1, use_gpu=use_gpu)
    poolinganddropout1 = PoolingAndDropout(4096, 4096)
    convolutional3 = ConvolutionalAutoencoderComponent(64,  128, 3, stride=1, pad=1, use_gpu=use_gpu)
    convolutional4 = ConvolutionalAutoencoderComponent(128, 128, 3, stride=1, pad=1, use_gpu=use_gpu)
    poolinganddropout2 = PoolingAndDropout(4096, 4096)
    convolutional5 = ConvolutionalAutoencoderComponent(128, 256, 3, stride=1, pad=1, use_gpu=use_gpu)
    convolutional6 = ConvolutionalAutoencoderComponent(256, 256, 3, stride=1, pad=1, use_gpu=use_gpu)
    convolutional7 = ConvolutionalAutoencoderComponent(256, 256, 3, stride=1, pad=1, use_gpu=use_gpu)
    convolutional8 = ConvolutionalAutoencoderComponent(256, 256, 3, stride=1, pad=1, use_gpu=use_gpu)
    poolinganddropout3 = PoolingAndDropout(4096, 4096)
    autoencoder1   = AutoencoderComponent(4096, 1024, use_gpu=use_gpu)
    autoencoder2   = AutoencoderComponent(1024, 1024, use_gpu=use_gpu)
    slp = SLPComponent(1024, 10, use_gpu=use_gpu)

    brica1.connect((convolutional1, "output"), (convolutional2, "input"))
    brica1.connect((convolutional2, "output"), (poolinganddropout1, "input"))
    brica1.connect((poolinganddropout1, "output"), (convolutional3, "input"))
    brica1.connect((convolutional3, "output"), (convolutional4, "input"))
    brica1.connect((convolutional4, "output"), (poolinganddropout2, "input"))
    brica1.connect((poolinganddropout2, "output"), (convolutional5, "input"))
    brica1.connect((convolutional5, "output"), (convolutional6, "input"))
    brica1.connect((convolutional6, "output"), (convolutional7, "input"))
    brica1.connect((convolutional7, "output"), (convolutional8, "input"))
    brica1.connect((convolutional8, "output"), (poolinganddropout3,   "input"))
    brica1.connect((poolinganddropout3, "output"), (autoencoder1, "input"))
    brica1.connect((autoencoder1,   "output"), (autoencoder2,   "input"))
    brica1.connect((autoencoder2,   "output"), (slp, "input"))

    stacked_autoencoder = brica1.ComponentSet()
    stacked_autoencoder.add_component("convolutional1", convolutional1, 1)
    stacked_autoencoder.add_component("convolutional2", convolutional2, 2)
    stacked_autoencoder.add_component("poolinganddropout1", poolinganddropout1, 3)
    stacked_autoencoder.add_component("convolutional3", convolutional3, 4)
    stacked_autoencoder.add_component("convolutional4", convolutional4, 5)
    stacked_autoencoder.add_component("poolinganddropout2", poolinganddropout2, 6)
    stacked_autoencoder.add_component("convolutional5", convolutional5, 7)
    stacked_autoencoder.add_component("convolutional6", convolutional6, 8)
    stacked_autoencoder.add_component("convolutional7", convolutional7, 9)
    stacked_autoencoder.add_component("convolutional8", convolutional8, 10)
    stacked_autoencoder.add_component("poolinganddropout3", poolinganddropout3, 11)
    stacked_autoencoder.add_component("autoencoder1", autoencoder1, 12)
    stacked_autoencoder.add_component("autoencoder2", autoencoder2, 13)
    stacked_autoencoder.add_component("slp", slp, 14)

    stacked_autoencoder.make_in_port("input", 3*64*3) #? convAutoにいれる値は3つある -　関係ないらしい
    stacked_autoencoder.make_out_port("output", 10)
    stacked_autoencoder.make_out_port("loss1", 1)
    stacked_autoencoder.make_out_port("loss2", 1)
    stacked_autoencoder.make_out_port("loss3", 1)
    stacked_autoencoder.make_out_port("loss4", 1)
    stacked_autoencoder.make_out_port("loss5", 1)
    stacked_autoencoder.make_out_port("loss6", 1)
    stacked_autoencoder.make_out_port("loss7", 1)
    stacked_autoencoder.make_out_port("loss8", 1)
    stacked_autoencoder.make_out_port("loss9", 1)
    stacked_autoencoder.make_out_port("loss10", 1)
    stacked_autoencoder.make_out_port("loss11", 1)
    stacked_autoencoder.make_in_port("target", 1)
    stacked_autoencoder.make_out_port("accuracy", 1)

    brica1.alias_in_port((stacked_autoencoder, "input"), (convolutional1, "input"))
    brica1.alias_out_port((stacked_autoencoder, "output"), (slp, "output"))
    brica1.alias_out_port((stacked_autoencoder, "loss1"), (convolutional1, "loss"))
    brica1.alias_out_port((stacked_autoencoder, "loss2"), (convolutional2, "loss"))
    brica1.alias_out_port((stacked_autoencoder, "loss3"), (convolutional3, "loss"))
    brica1.alias_out_port((stacked_autoencoder, "loss4"), (convolutional4, "loss"))
    brica1.alias_out_port((stacked_autoencoder, "loss5"), (convolutional5, "loss"))
    brica1.alias_out_port((stacked_autoencoder, "loss6"), (convolutional6, "loss"))
    brica1.alias_out_port((stacked_autoencoder, "loss7"), (convolutional7, "loss"))
    brica1.alias_out_port((stacked_autoencoder, "loss8"), (convolutional8, "loss"))
    brica1.alias_out_port((stacked_autoencoder, "loss9"), (autoencoder1, "loss"))
    brica1.alias_out_port((stacked_autoencoder, "loss10"), (autoencoder2, "loss"))
    brica1.alias_out_port((stacked_autoencoder, "loss11"), (slp, "loss"))
    brica1.alias_in_port((stacked_autoencoder, "target"), (slp, "target"))
    brica1.alias_out_port((stacked_autoencoder, "accuracy"), (slp, "accuracy"))

    scheduler = brica1.VirtualTimeSyncScheduler()
    agent = brica1.Agent(scheduler)
    module = brica1.Module()
    module.add_component("stacked_autoencoder", stacked_autoencoder)
    agent.add_submodule("module", module)

    time = 0.0

    logging.info('start training...')

    for epoch in xrange(args.epoch):
        perm = np.random.permutation(N_train)
        sum_loss1 = 0
        sum_loss2 = 0
        sum_loss3 = 0
        sum_loss4 = 0
        sum_loss5 = 0
        sum_loss6 = 0
        sum_loss7 = 0
        sum_loss8 = 0
        sum_loss9 = 0
        sum_loss10 = 0
        sum_loss11 = 0
        sum_accuracy = 0

        for batchnum in xrange(0, N_train, args.batchsize):
            x_batch = train_data[perm[batchnum:batchnum+args.batchsize]]
            y_batch = train_labels[perm[batchnum:batchnum+args.batchsize]]

            stacked_autoencoder.get_in_port("input").buffer = x_batch
            stacked_autoencoder.get_in_port("target").buffer = y_batch

            time = agent.step()

            loss1 = stacked_autoencoder.get_out_port("loss1").buffer
            loss2 = stacked_autoencoder.get_out_port("loss2").buffer
            loss3 = stacked_autoencoder.get_out_port("loss3").buffer
            loss4 = stacked_autoencoder.get_out_port("loss4").buffer
            loss5 = stacked_autoencoder.get_out_port("loss5").buffer
            loss6 = stacked_autoencoder.get_out_port("loss6").buffer
            loss7 = stacked_autoencoder.get_out_port("loss7").buffer
            loss8 = stacked_autoencoder.get_out_port("loss8").buffer
            loss9 = stacked_autoencoder.get_out_port("loss9").buffer
            loss10 = stacked_autoencoder.get_out_port("loss10").buffer
            loss11 = stacked_autoencoder.get_out_port("loss11").buffer
            accuracy = stacked_autoencoder.get_out_port("accuracy").buffer

            msg = "Time: {}\tLoss1: {}\tLoss2: {}\tLoss3: {}\tLoss4: {}\tLoss5: {}\tLoss6: {}\tLoss7: {}\tLoss8: {}\tLoss9: {}\tLoss10: {}\tLoss11: {}\tAccuracy: {}".format(time, loss1, loss2, loss3, loss4, loss5, loss6, loss7, loss8, loss9, loss10, loss11, accuracy)
            # logging.info(msg)
            print msg

            sum_loss1 += loss1 * batchsize
            sum_loss2 += loss2 * batchsize
            sum_loss3 += loss3 * batchsize
            sum_loss4 += loss4 * batchsize
            sum_loss5 += loss5 * batchsize
            sum_loss6 += loss6 * batchsize
            sum_loss7 += loss7 * batchsize
            sum_loss8 += loss8 * batchsize
            sum_loss9 += loss9 * batchsize
            sum_loss10 += loss10 * batchsize
            sum_loss11 += loss11 * batchsize
            sum_accuracy += accuracy * batchsize

        mean_loss1 = sum_loss1 / N_train
        mean_loss2 = sum_loss2 / N_train
        mean_loss3 = sum_loss3 / N_train
        mean_loss4 = sum_loss4 / N_train
        mean_loss5 = sum_loss5 / N_train
        mean_loss6 = sum_loss6 / N_train
        mean_loss7 = sum_loss7 / N_train
        mean_loss8 = sum_loss8 / N_train
        mean_loss9 = sum_loss9 / N_train
        mean_loss10 = sum_loss10 / N_train
        mean_loss11 = sum_loss11 / N_train
        mean_accuracy = sum_accuracy / N_train

        msg = "Epoch:{}\tLoss1: {}\tLoss2: {}\tLoss3: {}\tLoss4: {}\tLoss5: {}\tLoss6: {}\tLoss7: {}\tLoss8: {}\tLoss9: {}\tLoss10: {}\tLoss11: {}\tAccuracy: {}".format(epoch, mean_loss1, mean_loss2, mean_loss3, mean_loss4, mean_loss5, mean_loss6, mean_loss7, mean_loss8, mean_loss9, mean_loss10, mean_loss11, mean_accuracy)
        logging.info(msg)
        print msg
