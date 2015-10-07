#-*- coding: utf-8 -*-
from __future__ import print_function
import sys
sys.path.append('../')
sys.path.append('../../')
import argparse
import logging
import time
import os
import glob
import pickle
from progressbar import ProgressBar
import xml.etree.ElementTree as ET
import csv

import numpy as np
from chainer import optimizers, cuda, Variable
import cv2

from train_new import extract_data, load_data
from cnn1 import syuwa_cnn
from scripts.draw_image import draw_filters, draw_filters_sq, draw_image
from scripts.draw_loss import draw_loss_curve
from scripts.draw_histogram import draw_histogram
from scripts.confmat import save_confmat_fig, print_confmat

xp = np

def create_result_dir(args):
    if args.restart_from is None:
        result_dir = 'results/cnn_only'
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

def get_optimizer(opt):
    # prepare optimizer 
    if opt == 'MomentumSGD':
        optimizer = optimizers.MomentumSGD(lr=args.lr, momentum=0.7)
    elif opt == 'Adam':
        optimizer = optimizers.Adam(alpha=args.alpha)
    elif opt == 'AdaGrad':
        optimizer = optimizers.AdaGrad(lr=args.lr)
    else:
        raise Exception('No optimizer is selected')

    return optimizer
 
def train(train_vis, train_dep, train_labels, N, num_label, model,opt,args): 
    pbar = ProgressBar(N)

    cnn_correct_cnt = 0
    sum_frame = 0
    total_loss = np.array(0, dtype=np.float32)

    opt.setup(model)

    conf_array = np.zeros((num_label, num_label), dtype=np.int32)

    for i in range(0, N):
        x_vis_batch = xp.asarray(train_vis[i], dtype=np.float32)
        x_dep_batch = xp.asarray(train_dep[i], dtype=np.float32)
        y_batch = xp.asarray(train_labels[i], dtype=np.int32).reshape(-1)
        
        sum_frame += y_batch.shape[0]
        
        opt.zero_grads()

        loss, pred = model.forward(
            x_vis_batch, x_dep_batch, y_batch)
       
        pred = xp.argmax(pred.data, axis=1)
        if args.gpu >= 0:
            cnn_correct_cnt += xp.asnumpy(xp.sum(pred == y_batch))
        else:
            cnn_correct_cnt += np.sum(pred == y_batch)
        
        if y_batch.size == 1:
            conf_array[y_batch, pred] += 1
        else:
            for j in xrange(y_batch.size):
                conf_array[y_batch[j], pred[j]] += 1

        loss.backward()
        opt.update()

        if args.opt in ['AdaGrad', 'MomentumSGD']:
            opt.weight_decay(decay=args.weight_decay)

        pbar.update(i + 1 if (i + 1) < N else N)

        if args.gpu >= 0:
            import cupy
            total_loss += cupy.asnumpy(loss.data)
        else:
            total_loss += loss.data

    """
    正答率計算 
    """
    cnn_accuracy = cnn_correct_cnt / float(sum_frame)

    return total_loss / sum_frame, cnn_accuracy, conf_array

"""                                                                                                               
入力の正規化                                                                                                      
今回は動画全体に対してLocal Contrast Normalizationを行う                                                          
入力：ndarray(T, width, height)                                                                                   
"""
def norm(x):
    if not x.dtype == np.float32:
     x = x.astype(np.float32)
                                                                   
    x = (x - np.mean(x)) / (np.std(x) + np.finfo(np.float32).eps)

    return x

def validate(test_vis, test_dep, test_labels, N_test, num_label, model, args):
    # validate
    pbar = ProgressBar(N_test)
    cnn_correct_cnt = 0
    sum_frame = 0
    
    pred_list = []
    total_loss = np.array(0.0, dtype=np.float32)

    conf_array = np.zeros((num_label, num_label), dtype=np.int32)

    for i in range(0, N_test):
        # shape(T, width ,height)
        x_vis_batch = xp.asarray(test_vis[i], dtype=np.float32)
        x_dep_batch = xp.asarray(test_dep[i], dtype=np.float32)
        y_batch = xp.asarray(test_labels[i], dtype=np.int32).reshape(-1)

        sum_frame += y_batch.shape[0]

        loss, pred = model.forward(
            x_vis_batch, x_dep_batch, y_batch)

        pred = xp.argmax(pred.data, axis=1)
        if args.gpu >= 0:
            cnn_correct_cnt += xp.asnumpy(xp.sum(pred == y_batch))
        else:
            cnn_correct_cnt += np.sum(pred == y_batch)
        
        if y_batch.size == 1:
            conf_array[y_batch, pred] += 1
        else:
            for j in xrange(y_batch.size):
                conf_array[y_batch[j], pred[j]] += 1

        if args.gpu >= 0:
            import cupy
            total_loss += cupy.asnumpy(loss.data)
        else:
            total_loss += loss.data

        pred_list.append(pred.tolist())
        pbar.update(i + 1
                    if (i + 1) < N_test else N_test)

    """
    正答率を計算
    """
    cnn_accuracy = cnn_correct_cnt / float(sum_frame)

    return total_loss / sum_frame, cnn_accuracy, pred_list, conf_array

if __name__ == '__main__':
     
    parser = argparse.ArgumentParser()
    
    # 本来はモデル選択用だが名前のみ使用
    parser.add_argument('--model', type=str, default='cnn_only')
    
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--device_num', type=int, default=0)
    parser.add_argument('--epoch', type=int, default=100)

    parser.add_argument('--batchsize', type=int, default=10)
    parser.add_argument('--snapshot', type=int, default=5)
    parser.add_argument('--visualize', type=int, default=10)
    
    # 学習済みモデルの再開
    parser.add_argument('--restart_from', type=str)
    
    parser.add_argument('--epoch_offset', type=int, default=0)
    
    # 学習データが格納されているディレクトリ名
    parser.add_argument('--datadir', type=str, default='../../data/syuwa_minimum')
    parser.add_argument('--dataset_split', type=str, default='tse4')
    parser.add_argument('--size', type=int, default=28)
    
    parser.add_argument('--norm', type=int, default=0)

   # 学習パラメータの調整
    parser.add_argument('--opt', type=str, default='MomentumSGD',
                    choices=['MomentumSGD', 'Adam', 'AdaGrad'])
    parser.add_argument('--weight_decay', type=float, default=0.0005)
    parser.add_argument('--alpha', type=float, default=0.001)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--lr_decay_freq', type=int, default=10)
    parser.add_argument('--lr_decay_ratio', type=float, default=0.5)
    parser.add_argument('--seed', type=int, default=1701)

    parser.add_argument('--no_opencv', type=int, default=0)
    
    
    args = parser.parse_args()
    
    # 乱数のシードを指定
    np.random.seed(args.seed)

    global xp
    xp = cuda.cupy if args.gpu >= 0 else np

    # create result dir
    log_fn, result_dir = create_result_dir(args)

    print("Result directory: %s" % result_dir)

    logging.info('Prepareing Dataset...')

    # データセットの取得
    train_vis, train_dep, train_labels, test_vis, test_dep, test_labels = load_data(args.datadir, "%s/label.csv" % args.datadir, args.dataset_split, args.no_opencv)

    if args.no_opencv == 1:
        f = open("%s/train_vis.npy" % args.datadir, "r")
        train_vis = pickle.load(f)
        f.close()

        f = open("%s/test_vis.npy" % args.datadir, "r")
        test_vis = pickle.load(f)
        f.close()
 
    def return_max(labels):
        result = 0
        for i in xrange(len(labels)):
            for j in xrange(len(labels[i])):
                if labels[i][j] > result:
                    result = labels[i][j]

        return result
   
    # 出力素子数はデータを見て決定
    num_labels = max(return_max(train_labels), return_max(test_labels)) + 1
    print("Train data loaded: %d" % len(train_vis))
    print("Test data loaded: %d" % len(test_vis))
    print("num of labels: %d" % num_labels)
    logging.info("Train data loaded: %d" % len(train_vis))
    logging.info("Test data loaded: %d" % len(test_vis))
    logging.info("num of labels: %d" % num_labels)

    # prepare model
    model = syuwa_cnn(num_labels)

    if args.restart_from is not None:
        model = pickle.load(open(args.restart_from, 'rb'))
    if args.gpu >= 0:
        import cupy
        cuda.check_cuda_available()
        cuda.get_device(args.device_num).use()
        model.to_gpu()

    def xparray(data):
        if args.gpu >= 0:
            return cupy.asnumpy(data)
        else:
            return data
    
    opt = get_optimizer(args.opt)
    opt.setup(model)

    train_vis = [np.asarray(x).astype(np.float32) for x in train_vis]
    train_dep = [np.asarray(x).astype(np.float32) for x in train_dep]
    test_vis = [np.asarray(x).astype(np.float32) for x in test_vis]
    test_dep = [np.asarray(x).astype(np.float32) for x in test_dep] 

    if args.norm == 1:
        train_vis = map(norm, train_vis)
        train_dep = map(norm, train_dep)
        test_vis = map(norm, test_vis)
        test_dep = map(norm, test_dep) 
    
    logging.info('start training...')

    N = len(train_vis)
    N_test = len(test_vis)
    
    # 学習ループ
    n_epoch = args.epoch
    # 1度に学習させるのは1ファイル
    num_per_epoch = args.batchsize
    for epoch in range(1, n_epoch + 1):

        # train
        if args.opt == 'MomentumSGD':
            print('learning rate:', opt.lr)
            if epoch % args.lr_decay_freq == 0:
                opt.lr *= args.lr_decay_ratio

        print('learning rate:', opt.lr)
        logging.info('learning rate: %f' % opt.lr)
        perm = np.random.permutation(N)

        train_vis_epoch = [x for x in np.array(train_vis)[perm[0:min(num_per_epoch,N)]]]
        train_dep_epoch = [x for x in np.array(train_dep)[perm[0:min(num_per_epoch,N)]]]
        train_labels_epoch = [np.asarray([x]).astype(np.int32) for x in np.array(train_labels)[perm[0:min(num_per_epoch,N)]]]

        mean_loss, cnn_accuracy, conf_array_train = train(train_vis_epoch, train_dep_epoch, train_labels_epoch, min(num_per_epoch, N), num_labels, model, opt, args)
        msg = 'epoch:{:02d}\ttrain loss={}\ttrain accuracy={}'.format(
            epoch + args.epoch_offset, mean_loss, cnn_accuracy)
        logging.info(msg)
        print('\n%s' % msg)

        perm = np.random.permutation(len(test_vis))
        
        # 1度にテストする最大動画数
        # snapshotを取るタイミングで全データを使ってテスト精度を取る
        num_test_per_epoch = N_test if epoch % args.snapshot == 0 else min(50,N_test)
        test_vis_epoch = [x for x in np.array(test_vis)[perm[0:num_test_per_epoch]]]
        test_dep_epoch = [x for x in np.array(test_dep)[perm[0:num_test_per_epoch]]] 
        test_labels_epoch = [np.asarray([x]).astype(np.int32) for x in np.array(test_labels)[perm[0:num_test_per_epoch]]]  
        
        # validate
        mean_loss, lstm_accuracy, pred, conf_array_test = validate(
            test_vis_epoch, test_dep_epoch, test_labels_epoch, num_test_per_epoch, num_labels, model, args)
        msg = 'epoch:{:02d}\ttest loss={}\ttest accuracy={}'.format(
            epoch + args.epoch_offset, mean_loss, cnn_accuracy)
        logging.info(msg)
        print('\n%s' % msg)
        print('Prediction:\n{0}'.format(xparray(pred)))

        # エポックごとにモデルを打ち出す
        if epoch == 1 or epoch % args.snapshot == 0:
            model_fn = '%s/%s_epoch_%d.chainermodel' % (
                result_dir, args.model, epoch + args.epoch_offset)
            pickle.dump(model, open(model_fn, 'wb'), -1)

        # CNNで各層を可視化
        # 1,2層の可視化
        if epoch % args.visualize == 0:
            draw_filters(xparray(model.conv11.W), '%s/log_conv11_epoch_%d.jpg' % (result_dir, epoch))
            draw_filters(xparray(model.conv21.W), '%s/log_conv21_epoch_%d.jpg' % (result_dir, epoch))
            draw_filters_sq(xparray(model.conv12.W), '%s/log_conv12_epoch_%d.jpg' % (result_dir, epoch), 16)
            draw_filters_sq(xparray(model.conv22.W), '%s/log_conv22_epoch_%d.jpg' % (result_dir, epoch), 16)

            # 適当なファイルの画像を入力する
            video_num = int(np.random.random() * len(test_vis))
            s_image_vis = xp.asarray(test_vis[video_num]).astype(np.float32)
            s_image_dep = xp.asarray(test_dep[video_num]).astype(np.float32)
       
            # プーリング1層まで通した結果を出力
            frame_num = int(s_image_vis.shape[0]/ 2)
            draw_image(xparray(model.extract_pool11(s_image_vis[frame_num,:,:]).data), '%s/sample_vis_pool1_%d.jpg' % (result_dir, epoch))
            draw_image(xparray(model.extract_pool21(s_image_dep[frame_num,:,:]).data), '%s/sample_dep_pool1_%d.jpg' % (result_dir, epoch))

        # 学習曲線を出力
        draw_loss_curve(log_fn, '%s/log.jpg' % result_dir)

        print("Confusion Matrix for train data:")
        print_confmat(conf_array_train)

        print("Confusion Matrix for test data:")
        print_confmat(conf_array_test)

        np.savetxt('%s/confmat_train_epoch_%d.csv' % (result_dir, epoch), conf_array_train, delimiter=',', fmt='%d')
        np.savetxt('%s/confmat_test_epoch_%d.csv' % (result_dir, epoch), conf_array_test, delimiter=',', fmt='%d')
        
        # テストセットの予測を出力
        f = open('%s/pred_test_epoch_%d.csv' % (result_dir, epoch), 'w')
        writer = csv.writer(f)
        for i in range(len(pred)):
            writer.writerow(test_labels_epoch[i].tolist())
            writer.writerow(pred[i])
        f.close()
