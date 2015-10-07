#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import print_function
import sys
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

from models.cnn_lstm_new import syuwa_cnn_lstm
from models.conv_lstm import syuwa_conv_lstm
from models.cnn_lstm_drop import syuwa_cnn_lstm_drop
from models.cnn_lstm_fc1 import cnn_lstm_fc1
from scripts.draw_image import draw_filters, draw_filters_sq, draw_image
from scripts.draw_loss import draw_loss_curve
from scripts.draw_histogram import draw_histogram
from scripts.confmat import save_confmat_fig, print_confmat

xp = np

"""
引数の情報をログとして出力
Sample: results/cnn_lstm_15-09-07-17-25-30_1441614333116614/log.txt
"""
def create_result_dir(args):
    result_dir = 'results/%s' % args.model
    result_dir += '_' + time.strftime('%Y-%m-%d_%H-%M-%S_')
    result_dir += str(time.time()).replace('.', '')
    if args.gpu >= 0:
        result_dir +=  '_%d' % args.device_num
    if not os.path.exists(result_dir):
        os.makedirs(result_dir)
    log_fn = '%s/log.txt' % result_dir
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

"""
ディレクトリ内のファイルを読み込み、訓練集合とテスト集合に分割して返す
"""
def load_data(dirname, label_file_name, split, no_opencv, train=True):

    if split == 'tse4':
        vpattern = '/[sw]??_p??_e04_gray_f_132.avi'
        dpattern = '/[sw]??_p??_e04_depth_f_132.xml'
    elif split == 'tsp4':
        vpattern = '/[sw]??_p04_e??_gray_f_132.avi'
        dpattern = '/[sw]??_p04_e??_depth_f_132.xml'
    elif split == 'tsp5':
        vpattern = '/[sw]??_p05_e??_gray_f_132.avi'
        dpattern = '/[sw]??_p05_e??_depth_f_132.xml'
    elif split == 'tsp2':
        vpattern = '/[sw]??_p02_e??_gray_f_132.avi'
        dpattern = '/[sw]??_p02_e??_depth_f_132.xml'
    else:
        raise Exception('No split method is selected')

    files = glob.glob(dirname + '/*.avi')
    test_files_vis = sorted(glob.glob(dirname + vpattern))
    train_files_vis = sorted(list(set(files) - set(test_files_vis)))

    files = glob.glob(dirname + '/*.xml')
    test_files_dep = sorted(glob.glob(dirname + dpattern))
    train_files_dep = sorted(list(set(files) - set(test_files_dep)))

    if train == True:
        train_vis, train_dep, train_label = extract_data(train_files_vis, train_files_dep, label_file_name, no_opencv)
    test_vis, test_dep, test_label = extract_data(test_files_vis, test_files_dep, label_file_name, no_opencv)
   
    if train == True:
        return train_vis, train_dep, train_label, test_vis, test_dep, test_label
    else:
        return test_vis, test_dep, test_label

"""
データをnumpy配列に格納
入力：可視動画・深さ動画ファイル名一覧、教師ラベルファイル名
教師データのあるデータのみ読み込み、他はスキップする
"""
def extract_data(files_vis, files_dep, label_file_name, no_opencv):

    data_vis = []
    data_label = []
    data_dep = []

    # ラベル一覧をロード
    lf = open(label_file_name, 'rb')

    for f, fd in zip(files_vis, files_dep):
        
        def extract_file_id(file_name):
            w = int(file_name[1:3])
            if file_name[0] == "s":
                w += 100

            p = int(file_name[5:7])
            e = int(file_name[9:11])

            return w, p, e

        def expand_labels(line, line_num, video_length):
            
            # 指定がなかった場合の正解ラベルは1(finish)とする
            labels = [1 for x in xrange(video_length)]
            if len(line) % 3 != 0:
                raise Exception("Label format invalid in line %d" % line_num)

            for i in range(1, len(line) / 3):
                label = line[i*3]
                start = int(line[i*3+1])
                end = int(line[i*3+2])

                # 0はstart(手話開始)、1はfinish(手話終了)を示す
                if label == 's':
                    label = 0
                elif label == 'f':
                    label = 1
                else:
                    label = int(label) + 1

                for j in xrange(start, min(end+1, video_length+1)):
                    labels[j-1] = label

            return labels

        # ビデオ長だけを先行取得
        root = ET.parse(fd).getroot()
        video_length = len(root)

        # 対応ラベルの読み出し
        # 該当ラベルが見つからない場合は読み出しをスキップ
        labels = []
        w, p, e = extract_file_id(os.path.basename(f))
        
        found = False
        lf.seek(0)
        reader = csv.reader(lf)
        for i, line in enumerate(reader):
            if int(line[0]) == w and int(line[1]) == p and int(line[2]) == e:
                found = True
                labels = expand_labels(line, i, video_length)

        # ラベルがないと学習出来ないためスキップ
        if found == False:
            print("Label not found for w%dp%de%d: skip" % (w, p, e))
            continue
        
        data_label.append(labels)
        
        # 可視画像の読み出し
        if no_opencv == 0:
        
            video = []
            cap = cv2.VideoCapture(f)

            while(1):
 
                ret, frame = cap.read()
                if frame is None:
                    break

                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                gray = gray.astype(np.float32) / 255

                video.append(gray)

            cap.release()
        
            # 読み出しが確定した時点で追加
            data_vis.append(np.array(video))
        
        # Depth画像のロード
        root = ET.parse(fd).getroot()
        width = int(root[0][1].text)
        height = int(root[0][0].text)
        video = []

        for i in xrange(len(root)):
            frame = np.fromstring(root[i][3].text, sep=' ')
            frame = frame.reshape((width, height)).astype(np.float32) / 4095

            video.append(frame)

        data_dep.append(np.array(video))

        print('Loaded {0}'.format(os.path.basename(f)[0:11]))

    return data_vis, data_dep, data_label

"""
学習本体
train_data: 訓練入力
train_labels: 教師信号(正解ラベル)
N: 訓練データの数
"""
def train(train_vis, train_dep, train_labels, N, num_label, model, opt, args):

    # 訓練
    pbar = ProgressBar(N)
    
    # 学習するサンプルのバッチをランダムに取る
    # 当面はbatchsize=1
    lstm_correct_cnt = 0
    sum_frame = 0
    total_loss = np.array(0, dtype=np.float32)

    # 最適化器の準備
    opt.setup(model)

    # 混同行列
    conf_array = np.zeros((num_label, num_label), dtype=np.int32)
 
    """
    ここから訓練本体
    動画ごとのフレーム数が異なるため、バッチ数は1で固定
    """
    for i in range(0, N):
        x_vis_batch = xp.asarray(train_vis[i], dtype=np.float32)
        x_dep_batch = xp.asarray(train_dep[i], dtype=np.float32)
        y_batch = xp.asarray(train_labels[i], dtype=np.int32)

        # 学習の初期化
        opt.zero_grads()
        accum_loss = 0

        video_length = min(x_vis_batch.shape[0], x_dep_batch.shape[0])
        sum_frame += video_length

        c = Variable(xp.zeros((1, args.num_hunits)).astype(np.float32))
        h = Variable(xp.zeros((1, args.num_hunits)).astype(np.float32))

        for t in xrange(video_length):
        
            x_vis_frame = x_vis_batch[t,:,:]
            x_dep_frame = x_dep_batch[t,:,:]
            y_frame = y_batch[:,t]

            # 順伝播
            loss_i, pred, c, h = model.forward_one_step(
                x_vis_frame, x_dep_frame, y_frame, c, h)

            accum_loss += loss_i
            
            if args.gpu >= 0:
                import cupy
                total_loss += cupy.asnumpy(loss_i.data)
            else:
                total_loss += loss_i.data

            # 正解フレーム数をカウント
            pred = xp.argmax(pred.data)
            if(pred.tolist() == y_frame[0]):
                lstm_correct_cnt += 1
            conf_array[y_frame[0], pred.tolist()] += 1

            # truncated BPTTによる逆伝播
            if (t + 1) % args.bprop_len == 0 or t == video_length -1:  # Run truncated BPTT
                opt.zero_grads()
                accum_loss.backward()
                
                accum_loss.unchain_backward()  # truncate
                accum_loss = Variable(xp.zeros((), dtype=np.float32))

                opt.clip_grads(args.grad_clip)
                
                # パラメタの更新
                opt.update()

        if args.opt in ['AdaGrad', 'MomentumSGD']:
            opt.weight_decay(decay=args.weight_decay)

        pbar.update(i + 1 if (i + 1) < N else N)

    
    """
    正答率を計算
    """
    lstm_accuracy = lstm_correct_cnt / float(sum_frame)

    return total_loss / sum_frame, lstm_accuracy, conf_array

"""
入力の正規化
今回は動画全体に対してLocal Contrast Normalizationを行う
入力：ndarray(T, width, height)
"""
def norm(x):
    if not x.dtype == np.float32:
        x = x.astype(np.float32)
    # 元はチャンネルごとになんかやっている？
    x = (x - np.mean(x)) / (np.std(x) + np.finfo(np.float32).eps)

    return x

"""
テストデータを用いた評価
可視CNN: 総テスト誤差、正答率？
DepthCNN: 総テスト誤差、正答率？
LSTM: 総テスト誤差、正答率(正しくラベリングできたフレーム数/総テストフレーム数)
"""
def validate(test_vis, test_dep, test_labels, N_test, num_label, model, args):
    # validate
    pbar = ProgressBar(N_test)
    lstm_correct_cnt = 0
    sum_frame = 0
    
    pred_list = []
    total_loss = np.array(0.0, dtype=np.float32)

    conf_array = np.zeros((num_label, num_label), dtype=np.int32)

    for i in range(0, N_test):
        # shape(T, width ,height)
        x_vis_batch = xp.asarray(test_vis[i], dtype=np.float32)
        x_dep_batch = xp.asarray(test_dep[i], dtype=np.float32)
        y_batch = xp.asarray(test_labels[i], dtype=np.int32)

        video_length = min(x_vis_batch.shape[0], x_dep_batch.shape[0])
        sum_frame += video_length

        c = Variable(xp.zeros((1, args.num_hunits)).astype(np.float32), volatile=True)
        h = Variable(xp.zeros((1, args.num_hunits)).astype(np.float32), volatile=True)
        
        preds = []

        for t in xrange(video_length):

            x_vis_frame = x_vis_batch[t,:,:]
            x_dep_frame = x_dep_batch[t,:,:]
            y_frame = y_batch[:,t]

            # LSTMの順伝播
            loss_i, pred_lstm, c, h = model.forward_one_step(
            x_vis_frame, x_dep_frame, y_frame, c, h, volatile=True)

            if args.gpu >= 0:
                import cupy
                total_loss += cupy.asnumpy(loss_i.data)
            else:
                total_loss += loss_i.data

            pred = xp.argmax(pred_lstm.data)
            if(pred.tolist() == y_frame[0]):
                lstm_correct_cnt += 1
            conf_array[y_frame[0], pred.tolist()] += 1

            preds.append(pred.tolist())

        pred_list.append(preds)

        pbar.update(i + 1
                    if (i + 1) < N_test else N_test)

    """
    正答率を計算
    """
    lstm_accuracy = lstm_correct_cnt / float(sum_frame)

    return total_loss / sum_frame, lstm_accuracy, pred_list, conf_array 

if __name__ == '__main__':
     
    parser = argparse.ArgumentParser()

    # ハードウェアの実行条件
    parser.add_argument('--gpu', type=int, default=-1)
    parser.add_argument('--device_num', type=int, default=0)
    parser.add_argument('--epoch', type=int, default=100)
    parser.add_argument('--datadir', type=str, default='../data')
    
    # 学習済みモデルの再開
    parser.add_argument('--restart_from', type=str)
    parser.add_argument('--epoch_offset', type=int, default=0)

    # 学習の監視に関するパラメータ
    parser.add_argument('--snapshot', type=int, default=5)
    parser.add_argument('--visualize', type=int, default=10)
   
    # 学習データに関するパラメータ
    parser.add_argument('--dataset_split', type=str, default='tse4')
    
    # 学習器のパラメータ 
    parser.add_argument('--norm', type=int, default=0)
    parser.add_argument('--opt', type=str, default='MomentumSGD',
                    choices=['MomentumSGD', 'Adam', 'AdaGrad'])
    parser.add_argument('--weight_decay', type=float, default=0.0005)
    parser.add_argument('--alpha', type=float, default=0.001)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--lr_decay_freq', type=int, default=10)
    parser.add_argument('--lr_decay_ratio', type=float, default=0.5)
    parser.add_argument('--seed', type=int, default=1701)
    
    # CNN-LSTMのネットワーク構造に関わるパラメタ
    parser.add_argument('--model', type=str, default='syuwa_cnn_lstm')    
    parser.add_argument('--num_iunits', type=int, default=40)
    parser.add_argument('--num_hunits', type=int, default=256)
    parser.add_argument('--bprop_len', type=int, default=60)
    parser.add_argument('--grad_clip', type=int, default=5)

    # 非OpenCV用
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
    # 非opencv環境ではnpyファイルから可視画像を読み込み
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

    assert num_labels != 0
    assert len(train_vis) != 0
    assert len(test_vis) != 0

    print("Train data loaded: %d" % len(train_vis))
    print("Test data loaded: %d" % len(test_vis))
    print("num of labels: %d" % num_labels)
    logging.info("Train data loaded: %d" % len(train_vis))
    logging.info("Test data loaded: %d" % len(test_vis))
    logging.info("num of labels: %d" % num_labels)

    # モデルを用意する
    if args.model == 'syuwa_cnn_lstm':
        model = syuwa_cnn_lstm(args.num_iunits, args.num_hunits, num_labels)
    elif args.model == 'syuwa_conv_lstm':
        model = syuwa_conv_lstm(args.num_iunits, args.num_hunits, num_labels)
    elif args.model == 'syuwa_cnn_lstm_drop':
        model = syuwa_cnn_lstm_drop(args.num_iunits, args.num_hunits, num_labels)
    elif args.model == 'cnn_lstm_fc1':
        model = cnn_lstm_fc1(args.num_iunits, args.num_hunits, num_labels)
    else:
        raise Exception("Invalid model selection")

    if args.restart_from is not None:
        model = pickle.load(open(args.restart_from, 'rb'))
    
    # GPU使用時の前処理
    if args.gpu >= 0:
        import cupy
        cuda.check_cuda_available()
        cuda.get_device(args.device_num).use()
        model.to_gpu(args.device_num)

    # GPUモード時にcuda.cupyをndarrayに変換
    def xparray(data):
        if args.gpu >= 0:
            return cupy.asnumpy(data)
        else:
            return data
    
    opt = get_optimizer(args.opt)
    opt.setup(model)

    # 動画毎に長さが異なるので、ndarray(video_length, width, height)のリスト形式としてデータを保持
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

    # 学習エポック数
    n_epoch = args.epoch + args.epoch_offset 
    # 1度に学習させ最大動画数
    num_per_epoch = min(100,N)
    for epoch in range(args.epoch_offset + 1, n_epoch + 1):

        # train
        if args.opt == 'MomentumSGD':
            if epoch % args.lr_decay_freq == 0:
                opt.lr *= args.lr_decay_ratio

        print('learning rate:', opt.lr)
        logging.info('learning rate: %f' % opt.lr)
        perm = np.random.permutation(N)

        train_vis_epoch = [x for x in np.array(train_vis)[perm[0:num_per_epoch]]]
        train_dep_epoch = [x for x in np.array(train_dep)[perm[0:num_per_epoch]]]
        train_labels_epoch = [np.asarray([x]).astype(np.int32) for x in np.array(train_labels)[perm[0:num_per_epoch]]]

        mean_loss, lstm_accuracy, conf_array_train = train(
            train_vis_epoch, train_dep_epoch, train_labels_epoch, num_per_epoch, num_labels, model, opt, args)
        msg = 'epoch:{:02d}\ttrain loss={}\ttrain accuracy={}'.format(
            epoch, mean_loss, lstm_accuracy)
        logging.info(msg)
        print('\n%s' % msg)
    
        perm = np.random.permutation(N_test)
        
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
            epoch, mean_loss, lstm_accuracy)
        logging.info(msg)
        print('\n%s' % msg)
        print('Prediction:\n{0}'.format(xparray(pred)))

        # モデルをpickleで保存する
        if epoch == 1 or epoch % args.snapshot == 0:
            model_fn = '%s/%s_epoch_%d.chainermodel' % (
                result_dir, args.model, epoch)
            pickle.dump(model, open(model_fn, 'wb'), -1)

        """
        CNNで各層を可視化
        1,2層の可視化
        """
        if epoch % args.visualize == 0:
            draw_filters(xparray(model.conv11.W), '%s/log_conv11_epoch_%d.jpg' % (result_dir, epoch))
            draw_filters(xparray(model.conv21.W), '%s/log_conv21_epoch_%d.jpg' % (result_dir, epoch))
            draw_filters_sq(xparray(model.conv12.W), '%s/log_conv12_epoch_%d.jpg' % (result_dir, epoch), 16)
            draw_filters_sq(xparray(model.conv22.W), '%s/log_conv22_epoch_%d.jpg' % (result_dir, epoch), 16)

        # 学習曲線を出力
        draw_loss_curve(log_fn, '%s/loss_accuracy.jpg' % result_dir, result_dir, args.epoch_offset)

        labels = "sf123456789012345678901234567890123456789012345678901234567890"[0:num_labels]

        # 混同行列を表示
        print("Confusion Matrix for train data:")
        print_confmat(conf_array_train)

        print("Confusion Matrix for test data:")
        print_confmat(conf_array_test)

        # 混同行列をcsvに出力
        np.savetxt('%s/confmat_train_epoch_%d.csv' % (result_dir, epoch), conf_array_train, delimiter=',', fmt='%d')
        np.savetxt('%s/confmat_test_epoch_%d.csv' % (result_dir, epoch), conf_array_test, delimiter=',', fmt='%d')

        # テストセットの予測を出力
        f = open('%s/pred_test_epoch_%d.csv' % (result_dir, epoch), 'w')
        writer = csv.writer(f)
        for i in range(len(pred)):
            writer.writerow(test_labels_epoch[i][0])
            writer.writerow(pred[i])
        f.close()

