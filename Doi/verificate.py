#! /usr/bin/env python
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#
# Copyright © 2015 Takuma Yagi <takuma.8192.bogin@mbr.nifty.com>
#

from __future__ import print_function
import sys
sys.path.append('../../')
import argparse
import logging
import time
import os
import pickle
from progressbar import ProgressBar
import csv
import json

import numpy as np
from chainer import cuda, Variable

from train_new import extract_data, load_data
from predict_sign import predict_sign_simple

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
    result_dir = 'run_results/test'
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

        c = Variable(xp.zeros((1, model.h2h.W.shape[1])).astype(np.float32), volatile=True)
        h = Variable(xp.zeros((1, model.h2h.W.shape[1])).astype(np.float32), volatile=True)
        
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

"""
プーリング層の可視化
"""
def visualize_pool1(vis_video, dep_video, video_index, result_dir):

    for t in xrange(vis_video.shape[0]):

        # プーリング1層まで通した結果を出力
        draw_image(xparray(model.extract_pool11(vis_video[t,:,:]).data), '%s/sample_vis_pool1_%d_%d.jpg' % (result_dir, video_index, t))
        draw_image(xparray(model.extract_pool21(dep_video[t,:,:]).data), '%s/sample_dep_pool1_%d_%d.jpg' % (result_dir, video_index, t))

if __name__ == '__main__':
     
    parser = argparse.ArgumentParser()

    # ハードウェアによる実行条件
    parser.add_argument('--gpu', type=int, default=-1)
    parser.add_argument('--device_num', type=int, default=0)
    parser.add_argument('--datadir', type=str, default='../data')
 
    # 学習済みモデルを指定
    parser.add_argument('--model_path', type=str)
     
    # テストデータに関するパラメータ
    parser.add_argument('--dataset_split', type=str, default='tse4')
   
    # 学習器のパラメータ
    parser.add_argument('--norm', type=int, default=0)
    parser.add_argument('--seed', type=int, default=1701)
    
    # LSTM用のパラメータ
    parser.add_argument('--no_opencv', type=int, default=0) 
   
    # 追加出力用
    parser.add_argument('--pool1_output_index', type=int, default=-1)

    args = parser.parse_args()
    
    # 乱数のシードを指定
    np.random.seed(args.seed)

    global xp
    xp = cuda.cupy if args.gpu >= 0 else np

    # create result dir
    log_fn, result_dir = create_result_dir(args)

    print("Result directory: %s" % result_dir)

    logging.info('Prepareing Dataset...')

    # 評価データセットの取得
    test_vis, test_dep, test_labels = load_data(args.datadir, "%s/label.csv" % args.datadir, args.dataset_split, args.no_opencv, False)

    # 非opencv環境ではnpyファイルから可視画像を読み込み
    if args.no_opencv == 1:
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
    num_labels = return_max(test_labels) + 1
    print("Test data loaded: %d" % len(test_vis))
    print("num of labels: %d" % num_labels)
    logging.info("Test data loaded: %d" % len(test_vis))
    logging.info("num of labels: %d" % num_labels)

    # 読み込みたいモデルをロード
    model = pickle.load(open(args.model_path, 'rb'))
    
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
    

    # 動画毎に長さが異なるので、ndarray(video_length, width, height)のリスト形式としてデータを保持
    test_vis = [np.asarray(x).astype(np.float32) for x in test_vis]
    test_dep = [np.asarray(x).astype(np.float32) for x in test_dep] 
    test_labels = [np.asarray([x]).astype(np.int32) for x in np.array(test_labels)]

    if args.norm == 1:
        test_vis = map(norm, test_vis)
        test_dep = map(norm, test_dep) 
     
    N_test = len(test_vis)

    # validate
    mean_loss, lstm_accuracy, pred, conf_array_test = validate(
        test_vis, test_dep, test_labels, len(test_vis), num_labels, model, args)

    pred = xparray(pred)

    msg = 'test loss={}\ttest accuracy={}'.format(
            mean_loss, lstm_accuracy)
    logging.info(msg)
    print('\n%s' % msg)
    print('Prediction:\n{0}'.format(pred))
        
    # テストセットの予測を出力
    f = open('%s/pred_test.csv' % result_dir, 'w')
    writer = csv.writer(f)
    for i in range(len(pred)):
        writer.writerow(pred[i])
    f.close()

    # テストセットの予測を出力
    f = open('%s/compare_answer.csv' % result_dir, 'w')
    writer = csv.writer(f)
    for i in range(len(pred)):
        writer.writerow(test_labels[i][0])
        writer.writerow(pred[i])
    f.close()
    
    """
    単語認識結果を出力
    """
    print("****** one word prediction *******")
    f = open("%s/one_word_prediction.txt" % result_dir, 'w')
    result = []
    answer = []
    for i in xrange(len(test_vis)):
        result.append(predict_sign_simple(pred[i], 5, [0, 1]))
        answer.append(predict_sign_simple(test_labels[i][0], 5, [0, 1]))
    print(result)
    f.write(json.dumps(result))
    f.close()
    overall_accuracy = np.sum([x == y for x, y in zip(result, answer)]).astype(np.float32) / len(result)    
    logging.info("Overall Word Prediction Accuracy:")
    logging.info(overall_accuracy)

    print_confmat(conf_array_test)

    """
    CNNで各層を可視化
    1,2層の可視化
    """
    draw_filters(xparray(model.conv11.W), '%s/log_conv11.jpg' % result_dir)
    draw_filters(xparray(model.conv21.W), '%s/log_conv21.jpg' % result_dir)
    draw_filters_sq(xparray(model.conv12.W), '%s/log_conv12.jpg' % result_dir, 16)
    draw_filters_sq(xparray(model.conv22.W), '%s/log_conv22.jpg' % result_dir, 16)

    """
    テストデータのビデオの1部について、各フレームをpool1に通したあとの出力を行う
    """
    if args.pool1_output_index != -1:
        visualize_pool1(xp.asarray(test_vis[args.pool1_output_index]).astype(np.float32), xp.asarray(test_dep[i]).astype(np.float32), args.pool1_output_index, result_dir)

    # 混同行列をcsvに出力
    np.savetxt('%s/confmat_test.csv' % result_dir, conf_array_test, delimiter=',', fmt='%d')
