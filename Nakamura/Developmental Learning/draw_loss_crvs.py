#!/usr/bin/env python
# -*- coding: utf-8 -*-

import re
import sys
import matplotlib
if sys.platform in ['linux', 'linux2']:
    matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import argparse
import os

def drawing(train_set, outdir, fname, label):
    if not os.path.exists(outdir):
        os.makedirs(outdir)
        
    set1 = np.asarray(train_set[0])
    set2 = np.asarray(train_set[1])
    set3 = np.asarray(train_set[2])
    set4 = np.asarray(train_set[3])

    if not len(set1) > 2:
        return

    fig, ax1 = plt.subplots()
    ax1.plot(set1[:, 0], set1[:, 1], label='set1')
    ax1.plot(set2[:, 0], set2[:, 1], label='set2')
    ax1.plot(set3[:, 0], set3[:, 1], label='set3')
    ax1.plot(set4[:, 0], set4[:, 1], label='set4')
    ax1.set_xlim([1, len(set1)])
    ax1.set_xlabel('epoch')
    ax1.set_ylabel(label)

    ax1.legend(bbox_to_anchor=(0.25, -0.1), loc=9)
    plt.savefig(outdir + "/log-" + fname + ".png", bbox_inches='tight')


def draw_loss_curve(logfile, outdir):
    train_loss = []
    train_acc = []
    test_loss = []
    test_acc = []
    
    train_loss_set = []
    train_acc_set = []
    test_loss_set = []
    test_acc_set = []
    num = 0
    for line in open(logfile):
        line = line.strip()
        if not 'epoch:' in line:
            continue
        epoch = int(re.search('epoch:([0-9]+)', line).groups()[0])
            
        if 'train' in line:
            tr_l = float(re.search('loss=(.+),', line).groups()[0])
            tr_a = float(re.search('accuracy=([0-9\.]+)', line).groups()[0])
            train_loss.append([epoch, tr_l])
            train_acc.append([epoch, tr_a])
        if 'test' in line:
            te_l = float(re.search('loss=(.+),', line).groups()[0])
            te_a = float(re.search('accuracy=([0-9\.]+)', line).groups()[0])
            test_loss.append([epoch, te_l])
            test_acc.append([epoch, te_a])
            
        if epoch == 10:
            num += 1
            if num%2 == 0:
                train_loss_set.append(train_loss)
                train_acc_set.append(train_acc)
                test_loss_set.append(test_loss)
                test_acc_set.append(test_acc)
                train_loss = []
                train_acc = []
                test_loss = []
                test_acc = []
    
    print "train_loss_set: ", len(train_loss_set)
    drawing(train_loss_set, outdir, "train_loss", "loss")
    drawing(train_acc_set, outdir, "train_acc", "accuracy")
    drawing(test_loss_set, outdir, "test_loss", "loss")
    drawing(test_acc_set, outdir, "test_acc", "accuracy")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--logfile', '-f', type=str)
    parser.add_argument('--outdir', '-o', type=str)
    args = parser.parse_args()
    print(args)

    draw_loss_curve(args.logfile, args.outdir)