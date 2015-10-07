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


def draw_loss_curve(logfile, outfile, title='Title', offset=0):
    train_loss = []
    train_acc = []
    test_loss = []
    test_acc = []
    for line in open(logfile):
        line = line.strip()
        if not 'epoch:' in line:
            continue
        epoch = int(re.search('epoch:([0-9]+)', line).groups()[0])
        if 'train' in line:
            tr_a = float(re.search('accuracy=([0-9\.]+)', line).groups()[0])
            train_acc.append([epoch, tr_a])
            tr_l = float(re.search('loss=([0-9\.]+)', line).groups()[0])
            train_loss.append([epoch, tr_l])

        if 'test' in line:
            te_a = float(re.search('accuracy=([0-9\.]+)', line).groups()[0])
            test_acc.append([epoch, te_a])
            te_l = float(re.search('loss=([0-9\.]+)', line).groups()[0])
            test_loss.append([epoch, te_l])

    train_acc = np.asarray(train_acc)
    test_acc = np.asarray(test_acc)
    train_loss = np.asarray(train_loss)
    test_loss = np.asarray(test_loss)

    if not len(train_loss) > 2:
        return

        
    fig, ax1 = plt.subplots()
    ax1.set_xlim([1+offset, len(train_loss)+offset])
    ax1.set_xlabel('epoch')
    ax1.set_ylabel('loss')
    ax1.plot(train_loss[:, 0], train_loss[:, 1],
                label='training loss', c='b')
    ax1.plot(test_loss[:, 0], test_loss[:, 1],
                label='test loss', c='g')

    ax2 = ax1.twinx()
    ax2.plot(train_acc[:, 0], train_acc[:, 1],
                 label='training accuracy', c='r')
    ax2.plot(test_acc[:, 0], test_acc[:, 1], label='test accuracy', c='c')
    ax2.set_xlim([1, len(train_loss)])
    ax2.set_ylabel('accuracy')

    ax1.legend(bbox_to_anchor=(0.25, -0.1), loc=9)
    ax2.legend(bbox_to_anchor=(0.75, -0.1), loc=9)
    plt.title(title)
    plt.savefig(outfile, bbox_inches='tight')
    plt.close()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--logfile', '-f', type=str)
    parser.add_argument('--outfile', '-o', type=str)
    args = parser.parse_args()
    print(args)

    draw_loss_curve(args.logfile, args.outfile)
