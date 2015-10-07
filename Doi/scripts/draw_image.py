#!/usr/bin/env python
#-*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt

"""
CNNの各層のフィルターを可視化
sqrt(フィルタ数)+1のスクエアに可視化
入力：ndarray(c_O, c_I, k_H, k_W)
"""
def draw_filters(W, file_name):

    size = int(np.sqrt(W.shape[0] * W.shape[1])) + 1
    W = W.reshape(W.shape[0] * W.shape[1], W.shape[2], W.shape[3])

    width = W.shape[1]

    plt.figure(figsize=(15, 18))
    for i in xrange(W.shape[0]):
        plt.subplot(size, size, i+1)
        Z = W[i,:,:]
        plt.xlim(0, width)
        plt.ylim(0, width)
        #plt.axes().set_aspect('auto')
        plt.pcolor(Z)
        plt.gray()
        plt.tick_params(labelbottom="off")
        plt.tick_params(labelleft="off")

    plt.savefig(file_name)
    plt.close()

def draw_filters_sq(W, file_name, show_num_filters):

    size_in = W.shape[1]

    image_size = W.shape[2]

    plt.figure(figsize=(24, 15))
    for i in xrange(show_num_filters * size_in):
        plt.subplot(show_num_filters, size_in, i+1)
        Z = W[int(i / size_in), i % size_in,:,:]
        plt.xlim(0, image_size)
        plt.ylim(0, image_size)
        plt.pcolor(Z)
        plt.gray()
        plt.tick_params(labelbottom="off")
        plt.tick_params(labelleft="off")

    plt.savefig(file_name)
    plt.close()

"""
入力:ndarray(batchsize, num_channels, width, height)
"""
def draw_image(h, file_name):
    
    num_images = h.shape[0] * h.shape[1]
    width = h.shape[2]
    height = h.shape[3]
    graph_size = int(np.sqrt(h.shape[0] * h.shape[1])) + 1
    h = h.reshape((num_images, h.shape[2], h.shape[3]))

    plt.figure(figsize=(18, 18))
    for i in xrange(num_images):
        plt.subplot(graph_size, graph_size, i+1)
        Z = h[i,::-1,:]
        plt.xlim(0, width)
        plt.ylim(0, height)
        plt.pcolor(Z)
        plt.gray()
        plt.tick_params(labelbottom="off")
        plt.tick_params(labelleft="off")

    plt.savefig(file_name)
    plt.close()

"""
テスト
"""
def main():
    data = np.random.random((16, 1, 32, 32))
    draw_filters(data, 'test_draw_filters.png')
    # plt.show()

    data = np.random.random((32, 16, 5, 5))
    draw_filters_sq(data, 'test_draw_filters_sq.png', 16)
    plt.show()

if __name__ == "__main__":
    main()
