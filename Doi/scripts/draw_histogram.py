#!/usr/bin/env python
#-*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt

"""
重みのヒストグラムを出力
入力：ndarray(in, out)
"""
def draw_histogram(W, out_file_name, bins_arg):

    data = W.flatten()

    plt.figure(figsize=(20, 10))
    plt.xlabel('Weight')
    plt.ylabel('Number')
    plt.hist(data,bins=bins_arg)
    plt.savefig(out_file_name)
    plt.close()

def main():

    W = np.random.normal(0, 1, (200, 300))
    
    draw_histogram(W, "test_draw_histogram.png", 20)

if __name__ == "__main__":
    main()
