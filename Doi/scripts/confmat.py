#!/usr/bin/env python
#-*-coding:utf-8-*-
"""
original code is in stackoverflow
"How to plot confusion matrix with string axis rather than integer in python."
http://stackoverflow.com/questions/5821125/how-to-plot-confusion-matrix-with-string-axis-rather-than-integer-in-python
"""
import numpy as np
import matplotlib
# Force matplotlib to not use any Xwindows backend.
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# print confusion mat in cui
def print_confmat(confmat, mode="vote"):
    print "confmat"
    assert confmat.shape[0]==confmat.shape[1]
    print "================"
    print "l\\a |",
    for i in xrange(confmat.shape[0]):
        if mode == "vote":
            print "%3d "%i,
        if mode == "rate":
            print "%5d "%i,
    print "| total",
    print "\n------------------------------------------------------------"
    for row in xrange(confmat.shape[0]):
        print "%3d |"%row,
        for col in xrange(confmat.shape[1]):
            if mode == "vote":
                print "%3d "%confmat[row,col],
            if mode == "rate":
                print "%0.3f "%(confmat[row,col]/float(sum(confmat[row,:]))),
        print "| %3d "%sum(confmat[row,:])
    print ""

# save confusion matrix as .png image
def save_confmat_fig(conf_arr, savename, labels, 
                     xlabel=None, ylabel=None, saveFormat="png",
                     title=None, clim=(None,None), mode="vote"):
    if mode=="rate":
        conf_rate = []
        for i in conf_arr:
            tmp_arr = []
            total = float(sum(i))
            for j in i:
                if total == 0:
                    tmp_arr.append(float(j))
                else:
                    tmp_arr.append(float(j)/total)
            conf_rate.append(tmp_arr)
        conf_arr = conf_rate
    norm_conf = []
    for i in conf_arr:
        a = 0
        tmp_arr = []
        a = sum(i, 0)
        for j in i:
            if a == 0:
                tmp_arr.append(float(j))
            else:
                tmp_arr.append(float(j)/float(a))
        norm_conf.append(tmp_arr)
    fig = plt.figure()
    plt.clf()
    plt.subplots_adjust(top=0.85) # use a lower number to make more vertical space
    ax = fig.add_subplot(111)
    ax.set_aspect(1)
    if mode == "rate":
        res = plt.imshow(np.array(norm_conf)*100, cmap=plt.cm.Greys_r, 
                        interpolation='nearest')
        plt.clim(0,100)
        threshold = 0.5
    else:
        res = plt.imshow(np.array(norm_conf), cmap=plt.cm.Greys_r, 
                        interpolation='nearest')
        if clim!=(None,None):
            plt.clim(*clim)
        threshold = np.mean([np.max(norm_conf),np.min(norm_conf)])
    width = len(conf_arr)
    height = len(conf_arr[0])

    for x in xrange(width):
        for y in xrange(height):
            if norm_conf[x][y]>=threshold:
                textcolor = '0.0'
            else:
                textcolor = '1.0'
            if mode == "rate":
                ax.annotate("{0:d}".format(int(conf_arr[x][y]*100)), xy=(y, x), 
                            horizontalalignment='center',
                            verticalalignment='center',color=textcolor)
            else:
                ax.annotate("{0}".format(conf_arr[x][y]), xy=(y, x), 
                            horizontalalignment='center',
                            verticalalignment='center',color=textcolor)

    cb = fig.colorbar(res)
    if title != None:
        plt.text(0.5, 1.08, title,
                 horizontalalignment='center',
                 fontsize=15,
                 transform = ax.transAxes)
    ax.xaxis.tick_top()
    plt.xticks(range(width), labels[:width])
    plt.yticks(range(height), labels[:height])
    if xlabel != None:
        plt.xlabel(xlabel)
    if ylabel != None:
        plt.ylabel(ylabel)
        ax.xaxis.set_label_position("top")
    plt.savefig(savename, format=saveFormat)
    plt.close(fig) # closeしなくてはmemoryが解放されない

if __name__=="__main__":
    conf_arr = np.asarray([[33,2,0,0,0,0,0,0,0,1,3], 
                           [3,31,0,0,0,0,0,0,0,0,0], 
                           [0,4,41,0,0,0,0,0,0,0,1], 
                           [0,1,0,30,0,6,0,0,0,0,1], 
                           [0,0,0,0,38,10,0,0,0,0,0], 
                           [0,0,0,3,1,39,0,0,0,0,4], 
                           [0,2,2,0,4,1,31,0,0,0,2],
                           [0,1,0,0,0,0,0,36,0,2,0], 
                           [0,0,0,0,0,0,1,5,37,5,1], 
                           [3,0,0,0,0,0,0,0,0,39,0], 
                           [0,0,0,0,0,0,0,0,0,0,38]])
    labels = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ'
    savename = "confusion_matrix.png"
    save_confmat_fig(conf_arr, savename, labels)
    print_confmat(conf_arr)

