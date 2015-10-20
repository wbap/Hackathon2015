# -*- coding: utf-8 -*-

import os
import numpy as np
import glob
import pickle
import argparse
import re
import cv2


def image_data_make(image_file):
    image=cv2.imread(image_file, 1)
    image_array = np.array(image)
    image_red=image_array[:,:,0]
    image_green=image_array[:,:,1]
    image_blue=image_array[:,:,2]
    return image_array

def load_dataset(datadir='data_animal'):
    train_data = np.load('%s/train_data.npy' % datadir)
    train_labels = np.load('%s/train_labels.npy' % datadir)
    test_data = np.load('%s/test_data.npy' % datadir)
    test_labels = np.load('%s/test_labels.npy' % datadir)

    return train_data, train_labels, test_data, test_labels


if __name__ == '__main__':
    image_array=[]
    image_red=[]
    image_green=[]
    image_blue=[]
    images=[]
    cat_datas=[]
    cat_data=[]
    targets=[]
    cat_data_cov=[]
    image_list = glob.glob('C:/Users/urara3823/Desktop/images/*.jpg*')
    for file in image_list:
        images.append(file)

    for i in range(len(images)):
        if 'Abyssinian' in images[i]:
            cat_datas.append(image_data_make(images[i]))
            cat_data.append(image_data_make(images[i]).flatten())
            targets.append(1)
        if 'Bengal' in images[i]:
            cat_datas.append(image_data_make(images[i]))
            cat_data.append(image_data_make(images[i]).flatten())
            targets.append(2)
        if 'Birman' in images[i]:
            cat_datas.append(image_data_make(images[i]))
            cat_data.append(image_data_make(images[i]).flatten())
            targets.append(3)
        if 'Bombay' in images[i]:
            cat_datas.append(image_data_make(images[i]))
            cat_data.append(image_data_make(images[i]).flatten())
            targets.append(4)
        if 'British_Shorthair' in images[i]:
            cat_datas.append(image_data_make(images[i]))
            cat_data.append(image_data_make(images[i]).flatten())
            targets.append(5)
        if 'Egyptian_Mau' in images[i]:
            cat_datas.append(image_data_make(images[i]))
            cat_data.append(image_data_make(images[i]).flatten())
            targets.append(6)
        if 'Maine_Coon' in images[i]:
            cat_datas.append(image_data_make(images[i]))
            cat_data.append(image_data_make(images[i]).flatten())
            targets.append(7)
        if 'Persian' in images[i]:
            cat_datas.append(image_data_make(images[i]))
            cat_data.append(image_data_make(images[i]).flatten())
            targets.append(8)
        if 'Ragdoll' in images[i]:
            cat_datas.append(image_data_make(images[i]))
            cat_data.append(image_data_make(images[i]).flatten())
            targets.append(9)
        if 'Russian_Blue' in images[i]:
            cat_datas.append(image_data_make(images[i]))
            cat_data.append(image_data_make(images[i]).flatten())
            targets.append(10)
        if 'Siamese' in images[i]:
            cat_datas.append(image_data_make(images[i]))
            cat_data.append(image_data_make(images[i]).flatten())
            targets.append(11)
        if 'Sphynx' in images[i]:
            cat_datas.append(image_data_make(images[i]))
            cat_data.append(image_data_make(images[i]).flatten())
            targets.append(12)
        else:
            pass

    parser = argparse.ArgumentParser()
    parser.add_argument('--outdir', '-o', type=str, default='data_cat')
    args = parser.parse_args()
    print(args)

    if not os.path.exists(args.outdir):
        os.mkdir(args.outdir)

    data = np.array(cat_data)
    labels = np.array(targets)

    num = 1800
    train_data = data[0:1800].reshape((num, 3, 32, 32)).astype(np.float32)
    train_labels = labels[0:1800].astype(np.int32)

    np.save('%s/train_data' % args.outdir, train_data)
    np.save('%s/train_labels' % args.outdir, train_labels)

    num_test=590
    test_data = data[1800:2390].reshape((num_test, 3, 32, 32)).astype(np.float32)
    test_labels = labels[1800:2390].astype(np.int32)

    np.save('%s/test_data' % args.outdir, test_data)
    np.save('%s/test_labels' % args.outdir, test_labels)
