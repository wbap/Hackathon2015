# -*- coding: utf-8 -*-

import os
import numpy as np
import glob
import pickle
import argparse
import re
import cv2
import numpy as np
import random as rnd


def image_data_make(image_file):
    image=cv2.imread(image_file, 1)
    image_array = np.array(image)
    image_red=image_array[:,:,0]
    image_green=image_array[:,:,1]
    image_blue=image_array[:,:,2]
    return image_array

def load_dataset(datadir='data_cat_or_dog'):
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
            targets.append(1)
        if 'Birman' in images[i]:
            cat_datas.append(image_data_make(images[i]))
            cat_data.append(image_data_make(images[i]).flatten())
            targets.append(1)
        if 'Bombay' in images[i]:
            cat_datas.append(image_data_make(images[i]))
            cat_data.append(image_data_make(images[i]).flatten())
            targets.append(1)
        if 'British_Shorthair' in images[i]:
            cat_datas.append(image_data_make(images[i]))
            cat_data.append(image_data_make(images[i]).flatten())
            targets.append(1)
        if 'Egyptian_Mau' in images[i]:
            cat_datas.append(image_data_make(images[i]))
            cat_data.append(image_data_make(images[i]).flatten())
            targets.append(1)
        if 'Maine_Coon' in images[i]:
            cat_datas.append(image_data_make(images[i]))
            cat_data.append(image_data_make(images[i]).flatten())
            targets.append(1)
        if 'Persian' in images[i]:
            cat_datas.append(image_data_make(images[i]))
            cat_data.append(image_data_make(images[i]).flatten())
            targets.append(1)
        if 'Ragdoll' in images[i]:
            cat_datas.append(image_data_make(images[i]))
            cat_data.append(image_data_make(images[i]).flatten())
            targets.append(1)
        if 'Russian_Blue' in images[i]:
            cat_datas.append(image_data_make(images[i]))
            cat_data.append(image_data_make(images[i]).flatten())
            targets.append(1)
        if 'Siamese' in images[i]:
            cat_datas.append(image_data_make(images[i]))
            cat_data.append(image_data_make(images[i]).flatten())
            targets.append(1)
        if 'Sphynx' in images[i]:
            cat_datas.append(image_data_make(images[i]))
            cat_data.append(image_data_make(images[i]).flatten())
            targets.append(1)
        if 'american_bulldog' in images[i]:
            cat_datas.append(image_data_make(images[i]))
            cat_data.append(image_data_make(images[i]).flatten())
            targets.append(2)
        if 'american_pit_bull_terrier' in images[i]:
            cat_datas.append(image_data_make(images[i]))
            cat_data.append(image_data_make(images[i]).flatten())
            targets.append(2)
        if 'basset_hound' in images[i]:
            cat_datas.append(image_data_make(images[i]))
            cat_data.append(image_data_make(images[i]).flatten())
            targets.append(2)
        if 'beagle' in images[i]:
            cat_datas.append(image_data_make(images[i]))
            cat_data.append(image_data_make(images[i]).flatten())
            targets.append(2)
        if 'boxer' in images[i]:
            cat_datas.append(image_data_make(images[i]))
            cat_data.append(image_data_make(images[i]).flatten())
            targets.append(2)
        if 'chihuahua' in images[i]:
            cat_datas.append(image_data_make(images[i]))
            cat_data.append(image_data_make(images[i]).flatten())
            targets.append(2)
        if 'english_cocker_spaniel' in images[i]:
            cat_datas.append(image_data_make(images[i]))
            cat_data.append(image_data_make(images[i]).flatten())
            targets.append(2)
        if 'english_setter' in images[i]:
            cat_datas.append(image_data_make(images[i]))
            cat_data.append(image_data_make(images[i]).flatten())
            targets.append(2)
        if 'german_shorthaired' in images[i]:
            cat_datas.append(image_data_make(images[i]))
            cat_data.append(image_data_make(images[i]).flatten())
            targets.append(2)
        if 'great_pyrenees' in images[i]:
            cat_datas.append(image_data_make(images[i]))
            cat_data.append(image_data_make(images[i]).flatten())
            targets.append(2)
        if 'havanese' in images[i]:
            cat_datas.append(image_data_make(images[i]))
            cat_data.append(image_data_make(images[i]).flatten())
            targets.append(2)
        if 'japanese_chin' in images[i]:
            cat_datas.append(image_data_make(images[i]))
            cat_data.append(image_data_make(images[i]).flatten())
            targets.append(2)
        if 'keeshond' in images[i]:
            cat_datas.append(image_data_make(images[i]))
            cat_data.append(image_data_make(images[i]).flatten())
            targets.append(2)
        if 'leonberger' in images[i]:
            cat_datas.append(image_data_make(images[i]))
            cat_data.append(image_data_make(images[i]).flatten())
            targets.append(2)
        if 'miniature_pinscher' in images[i]:
            cat_datas.append(image_data_make(images[i]))
            cat_data.append(image_data_make(images[i]).flatten())
            targets.append(2)
        if 'newfoundland' in images[i]:
            cat_datas.append(image_data_make(images[i]))
            cat_data.append(image_data_make(images[i]).flatten())
            targets.append(2)
        if 'pomeranian' in images[i]:
            cat_datas.append(image_data_make(images[i]))
            cat_data.append(image_data_make(images[i]).flatten())
            targets.append(2)
        if 'pug' in images[i]:
            cat_datas.append(image_data_make(images[i]))
            cat_data.append(image_data_make(images[i]).flatten())
            targets.append(2)
        if 'saint_bernard' in images[i]:
            cat_datas.append(image_data_make(images[i]))
            cat_data.append(image_data_make(images[i]).flatten())
            targets.append(2)
        if 'samoyed' in images[i]:
            cat_datas.append(image_data_make(images[i]))
            cat_data.append(image_data_make(images[i]).flatten())
            targets.append(2)
        if 'scottish_terrier' in images[i]:
            cat_datas.append(image_data_make(images[i]))
            cat_data.append(image_data_make(images[i]).flatten())
            targets.append(2)
        if 'shiba_inu' in images[i]:
            cat_datas.append(image_data_make(images[i]))
            cat_data.append(image_data_make(images[i]).flatten())
            targets.append(2)
        if 'staffordshire_bull_terrier' in images[i]:
            cat_datas.append(image_data_make(images[i]))
            cat_data.append(image_data_make(images[i]).flatten())
            targets.append(2)
        if 'wheaten_terrier' in images[i]:
            cat_datas.append(image_data_make(images[i]))
            cat_data.append(image_data_make(images[i]).flatten())
            targets.append(2)
        if 'yorkshire_terrier' in images[i]:
            cat_datas.append(image_data_make(images[i]))
            cat_data.append(image_data_make(images[i]).flatten())
            targets.append(2)
        else:
            pass

    parser = argparse.ArgumentParser()
    parser.add_argument('--outdir', '-o', type=str, default='data_cat_or_dog')
    args = parser.parse_args()
    print(args)

    if not os.path.exists(args.outdir):
        os.mkdir(args.outdir)

    data = rnd.shuffle(np.array(cat_data))
    labels = rnd.shuffle(np.array(targets))

    num = 4920
    train_data = data[0:4920].reshape((num, 3, 32, 32)).astype(np.float32)
    train_labels = labels[0:4920].astype(np.int32)

    np.save('%s/train_data' % args.outdir, train_data)
    np.save('%s/train_labels' % args.outdir, train_labels)

    test_num = 2460
    test_data = data[4920:7380].reshape((test_num, 3, 32, 32)).astype(np.float32)
    test_labels = labels[4920:7380].astype(np.int32)

    np.save('%s/test_data' % args.outdir, test_data)
    np.save('%s/test_labels' % args.outdir, test_labels)
