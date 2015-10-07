#!/usr/bin/env python
#-*- coding: utf-8 -*-

import numpy as np
from scipy import linalg
import glob
import cv2
import csv
import os
import xml.etree.ElementTree as ET

# 全画像のファイルがあるとする
# array(??, 132, 132)
def zca_whiten(array):

    _mean = np.mean(array, axis=0)
    _array = (array - _mean)

    sigma = np.dot(_array.T, _array)/_array.shape[0]
    print sigma.shape
    
    U, S, V = linalg.svd(sigma)

    tmp = np.dot(U, np.diag( 1 / np.sqrt(S + np.finfo(np.float32).eps)))
    components = np.dot( tmp, U.T )
    np.save("components.npy", components)
    
    whiten = np.dot( _array, components.T )

    return whiten

def read_depth(dirname):

    file_list = glob.glob(dirname + "*.avi")

    num_frame_list = []
    array = []

    for f in file_list:

        root = ET.parse(fd).getroot()

        for i in xrange(len(root)):
            frame = np.fromstring(root[i][3].text, sep=' ')

            if array == []:
                array = frame.astype(np.float32) / 4095
            else:
                array = np.vstack((array, frame.astype(np.float32) / 4095))

        num_frame_list.append(len(root))

    print array.shape

    print "start whitening..."
    white = zca_whiten(array)

    np.save(dirname + "result_depth.npy", white)
    f = open(dirname + "timestamp_depth.csv", 'w')
    writer = csv.writer(f)
    writer.writerow(num_frame_list)
    f.close()

def extract_depth(dirname):

    file_list = glob.glob(dirname + "/*.xml")

    data = np.load(dirname + "result_depth.npy")
    df = open(dirname + "timestamp_depth.csv", 'r')
    reader = csv.reader(df)
    timestamp = reader[0]

    df.close()

    if os.path.isdir(dirname + "result") == False:
        os.path.makedir(dirname + "result")

    head = 0
    for t, f in zip(timestamp, file_list):

        print "Processing %s..." % os.path.basename(f)
        np.save(dirname + os.path.splitext(os.path.basename(f))[0] + ".npy", data[head:head+t,:])


"""
AVIファイルを読み込んで白色化する
"""
def read_avi(dirname):

    file_list = glob.glob(dirname + "*.avi")

    num_frame_list = []
    array = []

    for f in file_list:
        
        num_frame = 0
        cap = cv2.VideoCapture(f)

        while(1):

            ret, frame = cap.read()
            if frame is None:
               break

            num_frame += 1

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            gray = gray.astype(np.float32) / 255

            if array == []:
                array = gray.reshape(132*132)
            else:
                array=np.vstack((array, gray.reshape(132*132)))

        num_frame_list.append(num_frame)

    print array.shape

    print "start whitening..."
    white = zca_whiten(array)

    np.save(dirname + "result.npy", white)
    f = open(dirname + "timestamp.csv", 'w')
    writer = csv.writer(f)
    writer.writerow(num_frame_list)
    f.close()

def extract_avi(dirname):
    
    file_list = glob.glob(dirname + "/*." + avi)

    data = np.load(dirname + "result.npy")
    df = open(dirname + "timestamp.csv", 'r')
    reader = csv.reader(df)
    timestamp = reader[0]

    df.close()

    if os.path.isdir(dirname + "result") == False:
        os.path.makedir(dirname + "result")

    head = 0
    for t, f in timestamp, file_list:

        print "Processing %s..." % os.path.basename(f)
        out = cv2.VideoWriter(dirname + "result/" + os.path.splitext(os.path.basename(f))[0] + "_zca" + filetype, cv2.VideoWriter_fourcc('X', 'V', 'I', 'D'), 30.0, (132, 132), False)

        pixel_range = np.max(np.max(data[head:head+t,:]), np.abs(np.min(data[head:head+t,:])))
        for i in xrange(t):
            frame = data[i+head,:].reshape((132, 132))
            fout = (frame * 127 / pixel_range + 127).astype(np.uint8)
            out.write(fout)

        out.release()
        head += t

def create_avi():

    data = np.load("test.npy")

    out = cv2.VideoWriter("test.avi", cv2.VideoWriter_fourcc('X', 'V', 'I', 'D'), 30.0, (132, 132), False)

    for i in xrange(data.shape[0]):
        frame = data[i,:].reshape((132, 132))
        frame_out = (frame * 127 / max(np.max(frame), np.abs(np.min(frame))) + 128).astype(np.uint8)
        print frame_out
        print frame_out.shape
        print np.min(frame_out)

        out.write(frame_out)

    out.release()

def main():
    pass

if __name__ == '__main__':
    main()
