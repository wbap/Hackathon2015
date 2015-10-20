#coding:utf-8
import numpy as np
import csv

def mfcc_convert():
    mfcc = []
    with open('./all_mfcc.csv','rb') as f:
        reader = csv.reader(f)
        for row in reader:
            mfcc.append(row)

    converted_mfcc = []
    each_person = []
    all_person_mfcc = []
    count = 0

    for j in range(0,120,1):
        start = 0
    
        for i in range(1,16,1):
            converted_mfcc.append(mfcc[j][start:start+117])
            start = start + 117
    

        each_person.append(converted_mfcc)
        converted_mfcc = []

        count = count + 1

        if count == 10:
            all_person_mfcc.append(each_person)
            each_person = []
            count = 0

    return all_person_mfcc

def mfcc_convert2(mfcc1):
    list = []
    for i in range(12):
        for j in range(10):
            for k in range(15):
                list.append(mfcc1[i][j][k])

    return list

def img_convert(data):
    list = []
    for i in range(12):
        for j in range(10):
            for k in range(15):
                list.append(data[i][j][k])
    
    return list

#def gen_target():
#    target = range(150) * 12
#
#    return target

def gen_target():
    target = []
    sum_target = []
    for i in range(10):
        list = [int(i)] * 15
        target = np.r_[target,list]
    for i in range(12):
        sum_target = np.r_[sum_target,target]

    return sum_target






