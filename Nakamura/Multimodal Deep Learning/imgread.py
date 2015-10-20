#coding:utf-8
from PIL import Image
import numpy as np
import csv
import pandas as pd
#from pylab import *


all_person = []
for p in range(1,13,1): #person
    all_digit = []
    for d in range(0,10,1): #digit
        all_frames = []
    
        for f in range(0,15,1): #frame
            filename = str(d) + "-" + str(f) + ".gif"
            path =  "/Users/kazeto/Desktop/data/Video/"
            dilectory = path + str(p) + "/all/" + filename

            img = np.array( Image.open(dilectory) )

            #arrary declear
            all_img = []
            img_array = []

            for k in range(len(img)):
                img_array = np.r_[img_array , img[k]]
            
            all_frames.append(img_array)
            img_array = []

        all_digit.append(all_frames)
        all_frames = []

    all_person.append(all_digit)
    all_digit = []
    
print '1:{} 2:{} 3:{} 4:{}'.format(len(all_person),len(all_person[0]),len(all_person[0][0]),len(all_person[0][0][0]))
#    print all_img

with open('/Users/kazeto/Desktop/data/img.csv','wb') as f:
    writer = csv.writer(f)
    writer.writerows(all_person)

