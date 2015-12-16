#!/usr/bin/env python
# coding: UTF-8
import sys
import os
import re
a = 1
#指定する画像フォルダ
files = os.listdir('.')
for file in files:
    jpg = re.compile("jpg")
    png = re.compile("png")
    if jpg.search(file) or png.search(file):
        os.rename(file, "image%d.jpg" %(a))
        a+=1
    else:
        pass