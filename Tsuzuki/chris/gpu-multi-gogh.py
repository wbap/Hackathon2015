#-*- coding:utf-8 -*-

import os
import subprocess

p0 = subprocess.Popen(("python chainer-gogh.py -m vgg -i input_0.txt -o out_0 --g 0 --width 255 --lam=0.0200 --iter=3000").split(" "))
p1 = subprocess.Popen(("python chainer-gogh.py -m vgg -i input_1.txt -o out_1 --g 1 --width 255 --lam=0.0200 --iter=3000").split(" "))
p2 = subprocess.Popen(("python chainer-gogh.py -m vgg -i input_2.txt -o out_2 --g 2 --width 255 --lam=0.0200 --iter=3000").split(" "))
p3 = subprocess.Popen(("python chainer-gogh.py -m vgg -i input_3.txt -o out_3 --g 3 --width 255 --lam=0.0200 --iter=3000").split(" "))

