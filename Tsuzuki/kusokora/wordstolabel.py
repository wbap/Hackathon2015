#!/usr/bin/env python
# coding: UTF-8
import sys
import os
import re
import gensim
from gensim.models import word2vec
def cos_distance(l1,l2):
	if len(l1)!= len(l2):
		exit("can't calculate cos d")
	summ = 0
	zipped_list = zip(l1,l2)
	for item in zipped_list:
		summ = summ + item[0]*item[1]
	return summ
argvs = sys.argv  
argc = len(argvs) 
if (argc != 2):  
    print 'Usage: # python %s filename' % argvs[0]
    quit()         
model = word2vec.Word2Vec.load_word2vec_format('GoogleNews-vectors-negative300.bin', binary=True)  # C binary format
print("model loaded")
labels = ["bitter",  "bright",  "cold",  "cool",  "dark",  "evil", "gorgeous",  "hard",  "hot",  "poor",  "salty",  "smooth",  "soft",  "sore",  "sour",  "spicy",  "sweet",  "tasty",  "tepid",  "textured",  "warm"]
vectors = []
vec = model[sys.argv[1]]
max_d = 0
for label in labels:
	if max_d < cos_distance(vec,model[label]):
		max_d = cos_distance(vec,model[label])
		result = label
print result
	
