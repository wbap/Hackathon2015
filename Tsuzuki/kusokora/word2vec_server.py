#!/usr/bin/env python
# coding: UTF-8
from __future__ import print_function
import socket
from contextlib import closing
import sys
import os
import re
from gensim.models import word2vec
import select
global model 
def cos_distance(l1,l2):
        if len(l1)!= len(l2):
                exit("can't calculate cos d")
        summ = 0
        zipped_list = zip(l1,l2)
        for item in zipped_list:
                summ = summ + item[0]*item[1]
        return summ

def main():
  host = '127.0.0.1'
  port = 8080
  backlog = 10
  bufsize = 40960
  server_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
  readfds = set([server_sock])
  #model = word2vec.Word2Vec.load_word2vec_format('GoogleNews-vectors-negative300.bin', binary=True)
  model = word2vec.Word2Vec.load_word2vec_format('vector.bin', binary=True)
  noun_labels = ['Faces','Faces_easy','Leopards','Motorbikes','accordion','airplanes','anchor','ant','barrel','bass','beaver','binocular','bonsai','brain','brontosaurus','buddha','butterfly','camera','cannon','car_side','ceiling_fan','cellphone','chair','chandelier','cougar_body','cougar_face','crab','crayfish','crocodile','crocodile_head','cup','dalmatian','dollar_bill','dolphin','dragonfly','electric_guitar','elephant','emu','euphonium','ewer','ferry','flamingo','flamingo_head','garfield','gerenuk','gramophone','grand_piano','hawksbill','headphone','hedgehog','helicopter','ibis','inline_skate','joshua_tree','kangaroo','ketch','lamp','laptop','llama','lobster','lotus','mandolin','mayfly','menorah','metronome','minaret','nautilus','octopus','okapi','pagoda','panda','pigeon','pizza','platypus','pyramid','revolver','rhino','rooster','saxophone','schooner','scissors','scorpion','sea_horse','snoopy','soccer_ball','stapler','starfish','stegosaurus','stop_sign','strawberry','sunflower','tick','trilobite','umbrella','watch','water_lilly','wheelchair','wild_cat','windsor_chair','wrench','yin_yang']
  adj_labels = ["bitter",  "bright",  "cold",  "cool",  "dark",  "evil", "gorgeous",  "hard",  "hot",  "poor",  "salty",  "smooth",  "soft",  "sore",  "sour",  "spicy",  "sweet",  "tasty",  "tepid",  "textured",  "warm"]
  #model = word2vec.Word2Vec.load_word2vec_format('vector.bin', binary=True)
  print("model is set")
  #for item in model["i"]:
#	print(item)
  try:
    server_sock.bind((host, port))
    server_sock.listen(backlog)

    while True:
      rready, wready, xready = select.select(readfds, [], [])
      for sock in rready:
        if sock is server_sock:
          	conn, address = server_sock.accept()
          	#print("received:"+conn)
		readfds.add(conn)
        else:
          msg = sock.recv(bufsize)
	  msg_arr = msg.split(',')
          if len(msg) == 0:
            sock.close()
            readfds.remove(sock)
          else:
            	print(msg)
		if len(msg_arr)== 2:
			if msg_arr[0] in model and msg_arr[1] in model:
        			vectors_0 = []
        			vec_0 = model[msg_arr[0]]
        			max_d_0 = 0
        			for label in noun_labels:
                			if label in model:
						if max_d_0 < cos_distance(vec_0,model[label]):
                        				max_d_0 = cos_distance(vec_0,model[label])
                        				result_0 = label
				vectors_1 = []
                        	vec_1 = model[msg_arr[1]]
                        	max_d_1 = 0
                        	for label in adj_labels:
                                	if max_d_1 < cos_distance(vec_1,model[label]):
                                       		max_d_1 = cos_distance(vec_1,model[label])
                                        	result_1 = label
				result = result_0 + "," + result_1
				sock.send(result)
		else:
			print("isn't included")
			sock.send("#illegal")
  finally:
    for sock in readfds:
      sock.close()
  return

if __name__ == '__main__':
  main()
