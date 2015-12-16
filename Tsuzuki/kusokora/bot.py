#!/usr/bin/python
# -*- coding: utf-8 -*-

import tweepy
import sys,time,os
import codecs
from gensim.models import word2vec
from datetime import *
import time
import inspect

CONSUMER_KEY           =  'Fmx9H2RD96oECMuG1iSCmrVjf'
CONSUMER_SECRET       ='zWNVrQTmHRfbe6EOTOZMQULX8nYfW7k2bztmAI1t36ZrUEElDB'
TOKEN     = '3318358063-7nK7GhISNSVcNMoldutQbjIFrKAHkN1f3wCNvx5'
TOKEN_SECRET = 'fTEUFxacoJKglatSnrLLgYTdw5H2ch0i7oLUi8piHVKqS'
sys.stdout = codecs.getwriter('utf_8')(sys.stdout)
def cos_distance(l1,l2):
        if len(l1)!= len(l2):
                exit("can't calculate cos d")
        summ = 0
        zipped_list = zip(l1,l2)
        for item in zipped_list:
                summ = summ + item[0]*item[1]
        return summ

def bot(arr,model):
    try:
	noun_labels = ['Faces','Faces_easy','Leopards','Motorbikes','accordion','airplanes','anchor','ant','barrel','bass','beaver','binocular','bonsai','brain','brontosaurus','buddha','butterfly','camera','cannon','car_side','ceiling_fan','cellphone','chair','chandelier','cougar_body','cougar_face','crab','crayfish','crocodile','crocodile_head','cup','dalmatian','dollar_bill','dolphin','dragonfly','electric_guitar','elephant','emu','euphonium','ewer','ferry','flamingo','flamingo_head','garfield','gerenuk','gramophone','grand_piano','hawksbill','headphone','hedgehog',
#'helicopter','ibis','inline_skate','joshua_tree','kangaroo','ketch','lamp','laptop','llama','lobster','lotus','mandolin','mayfly','menorah','metronome','minaret','nautilus','octopus','okapi','pagoda','panda','pigeon','pizza','platypus','pyramid','revolver','rhino','rooster',
'saxophone','schooner','scissors','scorpion','sea_horse','snoopy','soccer_ball','stapler','starfish','stegosaurus','stop_sign','strawberry','sunflower','tick','trilobite','umbrella','watch','water_lilly','wheelchair','wild_cat','windsor_chair','wrench','yin_yang']
	adj_labels = ["bitter",  "bright",  "cold",  "cool",  "dark",  "evil", "gorgeous",  "hard",  "hot",  "poor",  "salty",  "smooth",  "soft",  "sore",  "sour",  "spicy",  "sweet",  "tasty",  "tepid",  "textured",  "warm"]
	auth = tweepy.OAuthHandler(CONSUMER_KEY, CONSUMER_SECRET)
        auth.set_access_token(TOKEN, TOKEN_SECRET)
        api = tweepy.API(auth_handler = auth)

        #print '====== user timeline ======'
        #for status in api.user_timeline():
        #    print status.text
	#timestamp表示
	
	now = int( time.mktime( datetime.now().timetuple() ) )


        #print '====== mentions ======'
        mentions = api.mentions_timeline(count=1)
        if mentions is not None:
	 for mention in mentions:
             #print(dir(mention))
	     created_at = int( time.mktime(mention.created_at.timetuple()))
	     print("created_at"+str(created_at)+" now"+str(now))
	     if (mention.id not in arr) and abs(created_at - now)<86400: 
                 arr.append(mention.id)
                 print mention.text
                 #print mention.date
                 print mention.user.screen_name
		 data = mention.text.split(' ')
		 noun = data[2]
		 adj = data[1]
		 if len(data)>=3:
	 		if noun in model and adj in model:
                                 vectors_0 = []
                                 vec_0 = model[noun]
                                 max_d_0 = 0
                                 for label in noun_labels:
                                         if label in model:
                                                 if max_d_0 < cos_distance(vec_0,model[label]):
                                                         max_d_0 = cos_distance(vec_0,model[label])
                                                         result_0 = label
                                 vectors_1 = []
                                 vec_1 = model[adj]
                                 max_d_1 = 0
                                 for label in adj_labels:
		 			if label in model:
                                         	if max_d_1 < cos_distance(vec_1,model[label]):
                                                	max_d_1 = cos_distance(vec_1,model[label])
                                                	result_1 = label
                                 result = result_0 + "," + result_1
				 print result
				 if os.path.exists("../chainer-gogh/out/"+result_0+"/"+result_1+"/im_03000.png"):
				 	print "../chainer-gogh/out/"+result_0+"/"+result_1+"/im_03000.png"
					fn = os.path.abspath( "../chainer-gogh/out/"+result_0+"/"+result_1+"/im_03000.png")
					api.update_with_media(fn,"@"+mention.user.screen_name+" input:"+noun+"+"+adj+" => ")
				 else:
					api.update_status(status=("@"+mention.user.screen_name+ " I'm sorry... I have no idea."))
			else:
				api.update_status(status=("@"+mention.user.screen_name+ " Too difficult for me..."))
		 else: 
		 	api.update_status(status=("@"+mention.user.screen_name+ " Please ask me in the correct format."))
		 print("it's valid")
	return arr
    except Exception, e:
        print >>sys.stderr, 'error: %s' % e
	return arr
        #sys.exit(1)

def main():
    arr = []
    model = word2vec.Word2Vec.load_word2vec_format('GoogleNews-vectors-negative300.bin', binary=True)    
    #model = word2vec.Word2Vec.load_word2vec_format('vector.bin', binary=True) 
    while True:
        arr = bot(arr,model)
	for item in arr:
		print(arr)
        time.sleep(60) 

if __name__ == '__main__':
    main()

