#-*- coding:utf-8 -*-

import os
import subprocess
noun_labels = ['Faces','Leopards','Motorbikes','accordion','airplanes','anchor','ant','barrel','bass','beaver','binocular','bonsai','brain','brontosaurus','buddha','butterfly','camera','cannon','car_side','ceiling_fan','cellphone','chair','chandelier','cougar_body','cougar_face','crab','crayfish','crocodile','crocodile_head','cup','dalmatian','dollar_bill','dolphin','dragonfly','electric_guitar','elephant','emu','euphonium','ewer','ferry','flamingo','flamingo_head','garfield','gerenuk','gramophone','grand_piano','hawksbill','headphone','hedgehog','helicopter','ibis','inline_skate','joshua_tree','kangaroo','ketch','lamp','laptop','llama','lobster','lotus','mandolin','mayfly','menorah','metronome','minaret','nautilus','octopus','okapi','pagoda','panda','pigeon','pizza','platypus','pyramid','revolver','rhino','rooster','saxophone','schooner','scissors','scorpion','sea_horse','snoopy','soccer_ball','stapler','starfish','stegosaurus','stop_sign','strawberry','sunflower','tick','trilobite','umbrella','watch','water_lilly','wheelchair','wild_cat','windsor_chair','wrench','yin_yang']

adj_labels = ["bitter",  "bright",  "cold",  "cool",  "dark",  "evil", "gorgeous",  "hard",  "hot",  "poor",  "salty",  "smooth",  "soft",  "sore",  "sour",  "spicy",  "sweet",  "tasty",  "tepid",  "textured",  "warm"]
list_p = []
f_0 = open('input_0.txt', 'w')
f_1 = open('input_1.txt', 'w')
f_2 = open('input_2.txt', 'w')
f_3 = open('input_3.txt', 'w')
i = 0
for n_label in noun_labels:
        for a_label in adj_labels:
                if os.path.exists("out/"+n_label+"/"+a_label)==False:
                        os.makedirs("out/"+n_label+"/"+a_label)
                list_p.append([n_label,a_label])
                #print(n_label)
                #print(a_label)
		if i%4 == 0:
			f_0.write("../moriga_cnn/chainer_imagenet_tools/101_ObjectCategories/"+n_label+"/image_0001.jpg ../texture/"+a_label+"/"+a_label+"1.jpg out/"+n_label+"/"+a_label+"\n")
			i = i + 1
		elif i%4 == 1:
			f_1.write("../moriga_cnn/chainer_imagenet_tools/101_ObjectCategories/"+n_label+"/image_0001.jpg ../texture/"+a_label+"/"+a_label+"1.jpg out/"+n_label+"/"+a_label+"\n")
			i = i + 1
		elif i%4 == 2:
			f_2.write("../moriga_cnn/chainer_imagenet_tools/101_ObjectCategories/"+n_label+"/image_0001.jpg ../texture/"+a_label+"/"+a_label+"1.jpg out/"+n_label+"/"+a_label+"\n")
 			i = i + 1
		elif i%4 == 3:
			f_3.write("../moriga_cnn/chainer_imagenet_tools/101_ObjectCategories/"+n_label+"/image_0001.jpg ../texture/"+a_label+"/"+a_label+"1.jpg out/"+n_label+"/"+a_label+"\n")
			i = i + 1
		#print("../moriga_cnn/chainer_imagenet_tools/101_ObjectCategories/"+n_label+"/image_0001.jpg ../texture/"+list_p[0][1]+"/"+a_label+"1.jpg")
f_0.close()
f_1.close()
f_2.close()
f_3.close()

