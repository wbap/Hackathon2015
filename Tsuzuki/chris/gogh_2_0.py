#-*- coding:utf-8 -*-

import os
import subprocess,time
noun_labels = ['Faces','Faces_easy','Leopards','Motorbikes','accordion','airplanes','anchor','ant','barrel','bass','beaver','binocular','bonsai','brain','brontosaurus','buddha','butterfly','camera','cannon','car_side','ceiling_fan','cellphone','chair','chandelier','cougar_body','cougar_face','crab','crayfish','crocodile','crocodile_head','cup','dalmatian','dollar_bill','dolphin','dragonfly','electric_guitar','elephant','emu','euphonium','ewer','ferry','flamingo','flamingo_head','garfield','gerenuk','gramophone','grand_piano','hawksbill','headphone','hedgehog','helicopter','ibis','inline_skate','joshua_tree','kangaroo','ketch','lamp','laptop','llama','lobster','lotus','mandolin','mayfly','menorah','metronome','minaret','nautilus','octopus','okapi','pagoda','panda','pigeon','pizza','platypus','pyramid','revolver','rhino','rooster','saxophone','schooner','scissors','scorpion','sea_horse','snoopy','soccer_ball','stapler','starfish','stegosaurus','stop_sign','strawberry','sunflower','tick','trilobite','umbrella','watch','water_lilly','wheelchair','wild_cat','windsor_chair','wrench','yin_yang']

adj_labels = ["bitter",  "bright",  "cold",  "cool",  "dark",  "evil", "gorgeous",  "hard",  "hot",  "poor",  "salty",  "smooth",  "soft",  "sore",  "sour",  "spicy",  "sweet",  "tasty",  "tepid",  "textured",  "warm"]
list_p = [] 
i=0
for n_label in noun_labels:
	for a_label in adj_labels:
		if os.path.exists("out/"+n_label+"/"+a_label)==False:
			os.makedirs("out/"+n_label+"/"+a_label)
		list_p.append([n_label,a_label])
		#print(n_label)
		#print(a_label)
		#print(i)
		#i=i+1

i=1060
#print("python chainer-gogh.py -m vgg -i ../moriga_cnn/chainer_imagenet_tools/101_ObjectCategories/"+list_p[i][0]+"/image_0001.jpg -s ../texture/"+list_p[i][1]+"/"+list_p[i][1]+" -o out/"+list_p[i][0]+"/"+list_p[i][1]+"/1.jpg -g 0 --width 255")
p0 = subprocess.Popen(("python chainer-gogh.py -m vgg -i ../moriga_cnn/chainer_imagenet_tools/101_ObjectCategories/"+list_p[i][0]+"/image_0001.jpg -s ../texture/"+list_p[i][1]+"/"+list_p[i][1]+"1.jpg -o out/"+list_p[i][0]+"/"+list_p[i][1]+"/ -g 0 --width 255 --lam=0.0400 --iter=3001").split(" "))
print "GPU:0"+list_p[0][0]+":"+list_p[0][1] 
i = i + 1 
time.sleep(0.1)
p1 = subprocess.Popen(("python chainer-gogh.py -m vgg -i ../moriga_cnn/chainer_imagenet_tools/101_ObjectCategories/"+list_p[i][0]+"/image_0001.jpg -s ../texture/"+list_p[i][1]+"/"+list_p[i][1]+"1.jpg -o out/"+list_p[i][0]+"/"+list_p[i][1]+"/ -g 1 --width 255 --lam=0.0400 --iter=3001").split(" "))
print "GPU:1"+list_p[1][0]+":"+list_p[1][1]
i = i + 1
time.sleep(0.1)

p2 = subprocess.Popen(("python chainer-gogh.py -m vgg -i ../moriga_cnn/chainer_imagenet_tools/101_ObjectCategories/"+list_p[i][0]+"/image_0001.jpg -s ../texture/"+list_p[i][1]+"/"+list_p[i][1]+"1.jpg -o out/"+list_p[i][0]+"/"+list_p[i][1]+"/ -g 2 --width 255 --lam=0.0400 --iter=3001").split(" "))
print "GPU:2"+list_p[2][0]+":"+list_p[2][1]
i = i + 1

p3 = subprocess.Popen(("python chainer-gogh.py -m vgg -i ../moriga_cnn/chainer_imagenet_tools/101_ObjectCategories/"+list_p[i][0]+"/image_0001.jpg -s ../texture/"+list_p[i][1]+"/"+list_p[i][1]+"1.jpg -o out/"+list_p[i][0]+"/"+list_p[i][1]+"/ -g 3 --width 255 --lam=0.0400 --iter=3001").split(" "))
print "GPU:3"+list_p[3][0]+":"+list_p[3][1]
i = i + 1
while i<1252:
	#GPU0
	if p0.poll() is not None:
		p0 = subprocess.Popen(("python chainer-gogh.py -m vgg -i ../moriga_cnn/chainer_imagenet_tools/101_ObjectCategories/"+list_p[i][0]+"/image_0001.jpg -s ../texture/"+list_p[i][1]+"/"+list_p[i][1]+"1.jpg -o out/"+list_p[i][0]+"/"+list_p[i][1]+"/ -g 0 --width 255 --lam=0.0400 --iter=3001").split(" "))
		print "GPU:0"+list_p[i][0]+":"+list_p[i][1]
		i = i + 1
	if p1.poll() is not None:
		p1 = subprocess.Popen(("python chainer-gogh.py -m vgg -i ../moriga_cnn/chainer_imagenet_tools/101_ObjectCategories/"+list_p[i][0]+"/image_0001.jpg -s ../texture/"+list_p[i][1]+"/"+list_p[i][1]+"1.jpg -o out/"+list_p[i][0]+"/"+list_p[i][1]+"/ -g 1 --width 255 --lam=0.0400 --iter=3001").split(" "))
		print "GPU:1"+list_p[i][0]+":"+list_p[i][1]
		i = i + 1
	if p2.poll() is not None:
		p2 = subprocess.Popen(("python chainer-gogh.py -m vgg -i ../moriga_cnn/chainer_imagenet_tools/101_ObjectCategories/"+list_p[i][0]+"/image_0001.jpg -s ../texture/"+list_p[i][1]+"/"+list_p[i][1]+"1.jpg -o out/"+list_p[i][0]+"/"+list_p[i][1]+"/ -g 2 --width 255 --lam=0.0400 --iter=3001").split(" "))
		print "GPU:2"+list_p[i][0]+":"+list_p[i][1]
		i = i + 1
	if p3.poll() is not None:
		p3 = subprocess.Popen(("python chainer-gogh.py -m vgg -i ../moriga_cnn/chainer_imagenet_tools/101_ObjectCategories/"+list_p[i][0]+"/image_0001.jpg -s ../texture/"+list_p[i][1]+"/"+list_p[i][1]+"1.jpg -o out/"+list_p[i][0]+"/"+list_p[i][1]+"/ -g 3 --width 255 --lam=0.0400 --iter=3001").split(" "))
		print "GPU:3"+list_p[i][0]+":"+list_p[i][1]
		i = i + 1
	if i == 1252:
		break

p0.wait()
p1.wait()
p2.wait()
p3.wait()
