#-*- coding:utf-8 -*-

import os,sys
import subprocess
noun_labels = ['Faces','Faces_easy','Leopards','Motorbikes','accordion','airplanes','anchor','ant','barrel','bass','beaver','binocular','bonsai','brain','brontosaurus','buddha','butterfly','camera','cannon','car_side','ceiling_fan','cellphone','chair','chandelier','cougar_body','cougar_face','crab','crayfish','crocodile','crocodile_head','cup','dalmatian','dollar_bill','dolphin','dragonfly','electric_guitar','elephant','emu','euphonium','ewer','ferry','flamingo','flamingo_head','garfield','gerenuk','gramophone','grand_piano','hawksbill','headphone','hedgehog','helicopter','ibis','inline_skate','joshua_tree','kangaroo','ketch','lamp','laptop','llama','lobster','lotus','mandolin','mayfly','menorah','metronome','minaret','nautilus','octopus','okapi','pagoda','panda','pigeon','pizza','platypus','pyramid','revolver','rhino','rooster','saxophone','schooner','scissors','scorpion','sea_horse','snoopy','soccer_ball','stapler','starfish','stegosaurus','stop_sign','strawberry','sunflower','tick','trilobite','umbrella','watch','water_lilly','wheelchair','wild_cat','windsor_chair','wrench','yin_yang']

adj_labels = ["bitter",  "bright",  "cold",  "cool",  "dark",  "evil", "gorgeous",  "hard",  "hot",  "poor",  "salty",  "smooth",  "soft",  "sore",  "sour",  "spicy",  "sweet",  "tasty",  "tepid",  "textured",  "warm"]
#list_p = []
argvs = sys.argv  
argc = len(argvs) 
i=0
if (argc != 3):   
    print 'Usage: # python %s filename' % argvs[0]
    quit()
  
for n_label in noun_labels:
        for a_label in adj_labels:
                if os.path.exists("out/"+n_label+"/"+a_label)==False:
                        os.makedirs("out/"+n_label+"/"+a_label)
                if n_label==argvs[1] and a_label==argvs[2]:
			print(n_label+","+a_label+" i:"+str(i))
		#list_p.append([n_label,a_label])
                #print(n_label)
                #print(a_label)
                #print(i)
                i=i+1

