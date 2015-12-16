#!/usr/bin/env python
# coding: UTF-8


#print type(lines1)
for line in open('data2.txt','r'):
	#print(line)
	words = line[:-1].split('|')
	#print(words[2])
	if len(words)==7:
		sentence = "INSERT INTO review(id,p_id,impression,time,ip) VALUES(" + words[1] + ", "+words[2] + "," + words[3] + ",'" + words[4] +"','"+words[5]+"');"
		print(sentence)