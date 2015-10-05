import sys
import os

for i in range(1000, 10000):
    for j in range(10, 510, 10):
        file_name = 'self_mit_body_v2_%05i_%05i' % (i+1, j)
        os.rename(file_name + 'json', file_name + '.json')
        os.rename(file_name + 'png', file_name + '.png')
