import sys
import os
import re

files = os.listdir('.')
for file_name in files:
    print file_name
    json = re.compile('.json')
    png = re.compile('.png')
    if json.search(file_name):
        print 'rename ' + file_name
        file_name_elems = file_name.split('_')
        movie_index = int(file_name_elems[3])
        frame_index = int(file_name_elems[4].split('.')[0])
        if movie_index < 100:
            new_file_name = 'other_mit_body_v2_%05i_%05i.json' % (movie_index+1, frame_index)
        elif (movie_index == 100):
            new_file_name = 'other_mit_body_v2_%05i_%05i.json' % (1, frame_index)
        os.rename(file_name, new_file_name)
    elif png.search(file_name):
        print 'rename ' + file_name
        os.rename(file_name, 'other_' + file_name)
    else:
        pass
