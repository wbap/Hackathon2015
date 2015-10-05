import sys
import os
import re

argvs = sys.argv
argc = len(argvs)
increment_movie_index = int(argvs[1])

files = os.listdir('.')
for file_name in files:
    print file_name
    json = re.compile('.json')
    png = re.compile('.png')
    if json.search(file_name) or png.search(file_name):
        print 'rename ' + file_name
        file_name_elems = file_name.split('_')
        movie_index = int(file_name_elems[4])
        frame_index = int(file_name_elems[5].split('.')[0])
        extension = file_name_elems[5].split('.')[1]
        new_file_name = 'self_mit_body_v2_%05i_%05i' % (movie_index+increment_movie_index, frame_index)
        os.rename(file_name, new_file_name + '.' + extension)
    else:
        pass
