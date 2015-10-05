import sys
import os

files = os.listdir('.')
for file_name in files:
    print file_name
    os.rename(file_name, 'self_' + file_name)
