import os

path = '../data/road'

with open(path+'.orig') as f:
    lines = f.readlines()

labels = [line.split(',')[0]+'\n' for line in lines]

lines = [' '.join(line.split(',')[1:]) for line in lines]

with open(path+'.data', 'w') as f:
    f.writelines(lines)

with open(path+'.gt', 'w') as f:
    f.writelines(labels)
