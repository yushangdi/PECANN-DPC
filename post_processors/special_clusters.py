import sys
import numpy as np
from collections import Counter


assert(len(sys.argv) >= 2)

cluster_path = sys.argv[1]
with open(cluster_path, 'r') as file:
    labels = [int(line.rstrip()) for line in file]

c_labels = Counter(labels)
print(c_labels.most_common()[:20])

special_labels = [key for key, val in c_labels.items() if val<=1]

special_locs = [i for i,label in enumerate(labels) if label in special_labels]

print(special_locs)

