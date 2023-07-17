import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import sys

assert(len(sys.argv) >= 2)

decision_graph_path = sys.argv[1]

print(f'reading decision_graph from {decision_graph_path}')

with open(decision_graph_path, 'r') as file:
	data = np.array([[float(num) for num in line.split()] for line in file])

for i in range(len(data)):
	if data[i, 1] >  10**11:
		data[i, 1] = -1

max_dist = np.max(data[:, 1])
for i in range(len(data)):
	if data[i, 1] < 0:
		data[i, 1] = max_dist

sns.scatterplot(x=data[:, 0], y=data[:, 1])
plt.savefig('decision_graph.png')