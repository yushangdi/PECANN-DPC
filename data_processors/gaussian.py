import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import sys

assert(len(sys.argv) >= 3)

seed_value = 42
np.random.seed(seed_value)

cluster_n = int(sys.argv[1])
approx_n = int(sys.argv[2])
dim = 2
if(len(sys.argv) >= 4):
	dim = int(sys.argv[3])

cluster_sizes = []
plot = True

with open("./data/gaussian_%s_%s_%s.data" % (cluster_n, approx_n, dim), 'w') as f:
	for i in range(cluster_n):
		means = [np.random.uniform(0, 100) for _ in range(dim)]
		# mx = np.random.uniform(0, 100)
		# my = np.random.uniform(0, 100)
		n = np.random.randint(approx_n // cluster_n // 1.5, approx_n // cluster_n * 2)
		print(n)
		cluster_sizes.append(n)
		# xs = np.random.normal(mx, 2, n)
		# ys = np.random.normal(my, 2, n)
		points = []
		for mean in means:
				points.append(np.random.normal(mean, 2, n))

		if plot:
			sns.scatterplot(x = points[0], y = points[1])
			# sns.scatterplot(x = xs, y = ys)

		# Transpose to get the format [[x1, y1, z1,...], [x2, y2, z2,...], ...]
		points = list(zip(*points))
		lines = [' '.join(map(str, point)) + '\n' for point in points]
		# lines = [f'{str(x)} {str(y)}\n' for x, y in zip(xs, ys)]
		f.writelines(lines)


with open('./data/gaussian_%s_%s_%s.gt'  % (cluster_n, approx_n, dim), 'w') as f:
	for i, cluster_size in enumerate(cluster_sizes):
		f.writelines([f'{str(i+1)}\n' for j in range(cluster_size)])

if plot:
	plt.savefig("./data/gaussian_%s_%s_%s.png" % (cluster_n, approx_n, dim))
print("done")