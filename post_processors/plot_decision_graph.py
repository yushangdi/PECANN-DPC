import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import sys

assert(len(sys.argv) >= 2)

# first arg is the path to decision graph input data, the second arg is the number of clusters. 
# The number that separates the ith and i+1th largest y values is plotted.

decision_graph_path = sys.argv[1]

num_cluster = -1

if (len(sys.argv) > 2):
	num_cluster = int(sys.argv[2])

suffix = ""
if (len(sys.argv) > 3):
	suffix = sys.argv[3]

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

# filtered_data = data[(data[:, 0] > 0.7) & (data[:, 1] > 3)]
# sns.scatterplot(x=filtered_data[:, 0], y=filtered_data[:, 1])

# Compute the product for each point
products = data[:, 0] * data[:, 1]
# Get the indices that would sort the products
sorted_indices = np.argsort(products)
# Get the top 10 points
top_points = data[sorted_indices[-10:][::-1]]  # We reverse the indices to get the largest values
sns.scatterplot(x=top_points[:, 0], y=top_points[:, 1])


if i != -1:
	# Extract the second column
	second_col = data[:, 1]

	# Sort the second column in descending order
	sorted_vals = np.sort(second_col)[::-1]

	# Check if i+1 exists in the list
	if num_cluster < len(sorted_vals) - 1:
			# Get the number between ith and i+1th largest values
			mid_value = (sorted_vals[num_cluster-1] + sorted_vals[num_cluster]) / 2
			print("distance threshold", mid_value)
			plt.axhline(y = mid_value)
			y_ticks = plt.gca().get_yticks().tolist()
			y_ticks.append(mid_value)
			plt.yticks(sorted(y_ticks))
	else:
			print("num_cluster+1th largest value doesn't exist.", num_cluster)

top_indices = np.argsort(data[:, 1])[::-1][:num_cluster]
for idx in top_indices:
		x, y = data[idx]
		plt.text(x, y, f'ID:{idx}', ha='right', va='bottom')


plt.xlabel("density")
plt.ylabel("distance to denpendent point")
plt.savefig('results/decision_graph_%s.png' % suffix)
print("done")