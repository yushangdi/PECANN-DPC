import numpy as np
import matplotlib.pyplot as plt
import sys


def plot_dims(filename, i, j):
    # Read data
    with open(filename, 'r') as f:
        lines = f.readlines()

    data = []
    for line in lines:
        data.append(list(map(float, line.strip().split())))
    data = np.array(data)
    
    # Plot i and j dimensions
    plt.scatter(data[:, i-1], data[:, j-1])
    plt.xlabel(f'Dimension {i}')
    plt.ylabel(f'Dimension {j}')
    plt.title(f'Plot of Dimension {i} vs Dimension {j}')
    plt.savefig("./data/tmp.png")

# Test the function
i = int(sys.argv[1])
j = int(sys.argv[2])
plot_dims("./data/gaussian_4_10000_128.data", i, j)