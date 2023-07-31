import matplotlib.pyplot as plt
import numpy as np

def read_results_from_file(filepath):
    with open(filepath, 'r') as f:
        return [float(line.strip()) for line in f]

def plot_histogram(data):
    plt.hist(data, bins=50, edgecolor='black')  # Adjust the number of bins as needed
    plt.title('Histogram of Results')
    plt.xlabel('Value')
    plt.ylabel('Frequency')
    plt.yscale("log")
    plt.savefig("results/histogram.png")

if __name__ == "__main__":
    results = read_results_from_file("results/num_rounds.txt")  # Change filename if different
    plot_histogram(results)
    top_indices = np.argsort(results)[::-1][:10]
    for idx in top_indices:
        print(idx, results[idx])
