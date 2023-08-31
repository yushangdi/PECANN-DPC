import numpy as np
import matplotlib.pyplot as plt
import sys
from sklearn.preprocessing import LabelEncoder



def plot_dims(filename, image_path, cluster_path, d1=0, d2=1):
    # Read data
    with open(filename, 'r') as f:
        lines = f.readlines()

    with open(cluster_path, 'r') as file:
        preds = np.array([line.rstrip() for line in file])
    encoder = LabelEncoder()
    encoded_preds = encoder.fit_transform(preds)
    unique_encoded = np.unique(encoded_preds)
    unique_labels = encoder.inverse_transform(unique_encoded)

    data = [list(map(float, line.strip().split())) for line in lines]
    data = np.array(data)
    
    # Plot i and j dimensions
    fig = plt.scatter(data[:, d1], data[:, d2], c=encoded_preds, cmap='tab20', s = 4)
    # for i, label in zip(unique_encoded, unique_labels):
    #     plt.scatter(data[encoded_preds == i, d1], data[encoded_preds == i, d2], label=label, s = 4)

    
    plt.xlabel(f'Dimension {d1}')
    plt.ylabel(f'Dimension {d2}')
    plt.title(f'Plot of Dimension {d1} vs Dimension {d2}')
    # plt.legend(loc='upper right')
    plt.savefig(image_path)


if __name__ == "__main__":
    filename = sys.argv[1]
    cluster_path = sys.argv[2]
    image_path = sys.argv[3]
    d1 = int(sys.argv[4])
    d2 = int(sys.argv[5])
    plot_dims(filename, image_path, cluster_path, d1, d2,)