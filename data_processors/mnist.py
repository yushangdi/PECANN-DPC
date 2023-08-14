import pandas as pd
import numpy as np

import torchvision.datasets as datasets
import torchvision.transforms as transforms

mnist_train = datasets.MNIST(root='./data', train=True, transform=transforms.ToTensor(), download=True)
mnist_test = datasets.MNIST(root='./data', train=False, transform=transforms.ToTensor(), download=True)
print("datasets loaded")

def save_data_and_labels(data_filename, labels_filename):
    with open(data_filename, 'w') as data_file, open(labels_filename, 'w') as labels_file:
        for dataset in [mnist_train, mnist_test]:
            for image, label in dataset:
                # Flatten the 28x28 image into a 1D list
                flattened = image.flatten().numpy()
                
                # Convert pixel values to strings and write to the data file
                pixel_strings = [str(pixel) for pixel in flattened]
                data_line = ' '.join(pixel_strings) + '\n'
                data_file.write(data_line)
                
                # Write the label to the labels file
                labels_file.write(str(label) + '\n')

save_data_and_labels('./data/mnist.txt', './data/mnist.gt')
print("done")

# df = pd.read_csv('../data/mnist_train.csv')

# with open('../data/mnist.data', 'w') as f:
#     lines = []
#     for i, row in df.iterrows():
#         nums = [str(num) for num in row.to_numpy()[1:]]
#         lines.append(' '.join(nums)+'\n')
#     f.writelines(lines)

# with open('../data/mnist.gt', 'w') as f:
#     lines = []
#     for item in df['label']:
#         lines.append(str(item)+'\n')
#     f.writelines(lines)
