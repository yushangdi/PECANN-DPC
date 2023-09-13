import pandas as pd
import numpy as np

import torchvision.datasets as datasets
import torchvision.transforms as transforms

mnist_train = datasets.MNIST(
    root="data/mnist", train=True, transform=transforms.ToTensor(), download=True
)
mnist_test = datasets.MNIST(
    root="data/mnist", train=False, transform=transforms.ToTensor(), download=True
)

all_data = np.vstack([mnist_train.data.numpy(), mnist_test.data.numpy()]).reshape(
    70000, 28 * 28
)
all_labels = np.vstack(
    [
        mnist_train.targets.numpy().reshape(60000, 1),
        mnist_test.targets.numpy().reshape(10000, 1),
    ]
)

np.save("data/mnist/mnist.npy", all_data)
np.save("data/mnist/mnist_gt.npy", all_labels)
np.savetxt("data/mnist/mnist.txt", all_data, fmt="%i", delimiter=" ")
np.savetxt("data/mnist/mnist.gt", all_labels, fmt="%i")
