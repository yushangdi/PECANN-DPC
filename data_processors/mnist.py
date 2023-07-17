import pandas as pd
import numpy as np

df = pd.read_csv('../data/mnist_train.csv')

with open('../data/mnist.data', 'w') as f:
    lines = []
    for i, row in df.iterrows():
        nums = [str(num) for num in row.to_numpy()[1:]]
        lines.append(' '.join(nums)+'\n')
    f.writelines(lines)

with open('../data/mnist.gt', 'w') as f:
    lines = []
    for item in df['label']:
        lines.append(str(item)+'\n')
    f.writelines(lines)
