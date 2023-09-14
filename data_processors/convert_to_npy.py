import numpy as np

# Assuming the data is in a file called 'data.txt'
with open('./data/gaussian_example/gaussian_4_1000.data', 'r') as f:
    data = [list(map(float, line.strip().split())) for line in f]

print(data[10])
array_data = np.array(data)

# Save as npy file
np.save('./data/gaussian_example/gaussian_4_1000.npy', array_data)

print("Data saved")