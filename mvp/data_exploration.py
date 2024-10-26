import os
import h5py
from mpl_toolkits import mplot3d
import numpy as np
import matplotlib.pyplot as plt

# This should go in the main file 
# file_path = os.path.realpath(__file__)
file_path = os.path.abspath(os.path.dirname(__file__))
data_path = file_path + "/datasets/"
train_data_path = data_path + "MVP_Train_CP.h5"
print(f"Data directory: {data_path}")
train_dataset = h5py.File(train_data_path, 'r') 

# This should be a function
# Load H5 data
print(f"Dataset: {train_dataset}")
print(f"INFO:")
print(f"datasets keys: {list(train_dataset.keys())}")
# output: ['complete_pcds', 'incomplete_pcds', 'labels']
print(f"dataset format: {type(train_dataset['complete_pcds'])}")
print(f"data sample format: {type(train_dataset['complete_pcds'][0])}")
print(f"complete point clouds : {train_dataset['complete_pcds']}")
print(f"incomplete point clouds : {train_dataset['incomplete_pcds']}")
print(f"first sample: {train_dataset['complete_pcds'][0]}")
print(f"shape: {train_dataset['complete_pcds'][0].shape}")

# This should be another function
# Data visualization
# plt.ion() # enable interactive mode
fig = plt.figure()
ax = plt.axes(projection='3d')
random_sample = np.random.randint(0, len(train_dataset['complete_pcds']))

# Complete sample
first_x = train_dataset['complete_pcds'][random_sample][:,0]
first_y = train_dataset['complete_pcds'][random_sample][:,1]
first_z = train_dataset['complete_pcds'][random_sample][:,2]
ax.scatter3D(first_x, first_y, first_z, color='blue', label=f'Complete sample {random_sample}')

# Complete sample
first_x = train_dataset['incomplete_pcds'][random_sample][:,0]
first_y = train_dataset['incomplete_pcds'][random_sample][:,1]
first_z = train_dataset['incomplete_pcds'][random_sample][:,2]
ax.scatter3D(first_x, first_y, first_z, color='red', label=f'Incomplete sample {random_sample}')

# Show 
ax.set_xlabel('X axis')
ax.set_ylabel('Y axis')
ax.set_zlabel('Z axis')
ax.legend()
plt.pause(0.001)
plt.show()

