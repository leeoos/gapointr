import os
import h5py
from mpl_toolkits import mplot3d
import numpy as np
import matplotlib.pyplot as plt

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
SAVE_DIR = os.path.join(BASE_DIR, "plots/")
os.makedirs(SAVE_DIR, exist_ok=True)


def exploration(name):

    # This should go in the main file 
    # file_path = os.path.realpath(__file__)
    file_path = os.path.abspath(os.path.dirname(__file__))
    data_path = file_path + "/datasets/"
    data_path = data_path + name
    print(f"Data directory: {data_path}")
    dataset = h5py.File(data_path, 'r') 

    # This should be a function
    # Load H5 data

    print(f"Dataset: {dataset}")
    print(f"INFO:")
    print(f"datasets keys: {list(dataset.keys())}")
    # output: ['complete_pcds', 'incomplete_pcds', 'labels']
    print(f"dataset format: {type(dataset['complete_pcds'])}")
    print(f"data sample format: {type(dataset['complete_pcds'][0])}")
    print(f"complete point clouds : {dataset['complete_pcds']}")
    print(f"incomplete point clouds : {dataset['incomplete_pcds']}")
    print(f"first sample: {dataset['complete_pcds'][0]}")
    print(f"shape: {dataset['complete_pcds'][0].shape}")
    print(f"label: {dataset['labels'][0]}")


    all_labels = {}
    for label in dataset['labels']:
        all_labels.setdefault(label, 0)
        all_labels[label] += 1

    data = np.array([
        value
        for kay, value in all_labels.items()
    ])

    print(f"Number of classes: {len(data)}")

    return data

    # This should be another function
    # Data visualization
    # plt.ion() # enable interactive mode
    # fig = plt.figure()
    # ax = plt.axes(projection='3d')
    # random_sample = np.random.randint(0, len(train_dataset['complete_pcds']))

    # # Complete sample
    # first_x = train_dataset['complete_pcds'][random_sample][:,0]
    # first_y = train_dataset['complete_pcds'][random_sample][:,1]
    # first_z = train_dataset['complete_pcds'][random_sample][:,2]
    # ax.scatter3D(first_x, first_y, first_z, color='blue', label=f'Complete sample {random_sample}')

    # # Complete sample
    # first_x = train_dataset['incomplete_pcds'][random_sample][:,0]
    # first_y = train_dataset['incomplete_pcds'][random_sample][:,1]
    # first_z = train_dataset['incomplete_pcds'][random_sample][:,2]
    # ax.scatter3D(first_x, first_y, first_z, color='red', label=f'Incomplete sample {random_sample}')

    # # Show 
    # ax.set_xlabel('X axis')
    # ax.set_ylabel('Y axis')
    # ax.set_zlabel('Z axis')
    # ax.legend()
    # plt.pause(0.001)
    # plt.show()

def dist_hist(data, name, colors):

    title = " ".join(name.split('_'))

    # # Define bins based on the data range
    # bins = np.arange(min(data), max(data) + 2, 1)  # Bin size = 1 for clarity
    
    # # Set the histogram limits
    # plt.xlim([min(data)-1, max(data)+1])
    # plt.ylim(bottom=0)  # Ensure y-axis starts at 0
    
    # # Create the histogram
    # plt.hist(data, bins=bins, alpha=0.5, edgecolor='black')  # Add edges for clarity

    # Generate indices (classes) from the length of the data
    indices = range(len(data))  # Class indices
    
    # Create the bar chart
    # plt.bar(indices, data, alpha=0.7, edgecolor='black')
    plt.bar(indices, data, alpha=0.7, edgecolor='lightgrey', width=1.0, color=colors)
    
    # Add titles and labels
    plt.title(f'Histogram of {title} set')
    plt.xlabel('classes')
    plt.ylabel('count')
    
    plt.ylim(bottom=0)  # Ensure y-axis starts at zero
    plt.xticks(indices)  # Ensure each column has its corresponding index
    
    # Save the figure
    plt.savefig(os.path.join(SAVE_DIR, f"{name}.png"))
    plt.close()  # Close the plot to avoid overlap in subsequent calls


if __name__ == "__main__":

    train_data = exploration("MVP_Train_CP.h5")
    test_data = exploration("MVP_Test_CP.h5")

    dist_hist(train_data, "MVP_Train", colors='#1f77b4')
    dist_hist(test_data, "MVP_Test", colors='#ff7f0e')

    # Blue: #1f77b4 
    # Orange: #ff7f0e