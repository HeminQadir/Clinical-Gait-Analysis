import numpy as np
import pickle
import os
         
# Define the number of features
num_channels = 24
num_features = 101

# Create an array to store the training data
num_samples_per_class = 10000  # One sample per class
training_data = np.zeros((num_samples_per_class * 3, num_channels, num_features))

# Create three distinct patterns for each class with unique randomness
np.random.seed(0)  # Set a unique seed for the first class
pattern1 = (2 * np.random.rand(num_samples_per_class, num_channels, num_features)) - 1

np.random.seed(1)  # Set a unique seed for the first class
pattern2 = (2 * np.random.rand(num_samples_per_class, num_channels, num_features)) - 1

np.random.seed(2)  # Set a unique seed for the second class
pattern3 = (2 * np.random.rand(num_samples_per_class, num_channels, num_features)) - 1

# Assign each pattern to a class
training_data[:num_samples_per_class] = pattern1
training_data[num_samples_per_class:2*num_samples_per_class] = pattern2
training_data[2*num_samples_per_class:] = pattern3

# You can also assign labels to each class
labels = np.array([0] * num_samples_per_class + [1] * num_samples_per_class + [2] * num_samples_per_class)


# Specify the directory where you want to save the data
save_dir = "/home/jacobo/Eirik/dataset"

# Create the directory if it doesn't exist
os.makedirs(save_dir, exist_ok=True)

# Save each sample as a separate pickle file
for i in range(len(training_data)):
    sample_data = training_data[i]
    sample_label = labels[i]
    
    sample_dict = {
        "data": sample_data,
        "label": sample_label
    }
    
    sample_filename = f"sample_{i}.pkl"
    sample_filepath = os.path.join(save_dir, sample_filename)
    
    with open(sample_filepath, "wb") as file:
        pickle.dump(sample_dict, file)
