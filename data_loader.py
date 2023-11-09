import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset
import os
import pandas as pd

# Find the folders with data files.
def find_file_names(data_folder):
    data_files = list()
    for x in sorted(os.listdir(data_folder)):
        data_file = os.path.join(data_folder, x)
        if os.path.isfile(data_file):
            data_files.append(data_file)
    return sorted(data_files)


def load_train_val_files(data_folder, split=True, split_ratio=0.1):
    
    file_ids = find_file_names(data_folder)
    num_files = len(file_ids)

    if num_files==0:
        raise FileNotFoundError('No data was provided.')

    if split:
        X_train, X_val = train_test_split(file_ids, test_size=split_ratio, 
                                        shuffle=True, random_state=42)
        return X_train, X_val
    else:
        X_train = file_ids
        return X_train


class dataset(Dataset):
    def __init__(self, data_folder, X_files, train=True):
        self.X_files = X_files
        self.train=train
        self.data_folder = data_folder

    def __len__(self):
        return len(self.X_files)

    def __getitem__(self, idx):
        file_ids = self.X_files[idx]
        my_df = pd.read_pickle(file_ids)
        return my_df