import numpy as np
import pandas as pd

import os

import torch
from torch.utils import data
from torchvision import datasets

# transform to sequence
# add 1 for label
def data2seq(data, window_size):
    _list = []
    steps = 1
    for i in range(len(data)-(window_size+steps)+1):
        _list.append(data[i:i+window_size+steps].tolist())
    return np.array(_list)

# load data from json file
def getData(data, mode, test_ratio, window_size):
    if mode == 'train':
        tr_data = data[:-int(len(data)*test_ratio)*2]
        tr_data = data2seq(tr_data, window_size)
        return tr_data
    elif mode == 'val':
        val_data = data[-int(len(data)*test_ratio)*2:-int(len(data)*test_ratio)]
        val_data = data2seq(val_data, window_size)
        return val_data
    elif mode == 'test':
        ts_data = data[-int(len(data)*test_ratio):]
        ts_data = data2seq(ts_data, window_size)
        return ts_data

# data loader class
class NYCTaxiLoader(data.Dataset):
    def __init__(self, data, mode, test_ratio, window_size, max_value):
        self.mode = mode
        self.test_ratio= test_ratio
        self.window = window_size
        self.max_value = max_value
        self.data = getData(data, mode, test_ratio, window_size)
        print(f"> Found {len(self.data)} samples in {self.mode} data...\n")
        
    def get_scaler(self):
        return self.scaler

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        if torch.is_tensor(index):
            index = index.tolist()

        samples = self.data[[index]][0]

        _input =  np.array(samples[:-1]) / self.max_value # normalize x
        _target = np.array(samples[-1:])
        return _input, _target