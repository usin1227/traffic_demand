import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from copy import copy, deepcopy
from datetime import datetime
from tqdm import tqdm
import json

import torch
import torch.nn as nn
import torch.optim as optim

import dataloader as dataloader # dataloader.py
import cnn as models # models.py
import utils as utils # utils.py

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "6"

import warnings 
warnings.filterwarnings("ignore")

torch.backends.cudnn.enabled = False

# check device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print('INFO:')
print("PyTorch Version: ", torch.__version__)
print('GPU State:', device)


# hyper parameters
window_size = 1
test_ratio = 0.2
n_batch = 8
n_workers = 4
n_epochs = 300
learning_rate = 0.0001
weight_decay = 1e-6
early_stopping_patience = 4
gamma = 1

loss_type = 'mse'

grid_size = (10,20)
spatial_hidden_size = 64
temporal_hidden_size = 512


# data loader
def create_dataloader(data, max_value):
    train_dataset = dataloader.NYCTaxiLoader(data=data, mode="train", test_ratio=test_ratio, window_size=window_size, max_value=max_value) 
    train_loader = torch.utils.data.DataLoader(
        dataset=train_dataset, 
        batch_size=n_batch, 
        shuffle=True, 
        num_workers=n_workers
    )
    val_dataset = dataloader.NYCTaxiLoader(data=data, mode="val", test_ratio=test_ratio, window_size=window_size, max_value=max_value) 
    val_loader = torch.utils.data.DataLoader(
        dataset=val_dataset, 
        batch_size=n_batch, 
        shuffle=False, 
        num_workers=n_workers
    )
    test_dataset = dataloader.NYCTaxiLoader(data=data, mode="test", test_ratio=test_ratio, window_size=window_size, max_value=max_value) 
    test_loader = torch.utils.data.DataLoader(
        dataset=test_dataset, 
        batch_size=n_batch, 
        shuffle=False, 
        num_workers=n_workers
    )
    loaders = {'train':train_loader, 'val':val_loader, 'test':test_loader}
    return loaders

def paper_loss(y_pred, y_true):
	loss = torch.tensor(0.0, device=device)
	y_true, y_pred = torch.flatten(y_true), torch.flatten(y_pred)
	for i in range(len(y_true)):
		if y_true[i] != 0:
			loss += (y_true[i]-y_pred[i])**2 + gamma*(((y_true[i]-y_pred[i])/y_true[i])**2)
	return loss/len(y_true)

# main 
root_path = "./data2"
volume_train = np.load("./data2/volume_train.npz")
volume_test = np.load("./data2/volume_test.npz")
volume_data = np.concatenate([volume_train['volume'], volume_test['volume']], axis=0)
print(f"Total data: {volume_data.shape}\n")

max_value = volume_data[:-int(len(volume_data)*test_ratio)*2].max() # data range [0, max_value] to back transform the predict value
dataloaders = create_dataloader(volume_data, max_value)


# train
best_loss = 99999
last_loss = -1
times = 0
early_stop = False
loss_dict = {'mse': nn.MSELoss(), 'mae': nn.L1Loss()}

model = models.DMVSTNet(spatial_hidden_size=spatial_hidden_size)
model.to(device)
criterion = loss_dict[loss_type]
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, betas=(0.9, 0.999), eps=1e-08, weight_decay=weight_decay)

for epoch in range(n_epochs): 
    if early_stop: 
        print("Early Stop\n")
        break
        
    print(f'Epoch {epoch+1}/{n_epochs}')
    mape_list, rmse_list, train_loss_list, val_loss_list = [], [], [], []
    
    for phase in ['train', 'val']:   
        print(phase)
        if phase == 'train':
            model.train()  # Set model to training mode
        elif phase == 'val':
            model.eval()   # Set model to evaluate mode

        running_loss = 0.0
        actual = []
        predicted = []
        
        for batch_x, batch_y in dataloaders[phase]:
            inputs, targets = batch_x.to(device).float(), batch_y.to(device).float()
            optimizer.zero_grad()
               
            with torch.set_grad_enabled(phase == 'train'):
                outputs = model(inputs[:,0,:,:,:])

                y_true = targets[:,:,:,:,0].view(len(targets),-1)
                y_pred = outputs * max_value # denormalize

                loss = paper_loss(y_pred, y_true)
                if phase == 'train':
                    loss.backward()
                    optimizer.step()
                actual.extend(y_true.cpu().detach().numpy())
                predicted.extend(y_pred.cpu().detach().numpy())
                running_loss += loss.item() * inputs.size(0)
        
        # loss
        epoch_loss = running_loss / len(dataloaders[phase].dataset)
        if phase == 'val' and epoch_loss < best_loss:
            best_loss = epoch_loss
            best_model_wts = deepcopy(model.state_dict())
        
        # Early Stopping
        if phase == 'train':
            if epoch_loss > last_loss:
                times += 1
                if times >= early_stopping_patience:
                    early_stop = True
            else:
                times = 0
            last_loss = epoch_loss
            
        # Metrics
        actual, predicted = np.array(actual).T[0], np.array(predicted).T[0]
        rmse = utils.root_mean_squared_error(actual, predicted)
        mape = utils.mean_absolute_percentage_error(actual, predicted)            
        print(f'| {phase} Loss: {epoch_loss:.5f} | rmse: {rmse:.5f} | mape: {mape:.5f}')
        
        mape_list.append(mape)
        rmse_list.append(rmse)
        if phase == 'train':
            train_loss_list.append(epoch_loss)
        else:
            val_loss_list.append(epoch_loss)
    print()



# output results
from datetime import datetime
timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
store_path = f"./results/result_{timestamp}"

# build directory
try:
    os.mkdir(store_path)
except OSError:
    print(f"Creation of the directory {store_path} failed")

# store files
with open(os.path.join(store_path, 'metrics_results.npy'), 'wb') as f:
    np.save(f, np.array(rmse_list))
    np.save(f, np.array(mape_list))
with open(os.path.join(store_path, 'loss_results.npy'), 'wb') as f:
    np.save(f, np.array(train_loss_list))
    np.save(f, np.array(val_loss_list))
torch.save(best_model_wts, os.path.join(store_path, "weight"))

note_json = {
    'window_size': window_size,
    'test_ratio': test_ratio,
    'n_batch': n_batch,
    'n_workers': n_workers,
    'n_epochs': n_epochs,
    'learning_rate': learning_rate,
    'weight_decay': weight_decay,
    'early_stopping_patience': early_stopping_patience,

    'loss': loss_type,
    
    'grid_size': grid_size,
    'spatial_hidden_size': spatial_hidden_size,
    'temporal_hidden_size': temporal_hidden_size
}
with open(os.path.join(store_path, 'note.txt'), 'w') as outfile:
    json.dump(note_json, outfile)

print(f"results stored: {store_path}")

