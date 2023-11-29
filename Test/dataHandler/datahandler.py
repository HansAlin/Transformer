
import numpy as np
import torch 
from torch.utils.data import DataLoader, TensorDataset
import os
import pandas as pd


def load_test_data(data_path='/home/halin/Autoencoder/Data/', 
                    save_path='/home/halin/Master/Transformer/Test/test_data/test_data.npy',
                    train_size=1000):
  """
    This function loads data from ARIANNA group, downloaded localy
    Args:
    data_path: where the data from ARIANNA is stored
    train_size: size of training data, test data is 100
    save_path: where you store the sorted data
    Returns:
    x_train, y_train,  x_test, y_test 
    
  """
  NOISE_URL = data_path + 'trimmed100_data_noise_3.6SNR_1ch_0000.npy'
  noise = np.load(NOISE_URL)

  SIGNAL_URL = data_path + "trimmed100_data_signal_3.6SNR_1ch_0000.npy"
  signal = np.load(SIGNAL_URL)
  
  x = np.vstack((noise, signal))
  y = np.ones(len(x))
  y[:len(noise)] = 0

  # 10 % for testing
  test_size = int(np.floor(train_size*0.1/0.9))
  shuffle = np.arange(x.shape[0], dtype=np.int64)
  np.random.shuffle(shuffle)
  x = x[shuffle]
  y = y[shuffle]
  
  x_train = x[:(train_size)]
  y_train = y[:(train_size)]
  x_test = x[train_size:(train_size + test_size)]
  y_test = y[train_size:(train_size + test_size)]

  with open(save_path, 'wb') as f:
    np.save(f, x_train)
    np.save(f, x_test)
    np.save(f, y_train)
    np.save(f, y_test)

def get_test_data(path=''):
  """ This method loads test data from folder
    Arg:
      path: wher data is saved
    Ret:
      x_train, x_test, y_train, y_test

  """
  if path == '':
    path = os.getcwd()
    path = path + '/Test/data/test_data.npy' 

   

    
  with open(path, 'rb') as f:
    x_train = np.load(f)
    x_test = np.load(f)
    y_train = np.load(f)
    y_test = np.load(f)

  print(f"Shape: x_train {x_train.shape}, x_test {x_test.shape}, y_train {y_train.shape}, y_test {y_test.shape}")  
  return x_train, x_test, y_train, y_test

def prepare_data(x_train, x_test, y_train, y_test, batch_size):
  """ This methode takes train and test data as numpy arrays and
      return pytorch train and test DataLoader
    Arg: 
      x_train, x_test, y_train, y_test, batch_size
    Return:
      train_loader, test_loader 

  """
  print(f"Train size = {len(x_train)}")
  print(f"Test size: {len(x_test)}")

  # to torch
  x_train = standardScaler(torch.tensor(x_train, dtype=torch.float32).view(-1, len(x_train[0]), 1))
  x_test =  standardScaler(torch.tensor(x_test, dtype=torch.float32).view(-1, len(x_test[0]), 1))
  y_train = standardScaler(torch.tensor(y_train, dtype=torch.float32).view(-1, 1))
  y_test =  standardScaler(torch.tensor(y_test, dtype=torch.float32).view(-1,  1))



  train_dataset = TensorDataset(x_train, y_train)
  train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)

  test_dataset = TensorDataset(x_test, y_test)
  test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=True)
  
  return train_loader, test_loader

def standardScaler(x):
  mean = x.mean(0, keepdim=True)
  std = x.std(0, unbiased=False, keepdim=True)
  x -= mean
  x /= std
  return x