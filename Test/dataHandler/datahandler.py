
import numpy as np
import torch 
from torch.utils.data import DataLoader, TensorDataset
import os
import pandas as pd
import pickle
import sys



def load_raw_data(data_path='/home/halin/Autoencoder/Data/', 
                    save_path='/home/halin/Master/Transformer/Test/test_data/test_data.npy',
                    tot_size=None, channels=1):
  """
    This function loads data from ARIANNA group, downloaded localy
    Args:
    data_path: where the data from ARIANNA is stored
    train_size: size of training data, test data is 100
    save_path: where you store the sorted data
    Returns:
    x_train, y_train, x_val, y_val, x_test, y_test 
    
  """
  if save_path == '':
    save_path = os.getcwd() 
    save_path += f'../data' 

 

  if channels == 1:
    NOISE_URL = data_path + 'trimmed100_data_noise_3.6SNR_1ch_0009.npy'
    noise = np.load(NOISE_URL)
    NOISE_URL = data_path + 'trimmed100_data_noise_3.6SNR_1ch_0010.npy'
    noise = np.vstack((noise, np.load(NOISE_URL)))

    SIGNAL_URL = data_path + "trimmed100_data_signal_3.6SNR_1ch_0000.npy"
    signal = np.load(SIGNAL_URL)
    SIGNAL_URL = data_path + "trimmed100_data_signal_3.6SNR_1ch_0001.npy"
    signal = np.vstack((signal, np.load(SIGNAL_URL)))



  x = np.vstack((noise, signal))
  y = np.ones(len(x))
  y[:len(noise)] = 0

  # Shuffle data
  shuffle = np.arange(x.shape[0], dtype=np.int64)
  np.random.shuffle(shuffle)
  x = x[shuffle]
  y = y[shuffle]
  
  # 10 % for testing, 10 % for validation, 80 % for training
  if tot_size == None:
    tot_size = len(x)
  else:
    tot_size = tot_size
    x = x[:tot_size]
    y = y[:tot_size]

  train_size = int(np.floor(tot_size*0.8))
  test_size = int(np.floor((tot_size - train_size)/2))
  val_size = tot_size - train_size - test_size
  x_train = x[:(train_size)]
  y_train = y[:(train_size)]
  x_val = x[train_size:(train_size + val_size)]
  y_val = y[train_size:(train_size + val_size)]
  x_test = x[(train_size + val_size):]
  y_test = y[(train_size + val_size):]
 
  with open(save_path, 'wb') as f:
    np.save(f, x_train)
    np.save(f, x_val)
    np.save(f, x_test)
    np.save(f, y_train)
    np.save(f, y_val)
    np.save(f, y_test)

# load_raw_data(data_path='/home/hansalin/Code/Transformer/Test/data/rawdata/', 
#               tot_size=None, 
#               save_path='/home/hansalin/Code/Transformer/Test/data/data.npy'
#               )

def get_data(path=''):
  """ This method loads test data from folder
    Arg:
      path: wher data is saved
    Ret:
      x_train, x_test, x_val, y_train, y_val, y_test

  """
  print("Loading data...")
  
  if path == '':
    path = os.getcwd()
    path = path + '/Test/data/' 
    path = path + 'data.npy'  

  with open(path, 'rb') as f:
    x_train = np.load(f)
    x_val = np.load(f)
    x_test = np.load(f)
    y_train = np.load(f)
    y_val = np.load(f)
    y_test = np.load(f)

  print(f"Shape: x_train {x_train.shape}, x_test {x_test.shape}, y_train {y_train.shape}, y_test {y_test.shape}")  
  return x_train, x_test, x_val, y_train, y_val, y_test

def prepare_data(x_train, x_val, x_test, y_train, y_val, y_test, batch_size):
  """ This methode takes train and test data as numpy arrays and
      return pytorch train and test DataLoader
    Arg: 
      x_train, x_test, y_train, y_test, batch_size
    Return:
      train_loader, val_loader, test_loader, number_of_noise, number_of_signals 

  """
  print(f"Train size = {len(x_train)}")
  number_of_signals = len(y_train[y_train == 1])
  number_of_noise = len(y_train[y_train == 0])
  print(f"Number of signals = {number_of_signals}")
  print(f"Number of noise = {number_of_noise}")
  print(f"Test size: {len(x_test)}")
  print(f"Max x value: {np.max(x_train)}")
  print(f"Min x value: {np.min(x_train)}")

  # to torch
  x_train = standardScaler(torch.tensor(x_train, dtype=torch.float32).view(-1, len(x_train[0]), 1))
  x_val =  standardScaler(torch.tensor(x_val, dtype=torch.float32).view(-1, len(x_val[0]), 1))
  x_test =  standardScaler(torch.tensor(x_test, dtype=torch.float32).view(-1, len(x_test[0]), 1))
  y_train = torch.tensor(y_train, dtype=torch.float32).view(-1, 1)
  y_val =  torch.tensor(y_val, dtype=torch.float32).view(-1, 1)
  y_test =  torch.tensor(y_test, dtype=torch.float32).view(-1,  1)



  train_dataset = TensorDataset(x_train, y_train)
  train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)

  val_dataset = TensorDataset(x_val, y_val)
  val_loader = DataLoader(dataset=val_dataset, batch_size=batch_size, shuffle=True)

  test_dataset = TensorDataset(x_test, y_test)
  test_loader = DataLoader(dataset=test_dataset, batch_size=1, shuffle=True)
  
  return train_loader, val_loader, test_loader, number_of_noise, number_of_signals

def standardScaler(x):

  mean = x.mean(0, keepdim=True)
  std = x.std(0, unbiased=False, keepdim=True)
  x -= mean
  x /= std
  return x

def save_model(trained_model, optimizer, config, global_step):

  path = config['model_path']

  saved_model_path = path + f'/saved_model'
  isExist = os.path.exists(saved_model_path)
  if not isExist:
    os.makedirs(saved_model_path)
    print("The new directory is created!")    
  torch.save({
              'epoch': config['current_epoch'],
              'model_state_dict': trained_model.state_dict(), 
              'optimizer_state_dict': optimizer.state_dict(),
              'global_step': global_step},
              saved_model_path + f'/model_{config["model_num"]}.pth')




def save_data(config, df):

  path = config['model_path']

  with open(path + 'config.txt', "wb") as fp:
    pickle.dump(config, fp)

  # A easier to read part
  with open(path + 'text_config.txt', 'w') as data:
    for key, value in config.items():
      data.write('%s: %s\n' % (key, value))

  df.to_pickle(path + 'dataframe.pkl')
  

def create_model_folder(model_number, path=''):
  """
  This function create a folder to save all model related stuf in.
  """
  # TODO uncomment after testing
  if path == '':
    path = os.getcwd() 
    path += f'/Test/ModelsResults/model_{model_number}/' 

    isExist = os.path.exists(path)
    if not isExist:
      os.makedirs(path)
      print("The new directory is created!")
    # else:
    #   reply = input("Folder already exist, proceed y/n ?")
    #   if reply.lower() != 'y':
    #     sys.exit()    
        
    return path 