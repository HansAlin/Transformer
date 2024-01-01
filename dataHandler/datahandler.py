import argparse
import numpy as np
import torch 
from torch import nn
from torch.utils.data import DataLoader, Dataset, TensorDataset
from matplotlib import pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.ticker import MultipleLocator
import os
import pandas as pd
import pickle
import sys
import time
from tqdm import tqdm



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

def get_test_data(path=''):
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

def get_data(batch_size, seq_len, subset=True, data_config_path='/home/halin/Master/Transformer/data_config.yaml'):
  """
  This function loads data from XXXX and returns train, test and validation data
  as DatasetContinuousStreamStitchless objects. It needs a data_config_path to
  know where to look for parameters for the data
  Arg:
    batch_size: batch size
    seq_len: sequence length
    subset: if True, only one file is loaded

  Ret:
    train_loader, val_loader, test_loader  
  """
  CODE_DIR_1  ='/home/acoleman/software/NuRadioMC/'
  sys.path.append(CODE_DIR_1)
  CODE_DIR_2 = '/home/acoleman/work/rno-g/'
  sys.path.append(CODE_DIR_2)
  type(sys.path)
  for path in sys.path:
    print(path)


  from NuRadioReco.utilities import units

  from analysis_tools import data_locations
  from analysis_tools.config import GetConfig
  from analysis_tools.data_loaders import DatasetContinuousStreamStitchless
  from analysis_tools.Filters import GetRMSNoise
  from analysis_tools.model_loaders import ConstructModelFromConfig, LoadModelFromConfig

  # from networks.Chunked import CountModelParameters, ChunkedTrainingLoop
  # from networks.Cnn import CnnTrainingLoop

  # plt.rcParams["font.family"] = "serif"
  # plt.rcParams["mathtext.fontset"] = "dejavuserif"
  ABS_PATH_HERE = str(os.path.dirname(os.path.realpath(__file__)))

  # parser = argparse.ArgumentParser()
  # parser.add_argument("--config", required=True, help="Yaml file with the configuration")
  # parser.add_argument("--seed", type=int, default=123, help="Numpy files to train on")
  # parser.add_argument("--test_frac", type=float, default=0.15, help="Fraction of waveforms to use for testing")
  # parser.add_argument("--n_epochs", type=int, default=50, help="Number of epoch to train on")
  # parser.add_argument("--cuda_core", type=str, default="", help="Which core to run on")
  # args = parser.parse_args()

  # Read the configuration for this training/network
 
  config = GetConfig(data_config_path)
  model_string = config["name"]

  band_flow = config["sampling"]["band"]["low"]
  band_fhigh = config["sampling"]["band"]["high"]
  sampling_rate = config["sampling"]["rate"]
  wvf_length = config["input_length"]

  config['training']['batch_size'] = batch_size
  config['input_length'] = seq_len

  random_seed = 123
  np_rng = np.random.default_rng(random_seed)

  waveform_filenames = data_locations.NoiselessSigFiles(
      cdf=config["training"]["cdf"], config=config, nu="*", inter="cc", lgE="1?.??"
  )
  print(f"\tFound {len(waveform_filenames)} signal files")


  waveforms = None
  signal_labels = None

  if subset:
    waveform_filenames = [waveform_filenames[0]]
    print(f"Only using {len(waveform_filenames)} files")

  print("\tReading in signal waveforms")
  t0 = time.time()
  # Precalculate how much space will be needed to read in all waveforms
  total_len = 0
  print("\t\tCalculating required size")
  for filename in waveform_filenames:
      shape = np.load(filename, mmap_mode="r")["wvf"].shape
      total_len += shape[0]
  waveforms = np.zeros((total_len, shape[1], shape[2]))
  signal_labels = np.zeros((total_len, shape[1], shape[2]))
  snrs = np.zeros((total_len, shape[1]))

  total_len = 0
  print("\t\tReading into RAM")
  for filename in tqdm(waveform_filenames):
      this_dat = np.load(filename)
      waveforms[total_len : total_len + len(this_dat["wvf"])] = this_dat["wvf"]
      signal_labels[total_len : total_len + len(this_dat["wvf"])] = this_dat["label"]
      snrs[total_len : total_len + len(this_dat["wvf"])] = this_dat["snr"]
      total_len += len(this_dat["wvf"])

  assert len(waveforms)
  assert waveforms.shape == signal_labels.shape
  print(
      f"\t\tLoaded signal data of shape {waveforms.shape} --> {waveforms.shape[0] * waveforms.shape[-1] / sampling_rate / units.s:0.3f} s of data"
  )




  if np.any(signal_labels > 1):
      print("Labels too big")
  if np.any(signal_labels < 0):
      print("Labels too small")
  if np.any(snrs <= 0):
      print("BAD SNR")

  if np.any(np.isnan(snrs)):
      print("Found NAN SNR")
      index = np.argwhere(np.isnan(snrs))
      snrs = np.delete(snrs, index, axis=0)
      waveforms = np.delete(waveforms, index, axis=0)
      signal_labels = np.delete(signal_labels, index, axis=0)

  if np.any(np.isnan(waveforms)):
      print("Found NAN WVF")
      index = np.argwhere(np.isnan(waveforms))
      print(index)

  if np.any(np.isnan(signal_labels)):
      print("NAN!!")
      print(np.argwhere(np.isnan(signal_labels)))

  del snrs

  print(f"\t--->Read-in took {time.time() - t0:0.1f} seconds")



  #######################################################
  ############## Renormalizing the data into units of SNR
  #######################################################

  rms_noise = GetRMSNoise(float(band_flow), float(band_fhigh), sampling_rate, 300 * units.kelvin)
  print(f"Scaling all values by 1 / {rms_noise / (1e-6 * units.volt):0.4f} uV to normalize to SNR")
  waveforms /= rms_noise


  #####################################################
  ############## Permuting everything
  #####################################################

  print("Performing initial scramble")
  p_data = np_rng.permutation(len(waveforms))
  waveforms = waveforms[p_data]
  signal_labels = signal_labels[p_data]

  # Make a plot of a waveform with the labels
  #PlotWaveformExample(waveforms[0], signal_labels[0], f"{output_plot_dir}/{base_output}_Labels.pdf")

  #####################################################
  ############## Join the labels across channels
  #####################################################

  print("Joining the label windows")
  signal_labels = np.max(signal_labels, axis=1)
  for i in range(len(signal_labels)):
      ones = np.where(signal_labels[i] > 0)[0]
      if len(ones):
          signal_labels[i, min(ones) : max(ones)] = 1



  ###########################
  ### Setting up data sets
  ###########################

  batch_size = config["training"]["batch_size"]  # Number of "mixtures" of signal/noise
  n_features = config["n_ant"]  # Number of antennas
  wvf_length = config["input_length"]

  mixture = np.linspace(0.0, 1.0, batch_size)  # Percentage of background waveforms in each batch

  # Where to split the dataset into training/validation/testing
  train_fraction = 0.8
  val_fraction = 0.1
  train_split= int(train_fraction * len(waveforms))
  val_split = int(val_fraction * len(waveforms))

  x_train = waveforms[:train_split]
  x_val = waveforms[train_split:train_split + val_split]
  x_test = waveforms[train_split + val_split:]
  y_train = signal_labels[:train_split]
  y_val = signal_labels[train_split:train_split + val_split]
  y_test = signal_labels[train_split + val_split:]

  print(f"Training on {len(x_train)} waveforms")
  print(f"Testing on {len(x_test)} waveforms")
  print(f"Number of signals in test set: {np.sum(y_test)}")
    # For costum dataloader
  mixture = np.linspace(0.0, 1.0, batch_size)
  train_loader = DatasetContinuousStreamStitchless(waveforms=x_train,
                                                   signal_labels=y_train,
                                                  config=config, 
                                                  mixture=mixture, 
                                                  np_rng=np_rng)
  del x_train
  del y_train
  val_loader = DatasetContinuousStreamStitchless(waveforms=x_val,
                                                   signal_labels=y_val,
                                                  config=config, 
                                                  mixture=mixture, 
                                                  np_rng=np_rng)
  del x_val
  del y_val
  test_loader = DatasetContinuousStreamStitchless(waveforms=x_test,
                                                   signal_labels=y_test,
                                                  config=config, 
                                                  mixture=mixture, 
                                                  np_rng=np_rng)
  del x_test
  del y_test

  

  return train_loader, val_loader, test_loader

def prepare_data(x_train, x_val, x_test, y_train, y_val, y_test, batch_size, multi_channel=False):
  """ This methode takes train and test data as numpy arrays and
      return pytorch train and test DataLoader
    Arg: 
      x_train, x_test, y_train, y_test, batch_size
    Return:
      train_loader, val_loader, test_loader, number_of_noise, number_of_signals 

  """
  print(f"Train size = {len(x_train)}")
  if not multi_channel:
    number_of_signals = len(y_train[y_train == 1])
    number_of_noise = len(y_train[y_train == 0])
    print(f"Number of signals = {number_of_signals}")
    print(f"Number of noise = {number_of_noise}")
  print(f"Test size: {len(x_test)}")
  print(f"Max x value: {np.max(x_train)}")
  print(f"Min x value: {np.min(x_train)}")

  # to torch
  x_train = torch.tensor(x_train, dtype=torch.float32)
  x_val =  torch.tensor(x_val, dtype=torch.float32)
  x_test = torch.tensor(x_test, dtype=torch.float32)
  
  y_train = torch.tensor(y_train, dtype=torch.float32)
  y_val =  torch.tensor(y_val, dtype=torch.float32)
  y_test =  torch.tensor(y_test, dtype=torch.float32)

  if not multi_channel:
    x_test = x_test.view(-1, len(x_test[0]), 1)
    x_train = x_train.view(-1, len(x_train[0]), 1)
    x_val = x_val.view(-1, len(x_val[0]), 1)
    y_train = y_train.view(-1, 1)
    y_val = y_val.view(-1, 1)
    y_test = y_test.view(-1, 1)

  if multi_channel:
    x_test = x_test.permute(0, 2, 1)
    x_train = x_train.permute(0, 2, 1)
    x_val = x_val.permute(0, 2, 1)





  train_dataset = TensorDataset(x_train, y_train)
  train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)

  val_dataset = TensorDataset(x_val, y_val)
  val_loader = DataLoader(dataset=val_dataset, batch_size=batch_size, shuffle=True)

  test_dataset = TensorDataset(x_test, y_test)
  test_loader = DataLoader(dataset=test_dataset, batch_size=1, shuffle=True)

  if not multi_channel:
    return train_loader, val_loader, test_loader, number_of_noise, number_of_signals
  else:
    return train_loader, val_loader, test_loader



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




def save_data(config, df, y_pred_data=None):

  path = config['model_path']

  with open(path + 'config.txt', "wb") as fp:
    pickle.dump(config, fp)

  # A easier to read part
  with open(path + 'text_config.txt', 'w') as data:
    for key, value in config.items():
      data.write('%s: %s\n' % (key, value))

  df.to_pickle(path + 'dataframe.pkl')
  if y_pred_data is not None:
    y_pred_data.to_pickle(path + 'y_pred_data.pkl') 
  

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
  else:
    path = path + f'model_{model_number}/' 
    isExist = os.path.exists(path)
    if not isExist:
      os.makedirs(path)
      print("The new directory is created!")
    return path  
  
def find_hyperparameters(model_number, parameter, models_path='/mnt/md0/halin/Models/', ):
  """ This function finds the hyperparameters for a list of models.
      Arg:
        model_number: list of model numbers
        parameter: parameter to find (str) e.g. 'h', 'd_model', 'N'
        models_path: path to models are stored
      Ret:
        hyper_parameters: list of hyperparameters      
      
  """
  hyper_parameters = []

  for i, model in enumerate(model_number):
    with open(models_path + f'model_{model}/config.txt', 'rb') as f:
      config = pickle.load(f)
      hyper_parameters.append(config[parameter])

  return hyper_parameters   

 
def save_example_data(save_path='/home/halin/Master/Transformer/Test/data/'):

  train_loader, val_loader, test_loader = get_data(batch_size=64, 
                                                   seq_len=256, 
                                                   subset=True, 
                                                   data_config_path='/home/halin/Master/Transformer/data_config.yaml')
  del train_loader
  del val_loader

  plot_data = None
  random_index = np.random.randint(0, len(test_loader))
  count = 0
  for batch in test_loader:
    if count == random_index:
      x_data = batch[0]
      y_data = batch[1]
      torch.save(x_data, save_path + 'example_x_data.pt')
      torch.save(y_data, save_path + 'example_y_data.pt')
      break
    count += 1


