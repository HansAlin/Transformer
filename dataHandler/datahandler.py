import argparse
import numpy as np
import torch 
from torch import nn
from torch.utils.data import DataLoader, Dataset, TensorDataset
from torch.utils.data import random_split
from matplotlib import pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.ticker import MultipleLocator
import os
import pandas as pd
from pandas import json_normalize
import pickle
import sys
import time
from tqdm import tqdm
import glob
import yaml
import itertools
import copy



# CODE_DIR_1  ='/home/acoleman/software/NuRadioMC/'
# sys.path.append(CODE_DIR_1)
# # CODE_DIR_2 = '/home/acoleman/work/rno-g/'
# # sys.path.append(CODE_DIR_2)
# # CODE_DIR_3 = '/home/halin/Master/nuradio-analysis/'
# # sys.path.append(CODE_DIR_3)
# CODE_DIR_4 = '/home/halin/Master/nuradio-analysis/'
# sys.path.append(CODE_DIR_4)

import model_configs as mc
import models.models as mm


from NuRadioReco.utilities import units, fft

# Uncomment this if originla data is used
# from analysis_tools import data_locations
# from analysis_tools.config import GetConfig
# from analysis_tools.data_loaders import DatasetContinuousStreamStitchless, DatasetSnapshot
# from analysis_tools.Filters import GetRMSNoise
# from analysis_tools.model_loaders import ConstructModelFromConfig, LoadModelFromConfig



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

def get_data(config, random_seed=123, subset=False, save_test_set=False):
 
  """
    This function loads pre-triggered or chunked data depending on the config file

    Args:
      config: data config file, yaml, needs to contain necessary parameters
      random_seed: random seed
      subset: if True, only one file is loaded
      save_test_set: if True, test set is saved

    Returns:
      train_loader, val_loader, test_loader
  """
  data_type = get_value(config, 'data_type')
  if data_type == 'trigger':
    train_loader, val_loader, test_loader = get_trigger_data(config, subset=subset)
  elif data_type == 'chunked':
    train_loader, val_loader, test_loader = get_chunked_data(config, subset=subset)

  return train_loader, val_loader, test_loader  

def get_trigger_data(config, random_seed=123, subset=False, save_test_set=False):
 
  """ Most of the code is copied from https://github.com/colemanalan/nuradio-analysis/blob/main/trigger-dev/TrainCnn.py
      
      This function loads data from XXXX and returns train, test and validation data.
      It needs a data_config_path to know where to look for parameters for the data

      Arg:
        data_config_path: path to data config file (yaml)
        random_seed: random seed
        batch_size: batch size
        test: if True, only one file is loaded

      Ret:
        train_data, val_data, test_data

  """
  # if 'transformer' in config:
  #   config = config['transformer']
    
  if 'input_length' not in config or 'sampling' not in config or 'training' not in config:
     new_config = get_data_config()
     new_config['training']['batch_size'] = get_value(config, 'batch_size')
     new_config['training']['learning_rate'] = get_value(config, 'learning_rate')
     new_config['input_length'] = get_value(config, 'seq_len')
     new_config['n_ant'] = get_value(config, 'n_ant')
     new_config['transformer']['architecture']['data_type'] = get_value(config, 'data_type')
     config = new_config



 
  wvf_length = config["input_length"]
  sampling_rate = config["sampling"]["rate"]
  band_flow = config["sampling"]["band"]["low"]
  band_fhigh = config["sampling"]["band"]["high"]
  batch_size = config["training"]["batch_size"]
  seq_len = config["input_length"]

  
  np_rng = np.random.default_rng(random_seed)
  use_beam = False
  nFiles = None 
  nu = "*"
  inter = "?c"
  lgE = "1?.??"


  waveform_filenames = data_locations.PreTrigSignalFiles(config=config, nu=nu, inter=inter, lgE=lgE, beam=use_beam) 
  if get_value(config, 'antenna_type') == 'LPDA':
    background_filenames = data_locations.HighLowNoiseFiles("3.421", config=config, nFiles=nFiles)
  elif get_value(config, 'antenna_type') == 'phased':  
     background_filenames = data_locations.PhasedArrayNoiseFiles(config, beam=use_beam)
     
  if subset:
    waveform_filenames = waveform_filenames[:3]
    background_filenames = background_filenames[:3]
     

  for background in background_filenames:
    print(background)

  for waveform in waveform_filenames:
    print(waveform)


  if not len(background_filenames):
    background_filenames = data_locations.PhasedArrayNoiseFiles(config, beam=use_beam)
  if not len(background_filenames):
    print("No background files found!")
    exit()
  print(f"\t\tFound {len(waveform_filenames)} signal files and {len(background_filenames)} background files")

  waveforms = None
  background = None

  frac_into_waveform = config["training"]["start_frac"]  # Trigger location will be put this far into the cut waveform
  trig_bin = config['training']['trigger_time'] * sampling_rate * config["training"]["upsampling"]
  cut_low_bin = max(0, int(trig_bin - wvf_length * frac_into_waveform))
  cut_high_bin = cut_low_bin + wvf_length
  print(f"Cutting waveform sizes to be {wvf_length} bins long, trigger bin: {trig_bin}, bins: {cut_low_bin} to {cut_high_bin}")

  ###################
  ## Signal waveforms
  ###################

  print("\tReading in waveforms")
  t0 = time.time()
  # Precalculate how much space will be needed to read in all waveforms
  print("\t\tPrecalculating RAM requirements")
  total_len = 0
  for filename in tqdm(waveform_filenames):
      shape = np.load(filename, mmap_mode="r")["wvf"].shape
      total_len += shape[0]

  print(f"\t\tSize on disk {(total_len, shape[1], shape[2])}")
  waveforms = np.zeros((total_len, shape[1], wvf_length), dtype=np.float32)
  print(f"\t\tWill load as {waveforms.shape}")

  total_len = 0
  for filename in tqdm(waveform_filenames):
      this_dat = np.load(filename)
      waveforms[total_len : total_len + len(this_dat["wvf"])] = this_dat["wvf"][:, :, cut_low_bin:cut_high_bin]
      total_len += len(this_dat["wvf"])
      del this_dat

  assert len(waveforms)

  upsampling = config.get('upsampling', 1)

  rms_noise = GetRMSNoise(float(band_flow), float(band_fhigh), sampling_rate*upsampling, 300 * units.kelvin)
  print(f"\t\tWill scale waveforms by values by 1 / {rms_noise / (1e-6 * units.volt):0.4f} uV")
  std = np.median(np.std(waveforms[:, :, int(waveforms.shape[-1] * 0.77) :]))
  print(f"\t\tFYI: the RMS noise of waveforms is {std / (1e-6 * units.volt):0.4f} uV")
  waveforms /= rms_noise


  print(
      f"\t\tLoaded signal data of shape {waveforms.shape} --> {waveforms.shape[0] * waveforms.shape[-1] / sampling_rate / units.s:0.3f} s of data"
  )

  nan_check = np.isnan(waveforms)
  if np.any(nan_check):
      print("Found NAN WVF")
      index = np.argwhere(nan_check)
      print(np.unique(index[:, 0])) 
      print(index)
      exit()

    
  #######################
  ## Background waveforms
  #######################

  print("\tReading in background")

  print("\t\tPrecalculating RAM requirements")
  total_len = 0
  for filename in tqdm(background_filenames):
      back_shape = np.load(filename, mmap_mode="r").shape
      total_len += back_shape[0]

  print(f"\t\tShape on disk is {(total_len, back_shape[1], back_shape[2])}")
  back_shape = (total_len, back_shape[1], wvf_length)
  print(f"\t\tWill load as {back_shape}")
  background = np.zeros(back_shape, dtype=np.float32)

  total_len = 0
  for filename in tqdm(background_filenames):
      this_dat = np.load(filename)
      background[total_len : total_len + len(this_dat)] = this_dat[:, :, cut_low_bin:cut_high_bin]
      total_len += len(this_dat)
      del this_dat


  assert total_len == len(background)
  assert background.shape[1] == waveforms.shape[1]
  print(
      f"\t\tLoaded background data of shape {background.shape} --> {background.shape[0] * background.shape[-1] / sampling_rate / units.s:0.3f} s of data"
  )
  print(f"\t--->Read-in took {time.time() - t0:0.1f} seconds")
      
  #######################################################
  ############## Renormalizing the data into units of SNR
  #######################################################

  std = np.std(background)
  print(f"Will scale backround by 1 / {rms_noise / (1e-6 * units.volt):0.4f} uV")
  print(f"FYI: the RMS noise of the backround is {std / (1e-6 * units.volt):0.4f} uV")
  background /= rms_noise
  new_std = np.std(background)
  print(f"New std: {new_std}")

  #####################################################
  ############## Permuting everything
  #####################################################

  print("Performing initial scramble")
  p_data = np_rng.permutation(len(waveforms))
  waveforms = waveforms[p_data]
  p_background = np_rng.permutation(len(background))
  background = background[p_background]

  #plot_examples(background, waveforms, sampling_rate, config)

  ###########################
  ### Setting up data sets
  ###########################


  train_fraction = 1 - config['training']['test_frac'] - config['training']['val_frac']  # Fraction of waveforms to use for testing

  n_antennas = waveforms.shape[1]

  # Where to split the dataset into training/validation/testing
  wwf_split_index = int(train_fraction * len(waveforms))
  background_split_index = int(train_fraction * len(background))
  val_wwf_split_index = int((train_fraction + config['training']['test_frac'])*len(waveforms))
  val_background_split_index = int((train_fraction + config['training']['test_frac'])*len(background))
  if subset:
    wwf_split_index = config['training']['batch_size']*2
    background_split_index = config['training']['batch_size']*2
    val_wwf_split_index = config['training']['batch_size']*2 + wwf_split_index
    val_background_split_index = config['training']['batch_size']*2 + background_split_index

  if save_test_set:
    save_path = '/home/halin/Master/Transformer/Test/data/'
    print(f"Saving test set to {save_path}")
    np.save(save_path + f'test_set_waves_binary_class_b_{batch_size}_s_{seq_len}.npy', waveforms[:200])
    np.save(save_path + f'test_set_bkgd_binary_class_b_{batch_size}_s_{seq_len}.npy', background[:200])
    

  train_data = DatasetSnapshot(
      waveforms=waveforms[:wwf_split_index],
      backgrounds=background[:background_split_index],
      n_features=n_antennas,
      batch_size=batch_size,
      np_rng=np_rng,
  ) 
  val_data = DatasetSnapshot(
      waveforms=waveforms[wwf_split_index:val_wwf_split_index],
      backgrounds=background[background_split_index:val_background_split_index],
      n_features=n_antennas,
      batch_size=batch_size,
      np_rng=np_rng,
  )
  if subset:
    backgrounds = background[val_background_split_index:val_background_split_index + background_split_index]
    waveforms = waveforms[val_wwf_split_index:val_wwf_split_index + wwf_split_index]
  else:
    backgrounds = background[val_background_split_index:]
    waveforms = waveforms[val_wwf_split_index:]
       
  test_data = DatasetSnapshot(
      waveforms=waveforms,
      backgrounds=backgrounds,
      n_features=n_antennas,
      batch_size=batch_size,
      np_rng=np_rng,
  )
  val_data.scramble_warn = False
  test_data.scramble_warn = False
  train_data.Scramble()
  print("Data sets are of size:")
  print("\tTraining:", len(waveforms[:wwf_split_index]))
  print("\tValidation:", len(waveforms[wwf_split_index:val_wwf_split_index]))
  print("\tTesting:", len(waveforms[val_wwf_split_index:]))

  del waveforms

  return train_data, val_data, test_data

def get_test_data(path='/home/halin/Master/Transformer/Test/data/', n_antennas=4, batch_size=32, seq_len=256, random_seed=123):
  """ This method loads test data from folder
    Arg:
      path: wher data is saved
    Ret:
      x_train, x_test, x_val, y_train, y_val, y_test

  """
  print("Loading test data...")
  
  if path == '':
    path = os.getcwd()
    path = path + '/Test/data/' 

  waves = np.load(path + f'test_set_waves_binary_class_b_{batch_size}_s_{seq_len}.npy' )
  background = np.load(path + f'test_set_bkgd_binary_class_b_{batch_size}_s_{seq_len}.npy' )
  np_rng = np.random.default_rng(random_seed)

  train_data = DatasetSnapshot(
      waveforms=waves[:160],
      backgrounds=background[:160],
      n_features=n_antennas,
      batch_size=batch_size,
      np_rng=np_rng,
  ) 
  val_data = DatasetSnapshot(
      waveforms=waves[160:180],
      backgrounds=background[160:180],
      n_features=n_antennas,
      batch_size=batch_size,
      np_rng=np_rng,
  )
  test_data = DatasetSnapshot(
      waveforms=waves[180:],
      backgrounds=background[180:],
      n_features=n_antennas,
      batch_size=batch_size,
      np_rng=np_rng,
  )

  return train_data, val_data, test_data


# TODO uncomment this before push
# def get_any_data(config, subset=False):
   
#   batch_size = config['training']['batch_size']
#   feat = config['n_ant']
#   seq_len = config['input_length']
#   test_frac = config['training']['test_frac']
#   val_frac = config['training']['val_frac']

#   # Number of samples per class
#   n_samples = 100000
#   if subset:
#     n_samples = 1000

#   # Generate 2D data points for two clusters
#   X1 = torch.randn(n_samples, seq_len, feat) + 0.05  # Cluster 1: centered at (1, 1)
#   X2 = torch.randn(n_samples, seq_len, feat) - 0.05  # Cluster 2: centered at (-1, -1)

#   # Concatenate the data points
#   X = torch.cat([X1, X2], dim=0)

#   # Generate labels
#   y1 = torch.zeros(n_samples, dtype=torch.long)  # Labels for cluster 1
#   y2 = torch.ones(n_samples, dtype=torch.long)   # Labels for cluster 2



#   # Concatenate the labels
#   y = torch.cat([y1, y2])

#   # Create a DataLoader for our data
#   dataset = TensorDataset(X, y)

#   train_set = int((1 - test_frac - val_frac) * len(dataset))
#   val_set = int(val_frac * len(dataset))
#   test_set = len(dataset) - train_set - val_set

#   train_data, val_data, test_data = random_split(dataset, [train_set, val_set, test_set])

#   train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
#   val_loader = DataLoader(val_data, batch_size=batch_size, shuffle=True)
#   test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=True)
  
#   print(f"Data shape: {train_loader.dataset.dataset.tensors[0][train_loader.dataset.indices].shape}")
 

#   return train_loader, val_loader, test_loader

# TODO comment this out before push
def augment_data(data, num_copies):
    # Calculate the mean and standard deviation of each sample in the data
    mean = data.mean(dim=1, keepdim=True)
    std = data.std(dim=1, keepdim=True)

    # Repeat each sample in the data and its mean and std num_copies times
    repeat_sizes = (num_copies,) + (1,) * (data.dim() - 1)
    data_repeated = data.repeat(*repeat_sizes)
    mean_repeated = mean.repeat(*repeat_sizes)
    std_repeated = std.repeat(*repeat_sizes)

    # Generate random noise
    noise = torch.randn_like(data_repeated) * std_repeated + mean_repeated

    # Add the noise to the repeated data
    synthetic_data = data_repeated + noise

    return synthetic_data

# TODO comment this out before push
def get_any_data(config, subset=False):

  batch_size = config['training']['batch_size']
  feat = config['n_ant']
  seq_len = config['input_length']
  test_frac = config['training']['test_frac']
  val_frac = config['training']['val_frac']
   
  data = np.load('/home/halin/Master/Transformer/Test/data/aurora_data.npz', allow_pickle=True)   

  aurora_data = torch.tensor(data['aurora']).float()
  background_data = torch.tensor(data['background']).float()

  # Augment the aurora_data
  nr_original_samples = len(aurora_data)
  nr_background_samples = len(background_data)

  num_copies = nr_background_samples // nr_original_samples *10
  synthetic_aurora_data = augment_data(aurora_data, num_copies)
  synthetic_background_data = augment_data(background_data, 10)

  # Concatenate the original and synthetic data
  augmented_aurora_data = torch.cat([aurora_data, synthetic_aurora_data], dim=0)

  # Co
  augmented_background_data = torch.cat([background_data, synthetic_background_data], dim=0)  

  concat_data = torch.cat([augmented_aurora_data, augmented_background_data], dim=0)
  y_label = torch.ones(len(concat_data))
  y_label[len(augmented_aurora_data):] = 0

  print(f"Number of aurora samples: {len(augmented_aurora_data)}")
  print(f"Number of background samples: {len(augmented_background_data)}")

  shuffled_indices = torch.randperm(len(concat_data))

  concat_data = concat_data[shuffled_indices]
  y_label = y_label[shuffled_indices]

  dataset = TensorDataset(concat_data, y_label)

  train_set = int((1 - test_frac - val_frac) * len(dataset))
  val_set = int(val_frac * len(dataset))
  test_set = len(dataset) - train_set - val_set

  train_data, val_data, test_data = random_split(dataset, [train_set, val_set, test_set])

  train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
  val_loader = DataLoader(val_data, batch_size=batch_size, shuffle=True)
  test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=True)

  return train_loader, val_loader, test_loader





# def plot_examples(background,waveforms,sampling_rate, config, output_plot_dir='/home/halin/Master/Transformer/Test/ModelsResults/test'):
#   n_events = 3
#   n_channels = background.shape[1]
#   ncols = 3
#   nrows = n_events * n_channels
#   fig, ax = plt.subplots(ncols=ncols, nrows=nrows, figsize=(ncols * 8, nrows * 3), gridspec_kw={"wspace": 0.2, "hspace": 0.0})

#   for ievent in range(n_events):
#       for ich in range(n_channels):
#           ax[ievent * n_channels + ich, 0].plot(waveforms[ievent, ich], label=f"Signal, evt{ievent+1}, ch{ich+1}")
#           ax[ievent * n_channels + ich, 1].plot(background[ievent, ich], label=f"Bkgd, evt{ievent+1}, ch{ich+1}")
#           ax[ievent * n_channels + ich, 0].legend()
#           ax[ievent * n_channels + ich, 1].legend()

#       spec = np.median(np.abs(fft.time2freq(waveforms[ievent], sampling_rate * config["training"]["upsampling"])), axis=0)
#       ax[ievent * n_channels, 2].plot(spec, color="k")
#       spec = np.median(np.abs(fft.time2freq(background[ievent], sampling_rate * config["training"]["upsampling"])), axis=0)
#       ax[ievent * n_channels, 2].plot(spec, color="r")
#       ax[ievent * n_channels, 2].set_yscale("log")


#   for i in range(len(ax)):
#       for j in range(len(ax[i])):
#           ax[i, j].tick_params(axis="both", which="both", direction="in")
#           ax[i, j].yaxis.set_ticks_position("both")
#           ax[i, j].xaxis.set_ticks_position("both")

#   example_plot_name = f"{output_plot_dir}/ExampleWaveforms.pdf"
#   print("Saving", example_plot_name)
#   fig.savefig(example_plot_name, bbox_inches="tight")
#   plt.close()

def get_chunked_data(config, subset=True):
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


  batch_size = config["training"]["batch_size"] 
  sampling_rate = config["sampling"]["rate"]
 


  random_seed = 123
  np_rng = np.random.default_rng(random_seed)


  #####################################################
  ############## Load in the data
  #####################################################

  print("READING IN DATA...")
  waveform_filenames = data_locations.NoiselessSigFiles(
      cdf=config["training"]["cdf"], config=config, nu="*", inter="cc", lgE="1?.??"
  )
  print(f"\tFound {len(waveform_filenames)} signal files")


  waveforms = None
  signal_labels = None

  if subset:
    waveform_filenames = waveform_filenames[:4]
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

  """
  rms_noise = GetRMSNoise(float(band_flow), float(band_fhigh), sampling_rate, 300 * units.kelvin)
  print(f"Scaling all values by 1 / {rms_noise / (1e-6 * units.volt):0.4f} uV to normalize to SNR")
  waveforms /= rms_noise
  """


  #####################################################
  ############## Join the labels across channels
  #####################################################

  print("Joining the label windows")

  ## only do this for LPDAs

  signal_labels = np.max(signal_labels, axis=1)

  ## delete *wrong* waveforms that overlap with boundary
  index=np.where( (signal_labels[:,0]>0) | (signal_labels[:,-1]>0))[0]
  signal_labels=np.delete(signal_labels, index, axis=0)
  waveforms=np.delete(waveforms, index, axis=0)

  index=np.where( signal_labels.sum(axis=1)==0)[0]

  if(len(index)>0):

      signal_labels=np.delete(signal_labels, index, axis=0)
      waveforms=np.delete(waveforms, index, axis=0)

  if(config["sampling"]["band"]["low"]==0.08):
    print("------ MAKING SURE LABELS FOR TRAINING ARE ALWAYS A SINGLE GROUP - ONLY FOR LPDA!!! -----")
    ## only merge for LPDA.. not really needed probably, just to make sure
    for i in range(len(signal_labels)):
        ones = np.where(signal_labels[i] > 0)[0]
        if len(ones):
            signal_labels[i, min(ones) : max(ones)] = 1

  print("Performing initial scramble")
  p_data = np_rng.permutation(len(waveforms))
  waveforms = waveforms[p_data]
  signal_labels = signal_labels[p_data]

  ###########################
  ### Setting up data sets
  ###########################

  batch_size = config["training"]["batch_size"]  # Number of "mixtures" of signal/noise
  n_features = config["n_ant"]  # Number of antennas
  wvf_length = config["input_length"]

  mixture = np.linspace(0.0, 1.0, batch_size)  # Percentage of background waveforms in each batch
  mixture[:]=1.0
  mixture[0]=0.0

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

  np_rng = np.random.default_rng(random_seed)
  
  train_loader = DatasetContinuousStreamStitchless(
      waveforms = x_train,
      signal_labels = y_train,
      config = config, 
      mixture = mixture, 
      np_rng = np_rng,
      permute_for_RNN_input=False, 
      scale_output_by_expected_rms=True,
      noise_relative_amplitude_scaling=1.0,
      randomize_batchitem_start_position=config["training"]["randomize_batchitem_start_position"],
      shift_signal_region_away_from_boundaries=config["training"]["shift_signal_region_away_from_boundaries"],
      set_spurious_signal_to_zero=config["training"]["set_spurious_signal_to_zero"],
      extra_gap_per_waveform=config["training"]["extra_gap_per_waveform"],
      probabilistic_sampling_ensure_signal_region=config["training"]["probabilistic_sampling_ensure_signal_region"],
      probabilistic_sampling_oversampling=config["training"]["probabilistic_sampling_oversampling"],
      probabilistic_sampling_ensure_min_signal_fraction=config["training"]["probabilistic_sampling_ensure_min_signal_fraction"],
      probabilistic_sampling_ensure_min_signal_num_bins=config["training"]["probabilistic_sampling_ensure_min_signal_num_bins"],
      test_data=subset
      )
  
  del x_train
  del y_train

  np_rng_validation = np.random.default_rng(random_seed + 1)

  val_mixture = np.linspace(0.0, 1.0, batch_size)

  val_loader = DatasetContinuousStreamStitchless(
      waveforms = x_val,
      signal_labels = y_val,
      config = config, 
      mixture = val_mixture, 
      np_rng = np_rng_validation,
      permute_for_RNN_input=False, 
      scale_output_by_expected_rms=True,
      noise_relative_amplitude_scaling=1.0,
      randomize_batchitem_start_position=config["training"]["randomize_batchitem_start_position"],
      shift_signal_region_away_from_boundaries=config["training"]["shift_signal_region_away_from_boundaries"],
      set_spurious_signal_to_zero=config["training"]["set_spurious_signal_to_zero"],
      extra_gap_per_waveform=config["training"]["extra_gap_per_waveform"],
      probabilistic_sampling_ensure_signal_region=config["training"]["probabilistic_sampling_ensure_signal_region"],
      probabilistic_sampling_oversampling=config["training"]["probabilistic_sampling_oversampling"],
      probabilistic_sampling_ensure_min_signal_fraction=config["training"]["probabilistic_sampling_ensure_min_signal_fraction"],
      probabilistic_sampling_ensure_min_signal_num_bins=config["training"]["probabilistic_sampling_ensure_min_signal_num_bins"],
      test_data=subset
      )
  
  del x_val
  del y_val

  np_rng_test = np.random.default_rng(random_seed + 2)

  test_mixture = np.linspace(0.0, 1.0, batch_size)

  test_loader = DatasetContinuousStreamStitchless(
        waveforms=x_test,
        signal_labels=y_test,
        config=config, 
        mixture=test_mixture, 
        np_rng=np_rng_test,
        permute_for_RNN_input=False, 
        scale_output_by_expected_rms=True,
        noise_relative_amplitude_scaling=1.0,
        randomize_batchitem_start_position=config["training"]["randomize_batchitem_start_position"],
        shift_signal_region_away_from_boundaries=config["training"]["shift_signal_region_away_from_boundaries"],
        set_spurious_signal_to_zero=config["training"]["set_spurious_signal_to_zero"],
        extra_gap_per_waveform=config["training"]["extra_gap_per_waveform"],
        probabilistic_sampling_ensure_signal_region=config["training"]["probabilistic_sampling_ensure_signal_region"],
        probabilistic_sampling_oversampling=config["training"]["probabilistic_sampling_oversampling"],
        probabilistic_sampling_ensure_min_signal_fraction=config["training"]["probabilistic_sampling_ensure_min_signal_fraction"],
        probabilistic_sampling_ensure_min_signal_num_bins=config["training"]["probabilistic_sampling_ensure_min_signal_num_bins"],
        test_data=subset
        )
  
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

def get_data_config(data_config_path = '/home/halin/Master/Transformer/trigger_data_config.yaml'):
  """ This function reads the data config file and returns the data config dictionary
      Arg:
        data_config_path: path to data config file (yaml)
      Ret:
        data_config: dictionary, the data config dictionary
  """
  with open(data_config_path, 'r') as file:
    config = yaml.safe_load(file)
  return config

def standardScaler(x):

  mean = x.mean(0, keepdim=True)
  std = x.std(0, unbiased=False, keepdim=True)
  x -= mean
  x /= std
  return x

def save_model(trained_model, optimizer, scheduler, config, global_step, text, threshold):

  path = config['transformer']['basic']['model_path']

  saved_model_path = path + f'/saved_model'
  isExist = os.path.exists(saved_model_path)
  if not isExist:
    os.makedirs(saved_model_path)
    print("The new directory is created!")    
  torch.save({
              'epoch': global_step,
              'model_state_dict': trained_model.state_dict(), 
              'optimizer_state_dict': optimizer.state_dict(),
              'scheduler_state_dict': scheduler.state_dict(),
              'global_step': global_step,
              'threshold': threshold
              },
              saved_model_path + f'/model_{config["transformer"]["basic"]["model_num"]}_{text}.pth')

def file_exist(directory, filename):
   
  filepath = os.path.join(directory, filename)

  if os.path.isfile(filepath):
    print(f"The file '{filename}' exists in the directory '{directory}'.")
    return True
  else:
    print(f"The file '{filename}' does not exist in the directory '{directory}'.")
    return False

def find_file(directory, extension):
   
  files = glob.glob(os.path.join(directory, f'*.{extension}'))

  for file in files:
    print(file)
    return file
   
  return None



def get_same_diff_df(df):
    # Create two empty dataframes
  df_same = pd.DataFrame()
  df_diff = pd.DataFrame()

  # Iterate over each column
  for col in df.columns:
      # If the column has only one unique value, add it to df_same
      if df[col].nunique() == 1:
          df_same[col] = df[col]
      # If the column has more than one unique value, add it to df_diff
      else:
          df_diff[col] = df[col]
  return df_same, df_diff   

def get_model_path(config, text=''):
  """
  Returns the path to the model file. If a specific text is given, the function
  looks for a file containing that text. E.g. if text='early_stop', the function
  will return the path to the file containing 'early_stop' in its name. If no
  specific text is given or no file containing the text is found, the function
  returns the first file with the '.pth' extension. If no '.pth' file is found,
  the function returns None.

  Args:
    config: dictionary, the configuration dictionary
    text: string, the text to look for in the file name

  Returns:
    string, the path to the model file
  
  """
  if 'transformer' in config:
    config = config['transformer']
  if '/mnt/md0/halin/Models' not in config['basic']['model_path']:  
    folder = config['basic']['model_path'] + 'saved_model/'
  else:
    folder = config['basic']['model_path'] + 'saved_model/'

  # List all files in the given folder
    
  files = os.listdir(folder)

  # If a specific text is given, look for a file containing that text
  if text != '':
      for file in files:
          if text in file:
              return os.path.join(folder, file)

  # If no specific text is given or no file containing the text is found,
  # return the first file with the '.pth' extension
  for file in files:
      if file.endswith('.pth'):
          return os.path.join(folder, file)

  # If no '.pth' file is found, return None
  return None

def save_data(config, df=None, y_pred_data=None, text=''):

  if text != '':
    text = '_' + text


  if '/home/halin/Master/nuradio-analysis/data/models/fLow_0.08-fhigh_0.23-rate_0.5/' in config['transformer']['basic']['model_path']:
    path = f"/home/halin/Master/nuradio-analysis/configs/chunked/config_{config['transformer']['basic']['model_num']}.yaml"
    with open(path, 'w') as data:
      yaml.dump(config, data, default_flow_style=False) 
  elif '/home/halin/Master/nuradio-analysis/data/models/fLow_0.096-fhigh_0.22-rate_0.5' in config['transformer']['basic']['model_path']:
    path = f"/home/halin/Master/nuradio-analysis/configs/chunked/config_{config['transformer']['basic']['model_num']}.yaml"
    with open(path, 'w') as data:
      yaml.dump(config, data, default_flow_style=False)
                  
  else:
    path = config['transformer']['basic']['model_path']
    with open(path + f'config{text}.yaml', 'w') as data:
      yaml.dump(config, data, default_flow_style=False)  


  if df is not None:  
    df.to_pickle(path + f'dataframe{text}.pkl')

  if y_pred_data is not None:
    y_pred_data.to_pickle(path + f'y_pred_data{text}.pkl') 
  

def create_model_folder(model_number, path=''):
  """
  This function create a folder to save all model related stuf in.
  """

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
      hyper_parameters.append(config['architecture'][parameter])

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

def get_model_config(model_num, path='/mnt/md0/halin/Models/', type_of_file='yaml', sufix=''):
    
    """
    This function loads the model configuration from a file. The file can be either a .txt or a .yaml file.

    Args:
      model_num:    model number
      path:         path to the models (optional)
      type_of_file: type of file to load the configuration from (optional)
      sufix:        sufix to add to the file name (optional)

    Returns:
      config: dictionary, the model configuration
        
    """
    try:
      if 'yaml' in path:
        CONFIG_PATH = path 
        with open(CONFIG_PATH, 'r') as file:
            config = yaml.safe_load(file)
      elif type_of_file == 'txt':  
        CONFIG_PATH = path + f'model_{model_num}/config{sufix}.txt'
        with open(CONFIG_PATH, 'rb') as f:
            config = pickle.load(f)
      else:
        CONFIG_PATH = path + f'model_{model_num}/config{sufix}.yaml'
        with open(CONFIG_PATH, 'r') as file:
            config = yaml.safe_load(file)

      if 'transformer' not in config:
        return {'transformer': config}
      else:
        return config
    except:
      CONFIG_PATH = '/home/halin/Master/nuradio-analysis/configs/chunked/' + f'config_{model_num}.yaml'
      with open(CONFIG_PATH, 'r') as file:
        config = yaml.safe_load(file)
      return config  



def modify_keys(dictionary):
    new_dict = {}
    for key, value in dictionary.items():
        if '.' in key:
            new_key = key.split('.')[-1]
            new_dict[new_key] = value
        else:
            new_dict[key] = value
    return new_dict

def collect_config_to_df(model_numbers, model_path='/home/hansalin/mnt/Models/', save_path='', save=False, type_of_file='yaml'):
  df = pd.DataFrame()
  counter = 0
  for model_num in model_numbers:
    config = get_model_config(model_num, model_path, type_of_file=type_of_file)

    if 'transformer' in config:
      config = config['transformer']

    flatt_dict = pd.json_normalize(config)
    flatt_dict = modify_keys(flatt_dict)
    flatt_dict_df = pd.DataFrame(flatt_dict, index=[0])
    df = pd.concat([df, flatt_dict_df])
    header_order = ['model_num', 
                'embed_type', 
                'pos_enc_type', 
                'max_relative_position', 
                'encoder_type',
                'final_type',
                'location',
                'normalization',
                'activation',
                'N',
                'd_model',
                'd_ff',
                'h',
                'n_ant',
                'seq_len',
                'batch_size',
                'loss_function',
                'learning_rate',
                'step_size',
                'decreas_factor',
                'dropout',
                'num_param',
                'input_param',
                'pos_param',
                'encoder_param',
                'final_param',
                'NSE_AT_10KNRF',
                'MACs'


                ]
    df = df.reindex(columns=header_order)

    if save:
      if save_path == '':
        df.to_pickle(model_path + 'collections/' + 'dataframe.pkl')
      else:
        df.to_pickle(save_path + 'dataframe.pkl')  
    counter += 1
  return df  




def get_predictions(model_number, model_path='/mnt/md0/halin/Models/'):
  """ This function loads the predictions from a model
      Arg:
        model_number: model number
        model_path: path to model (optional)
      Return: y, y_pred
        y: true labels
        y_pred: predicted labels
       
  """
  path = model_path + f'model_{model_number}/'
  y_pred_data = pd.read_pickle(path + 'y_pred_data.pkl')
  y = y_pred_data['y'].to_numpy()
  y_pred = y_pred_data['y_pred'].to_numpy()
  return y, y_pred

def get_value(dictionary, key):
  """
  This function returns the value of a key in a dictionary. If the key is not found, the function returns None.

  Args:
    dictionary: dictionary, the dictionary to search
    key: string, the key to search for

  Returns:
    value: the value of the key in the dictionary
  """
  if key in dictionary:
      return dictionary[key]
  for k, value in dictionary.items():
      if isinstance(value, dict):
          result = get_value(value, key)
          if result is not None:
              return result
  return None

def update_value(dictionary, key, value):
  """
  This function updates the value of a key in a dictionary. If the key is not found, the function returns the original dictionary.

  Args:
    dictionary: dictionary, the dictionary to update
    key: string, the key to update
    value: the new value of the key

  Returns:
    dictionary: dictionary, the updated dictionary
  """
  if key in dictionary:
      dictionary[key] = value
      return dictionary
  for k, v in dictionary.items():
      if isinstance(v, dict):
          result = update_value(v, key, value)
          if result is not None:
              dictionary[k] = result
              return dictionary
  return None

def update_nested_dict(d, key, value):
    """
    This function updates the value of a key in a nested dictionary.
    Args:
      d: dictionary, the dictionary to update
      key: string, the key to update
      value: the new value of the key
    """
    for k, v in d.items():
        if isinstance(v, dict):
            update_nested_dict(v, key, value)
        if k == key:
            d[k] = value

  
def config_production(base_config_number, 
                      test_dict,
                      cnn_configs=[{'kernel_size': 3, 'stride': 1}],
                          vit_configs = [{'kernel_size': 3, 'stride': 3}],
                          alt_combination='combi',
                          subset=None
                        ):
    """
        This function creates a list of configs based on a base config and a test dictionary. The
        test config should contain the keys that can be varied in the base config. 

        Args:
          base_config_number: int, 0 gives the default config from file every other value would try to 
                              load the config from the model with that number
          test_dict: dictionary, the dictionary containing the keys that can be varied                    
                     Example: {'embed_type': ['linear', 'cnn', 'ViT']}
          cnn_configs: list of cnn configurations valid for the other hyper parameters.
          vit_configs: list of ViT configurations valid for the other hyper parameters.
          alt_combination: option to choose between single = one item from each, 
                                                    combi  = all combinations
                                                    restrict = only combinations that satisfy FLOPs constraints
          subset: int, the number of configurations to produce                                          

        Returns:
          configs: list of dictionaries, the list of configurations                                 
    """

    try:
      config = mc.config.get_config(base_config_number)
    except:
      print(f'Loding config from model {base_config_number}')
      config = get_model_config(base_config_number)
      remove_old_values(config)

    if alt_combination == 'single':
      combinations = list(zip(*test_dict.values()))
    elif alt_combination == 'combi':
      combinations = list(itertools.product(*test_dict.values()))
    elif alt_combination == 'restrict':
       combinations = configs_with_flops_constarints(test_dict=test_dict, 
                                                        flop_limit=3e6,
                                                        config_number=base_config_number) 
    configs = []

    if subset is not None:
      combinations = combinations[:subset]

    for combination in combinations:
      params = dict(zip(test_dict.keys(), combination))

      if params.get('embed_type') == 'linear' and params.get('max_pool') == True:
          continue

      if params.get('embed_type') == 'cnn':

          if config['n_ant'] == 5 or params.get('n_ant') == 5:
              if params.get('d_model') != None:
                  if params.get('d_model') % 5 == 0:
                      pass
                  elif config['transformer']['architecture']['d_model']  % 5 == 0:
                      pass
              elif config['transformer']['architecture']['d_model']  % 5 == 0:
                  pass    
              else:
                  assert False, "d_model must be divisible by 5"

          for cnn_config in cnn_configs:
              params_copy = params.copy()
              params_copy.update(cnn_config)
              params.update(cnn_config)
              for (test_key, value) in params_copy.items():
                  update_nested_dict(config, test_key, value)
              config['transformer']['architecture']['max_relative_position'] = config['transformer']['architecture']['max_relative_position'] // params['stride']    
              if params.get('max_pool') == True:
                config['transformer']['architecture']['max_relative_position'] = config['transformer']['architecture']['max_relative_position'] // 2    
              new_config = copy.deepcopy(config)
              configs.append(new_config)


      elif params.get('embed_type') == 'ViT':

          for vit_config in vit_configs:
              params_copy = params.copy()
              params_copy.update(vit_config)
              params.update(vit_config)
              for (test_key, value) in params_copy.items():
                  update_nested_dict(config, test_key, value)
              config['transformer']['architecture']['max_relative_position'] = config['transformer']['architecture']['max_relative_position'] // params['stride']    
              if params.get('max_pool') == True:
                config['transformer']['architecture']['max_relative_position'] = config['transformer']['architecture']['max_relative_position'] // 2    

              new_config = copy.deepcopy(config)
              configs.append(new_config)
   


      else:
          for (test_key, value) in params.items():
              update_nested_dict(config, test_key, value)
          if params.get('max_pool') == True:
            config['transformer']['architecture']['max_relative_position'] = config['transformer']['architecture']['max_relative_position'] // 2    

          new_config = copy.deepcopy(config)
          configs.append(new_config)



    return configs


def remove_old_values(config):
    """
    This function removes the values obtained during training, like parameters and results
    """
    for key_1, value_1 in config['transformer']['num of parameters'].items():
       
       config['transformer']['num of parameters'][key_1] = 0

    for key_2, value_2 in config['transformer']['results'].items():

      config['transformer']['results'][key_2] = 0   
        

def configs_with_flops_constarints(test_dict, flop_limit, config_number):
  """
    This function returns a list of posible hyperparameters that satisfy the FLOPs constraints.

    Args:
      test_dict: dictionary, the dictionary containing the keys that can be varied                    
                 Example: {
                'd_model':[16,2,128],
                'h':[4],
                'd_ff':[8,8,256],
                'N':[1,2,3,4],
                }
      flop_limit: float, the FLOPs limit
      config_number: int, the base config number, 0 gives the default config

    Returns:
      hyper_param: list of tuples, the list of hyperparameters that satisfy the FLOPs constraints given the base config
  """
  configs = config_production(base_config_number=config_number,
                               test_dict=test_dict)

 
  hyper_param = []

  count = 0
  for config in configs:
      remove_old_values(config)
      model = mm.build_encoder_transformer(config['transformer'])
      flops = mm.get_FLOPs(model, config)
      
      if flop_limit*0.9 < flops < flop_limit*1.1:
          print(f"Flops: {flops/1e6:>.2f} M, N: {config['transformer']['architecture']['N']}, d_model: {config['transformer']['architecture']['d_model']:>3}, d_ff: {config['transformer']['architecture']['d_ff']:>3}, h: {config['transformer']['architecture']['h']:>3}")
          hyper_param.append((get_value(config, 'd_model'), get_value(config, 'h'), get_value(config, 'd_ff'), get_value(config, 'N') ))
          count += 1
  print(f"Number of models: {count}")

  return hyper_param
   