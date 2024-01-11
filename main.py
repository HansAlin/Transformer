from os.path import dirname, abspath, join
import sys
#import numpy as np
#from sklearn.preprocessing import StandardScaler
import torch 
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from torch.optim.lr_scheduler import ReduceLROnPlateau
import matplotlib.pyplot as plt
from os.path import dirname, abspath, join
import sys



type(sys.path)
for path in sys.path:
   print(path)

import os
myhost = os.uname()[1]

print("Host name: ", myhost)

import dataHandler.datahandler as dh
import models.models as md
from training.train import training
from plots.plots import plot_collections

# TODO implement loading and saving of model
# TODO Implement train, val, test set use pytorch randomsplit()
# TODO implement unity test?
# TODO read Visualiz... and  Checklist....
# TODO use Pathlib?
# TODO save the optimaizer also
# TODO FLOP's from fvcore.nn import FlopCountAnalysis
# TODO Check final bloch down sizing if it is correct
# TODO Do I have to concern about float 32 and float 64

# TODO Add som ekind of timer
# TODO Scheck if scheduler works!
# TODO secure no model gets overwritten when running a new model


models_path = '/mnt/md0/halin/Models/'
hyper_paramters = [64]
hyper_param_key = 'd_model'
labels = {'hyper_parameters': hyper_paramters, 'name': 'Model size: ({hyper_param_key}})'}
start_model_num = 666
batch_size = 32
epochs = 5
test = True
cuda_device = 2

model_num = start_model_num

configs = []
for i in range(len(hyper_paramters)):

  config = {'model_name': "Attention is all you need",
            'model_type': "base_encoder",
              'model':None,
              'inherit_model': None, # The model to inherit from
              'embed_type': 'basic', # Posible options: 'relu_drop', 'gelu_drop', 'basic'
              'by_pass': False, # If channels are passed separatly through the model
              'pos_enc_type':'Sinusoidal', # Posible options: 'Sinusoidal', 'Relative', 'None', 'Learnable'
              'final_type': 'slim', # Posible options: 'basic', 'slim'
              'loss_function': 'BCEWithLogits', # Posible options: 'BCE', 'BCEWithLogits'
              'model_num': model_num,
              'seq_len': 256,
              'd_model': 64, # Have to be dividable by h
              'd_ff': 64,
              'N': 2,
              'h': 2,
              'output_size': 1,
              'dropout': 0.1,
              'num_epochs': epochs,
              'batch_size': batch_size,
              "learning_rate": 1e-3,
              "decreas_factor": 0.5,
              "num_parms":0,
              "data_path":'',
              "current_epoch":0,
              "global_epoch":0,
              "model_path":'',
              "test_acc":0,
              "early_stop":7,
              "omega": 10000,
              "trained_noise":0,
              "trained_signal":0,
              "data_type": "classic", # Possible options: 'classic', 'chunked'
              "n_ant":4,
              "metric":'Efficiency', # Posible options: 'Accuracy', 'Efficiency', 'Precision'
              "Accuracy":0,
              "Efficiency":0,
              "Precission":0,
              "trained": False

            }
  
  # Copy the config from the model to inherit from
  if config['inherit_model'] != None:
    inherit_model = config['inherit_model']
    old_config = dh.get_model_config(inherit_model)
    config = old_config
    config['model_num'] = model_num
    config['model_path'] = ''
    config['inherit_model'] = inherit_model
    config['num_epochs'] = epochs
    config['Accuracy'] = 0
    config['Efficiency'] = 0
    config['Precission'] = 0
    config['training_time'] = 0

  # Update the hyper parameter
  config[hyper_param_key] = hyper_paramters[i]

  configs.append(config)
  model_num += 1

training(configs=configs, 
         cuda_device=cuda_device,
         batch_size=configs[0]['batch_size'], 
         channels=configs[0]['n_ant'],
         model_folder=models_path,
         test=test,)


