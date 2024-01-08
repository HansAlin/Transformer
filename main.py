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
# Find code directory relative to our directory
# THIS_DIR = dirname(__file__)
# CODE_DIR = abspath(join(THIS_DIR, '..', ''))
# sys.path.append(CODE_DIR)

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
hyper_paramters = [256,512]
labels = {'hyper_parameters': hyper_paramters, 'name': 'Model size: (d_model)'}
start_model_num = 11
batch_size = 32
epochs = 100
test = False

model_num = start_model_num
models = []
configs = []
for i, hyper_paramter in enumerate(hyper_paramters):
  models.append(model_num)
  config = {'model_name': "Attention is all you need",
            'model_type': "base_encoder",
              'model':None,
              'pre_trained': None,
              'embed_type': 'basic', # Posible options: 'relu_drop', 'gelu_drop', 'basic'
              'pos_enc_type':'Sinusoidal', # Posible options: 'Sinusoidal', 'Relative', 'None', 'Learnable'
              'final_type': 'basic',
              'loss_function': 'BCE', # No options yet
              'model_num': model_num,
              'seq_len': 256,
              'd_model': hyper_paramters[i], # Have to be dividable by h
              'd_ff': 64,
              'N': 2,
              'h': 2,
              'dropout': 0.1,
              'num_epochs': epochs,
              'batch_size': batch_size,
              "learning_rate": 1e-3,
              "decreas_factor": 0.5,
              "num_parms":0,
              "data_path":'',
              "current_epoch":0,
              "model_path":'',
              "test_acc":0,
              "early_stop":7,
              "omega": 10000,
              "trained_noise":0,
              "trained_signal":0,
              "n_ant":4,
              "metric":'Efficiency', # Posible options: 'Accuracy', 'Efficiency', 'Precision'
              "Accuracy":0,
              "Efficiency":0,
              "Precission":0,
              "FP":0,
              "FN":0,


            }
  configs.append(config)
  model_num += 1
#'/Test/data/test_100_data.npy'

  PATH = ''  
# /mnt/md0/halin/Models/
training(configs=configs, 
         data_path=PATH, 
         batch_size=configs[0]['batch_size'], 
         channels=configs[0]['n_ant'],
         save_folder=models_path,
         test=test,)


  
  
# if len(hyper_paramters) > 1:
#   plot_collections(models, labels, models_path=models_path, curve='nr', window_pred=False, bins=1000, x_lim=[0,1])
#   plot_collections(models, labels, models_path=models_path, curve='nr', window_pred=True, bins=1000, x_lim=[0,1])  
#   plot_collections(models, labels, models_path=models_path, curve='roc', window_pred=False, bins=1000, x_lim=[0,1])
