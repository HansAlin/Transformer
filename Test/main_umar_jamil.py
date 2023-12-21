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
THIS_DIR = dirname(__file__)
CODE_DIR = abspath(join(THIS_DIR, '..', ''))
sys.path.append(CODE_DIR)

type(sys.path)
for path in sys.path:
   print(path)

import os
myhost = os.uname()[1]

print("Host name: ", myhost)

import dataHandler.datahandler as dh
import models.models_1 as md
from training.train import training
from plots.plots import plot_collections

# TODO implement loading and saving of model
# TODO Implement train, val, test set use pytorch randomsplit()
# TODO implement unity test?
# TODO read Visualiz... and  Checklist....
# TODO use Pathlib?
# TODO save the optimaizer also
# TODO Verify that the validation is correctly implemented according to batch size 
# TODO implement the some thing better for the last block loch at the time series
# TODO change the size in the posstional encoder
# TODO FLOP's from fvcore.nn import FlopCountAnalysis
# TODO Check final bloch down sizing if it is correct
# TODO Do I have to concern about float 32 and float 64
# TODO Make a total plot for noise reduction factor
# TODO Add som ekind of timer
# TODO Scheck if scheduler works!
# '/mnt/md0/halin/Models'
# Hyper paramters:
#     learning rate 
#         ¤  learning rate functions  
#          ¤  facor 
#          ¤  based on val_loss or val_acc ???  


hyper_paramters = [64, 256, 512]
labels = {'hyper_parameters': hyper_paramters, 'name': 'Model Size (d_model)'}
start_model_num = 993
batch_size = 64
epochs = 100
test = False

model_num = start_model_num
models = []
configs = []
for i, hyper_paramter in enumerate(hyper_paramters):
  models.append(model_num)
  config = {'model_name': "with_out_activation_in_final_block",
            'model_type': "base_encoder",
              'model':None,
              'pre_trained': None,
              'pos_enc_type':'Sinusoidal', # Posible options: 'Sinusoidal', 'Relative', 'None', 'Learnable'
              'model_num': model_num,
              'embed_size': 64,
              'seq_len': 1000,
              'd_model': hyper_paramters[i],
              'd_ff': 512,
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
              "metric":'Accuracy', # Posible options: 'Accuracy', 'Efficiency', 'Precision'
              "Accuracy":0,
              "Efficiency":0,
              "Precission":0,
              "FP":0,
              "FN":0,


            }
  configs.append(config)
  model_num += 1
if test:
  PATH = os.getcwd() + '/Test/data/test_100_data.npy'
else:
  PATH = ''  
# /mnt/md0/halin/Models/
training(configs=configs, 
         data_path=PATH, 
         batch_size=configs[0]['batch_size'], 
         channels=configs[0]['n_ant'],
         save_folder='/mnt/md0/halin/Models/',)


  
  


plot_collections(models, labels, save_path='')  
