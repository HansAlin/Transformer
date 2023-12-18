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
import training.train as tr
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

# Hyper paramters:
#     learning rate 
#         ¤  learning rate functions  
#          ¤  facor 
#          ¤  based on val_loss or val_acc ???  


hyper_paramters = [0.01, 2, 10000]
labels = ['Omega: 0.01', 'Omega: 2', 'Omega: 10000']
start_model_num = 997
epochs = 100
test = False

model_num = start_model_num
models = []
for i, hyper_paramter in enumerate(hyper_paramters):
  models.append(model_num)
  config = {'model_name': "with_out_activation_in_final_block",
            'model_type': "base_encoder",
              'model':None,
              'pre_trained': None,
              'pos_enc_type':'normal',
              'model_num': model_num,
              'embed_size': hyper_paramter,
              'seq_len': 100,
              'd_model': 512,
              'N': 8,
              'h': 4,
              'dropout': 0.1,
              'num_epochs': epochs,
              'batch_size': 32,
              #"experiment_name": f"/home/halin/Master/Transformer/Test/ModelsResults/model_{model_num}/runs",
              "learning_rate": 1e-3,
              "decreas_factor": 0.5,
              "num_parms":0,
              "data_path":'',
              "current_epoch":0,
              "model_path":'',
              "test_acc":0,
              "early_stop":5,
              "omega": 10000,
              "trained_noise":0,
              "trained_signal":0,
              "acc":0,
              "TP":0,
              "TN":0,
              "FP":0,
              "FN":0,


            }
  if test:
    PATH = os.getcwd() + '/Test/data/test_1000_data.npy'
  else:
    PATH = ''  
  
  tr.training(config, data_path=PATH)

  tr.plot_results(model_num)
  
  model_num += 1


plot_collections(models, labels, save_path='')  
