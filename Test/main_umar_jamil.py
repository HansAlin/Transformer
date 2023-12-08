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

# TODO implement loading and saving of model
# TODO Implement some kind of early stopping
# TODO Implement train, val, test set use pytorch randomsplit()
# TODO implement last test
# TODO some kind of adabtive lerarning rate
# TODO implement unity test?
# TODO read Visualiz... and  Checklist....
# TODO Prepare slides
# TODO use Pathlib?
# TODO save the optimaizer also
# TODO Verify that the validation is correctly implemented according to batch size 
# TODO implement the some thing better for the last block loch at the time series
# TODO change the size in the posstional encoder
# TODO FLOP's from fvcore.nn import FlopCountAnalysis
# TODO Check final bloch down sizing if it is correct
# Hyper paramters:
#     learning rate 
#         ¤  learning rate functions  
#          ¤  facor 
#          ¤  based on val_loss or val_acc ???  




model_num = 994
config = {'model_name': "base_encoder",
            'model':None,
            'pos_enc_type':'none',
            'model_num': model_num,
            'embed_size': 64,
            'seq_len': 100,
            'd_model': 512,
            'N': 2,
            'h': 2,
            'dropout': 0.1,
            'num_epochs': 100,
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

          }
tr.training(config, data_path='')
# '/home/halin/Master/Transformer/Test/data/mini_test_data.npy'

tr.plot_results(model_num)