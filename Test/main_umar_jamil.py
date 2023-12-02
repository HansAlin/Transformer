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


# TODO Implement some kind of early stopping
# TODO Implement train, val, test set
# TODO some kind of adabtive lerarning rate
# TODO implement unity test?
# TODO read Visualiz... and  Checklist....
# TODO Prepare slides


model_num = 998
config = {'model_name': "base_encoder",
            'model':None,
            'model_num': model_num,
            'embed_size': 64,
            'seq_len': 100,
            'd_model': 512,
            'N': 8,
            'h': 4,
            'dropout': 0.1,
            'num_epochs': 50,
            'batch_size': 32,
            #"experiment_name": f"/home/halin/Master/Transformer/Test/ModelsResults/model_{model_num}/runs",
            "learning_rate": 1e-4,
            "num_parms":0,
            "data_path":'',
            "current_epoch":0,
            "model_path":'',
            "test_acc":0,

          }
tr.training(config, test=False)


tr.plot_results(model_num)