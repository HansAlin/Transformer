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

# device = None
device = (
    "mps"
    if getattr(torch, "has_mps", False)
    else "cuda"
    if torch.cuda.is_available()
    else "cpu"
)
print(f"Using device: {device}")


x_train, x_test, y_train, y_test = dh.get_test_data(path='/home/halin/Master/Transformer/Test/data/test_data.npy')
train_loader, test_loader = dh.prepare_data(x_train, x_test, y_train, y_test, 32)

model = md.build_encoder_transformer(embed_size=64,
                                      seq_len=100,
                                      d_model=64,
                                      N=1,
                                      h=1,
                                      dropout=0.1)
print(model)

config = {}
trained_model = tr.training(model, config)
                            
tr.save_data(trained_model=trained_model)

tr.plot_results(999)