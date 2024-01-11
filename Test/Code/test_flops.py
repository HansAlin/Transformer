import sys
import torch
import torch.nn as nn
import pickle
import os
from ptflops import get_model_complexity_info
CODE_DIR_1  ='/home/halin/Master/Transformer/'
sys.path.append(CODE_DIR_1)
from models.models import build_encoder_transformer, get_n_params
from evaluate.evaluate import get_MMac
from dataHandler.datahandler import get_model_config

from os.path import dirname, abspath, join
myhost = os.uname()[1]
print("Host name: ", myhost)
config = get_model_config(model_num=13)
batch_size = config['batch_size']
seq_len = config['seq_len']
n_ant = config['n_ant']

model = build_encoder_transformer(config)
x_train = torch.randn(batch_size, seq_len, n_ant)
out = model.encode(x_train, src_mask=None)

macs, params = get_MMac(model, batch_size=batch_size,  seq_len=seq_len, channels=n_ant)
print("MACs: ", macs)
print("Params: ", params)

