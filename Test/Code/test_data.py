import argparse
import numpy as np
from matplotlib import pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.ticker import MultipleLocator
import os
import time
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import sys
import pickle
CODE_DIR_1  ='/home/halin/Master/Transformer/'
sys.path.append(CODE_DIR_1)

from dataHandler.datahandler import get_data, prepare_data, find_hyperparameters, get_data_binary_class, get_test_data, collect_config_to_df

#train_data, val_data, test_data = get_data_binary_class(save_test_set=True, subset=True)
# train_data, val_data, test_data = get_test_data()
df = collect_config_to_df(model_numbers=[2,3,4,5,6,7,8,9,10,11,12], save_path='/home/halin/Master/Transformer/Test/ModelsResults/collections/')
print(df[['model_num', 'embed_type', 'pos_enc_type', 'loss_function', 'seq_len', 'd_model', 'd_ff', 'N', 'h', "num_parms", "current_epoch", 'Accuracy', 'Efficiency', 'Precission']])
CONFIG_PATH = '/mnt/md0/halin/Models/model_13/config.txt'
TEXT_CONFIG_PATH = '/mnt/md0/halin/Models/model_13/text_config.txt'
with open(CONFIG_PATH, 'rb') as f:
    config = pickle.load(f)
config['model_path'] = '/mnt/md0/halin/Models/model_13/'

with open(CONFIG_PATH, "wb") as fp:
    pickle.dump(config, fp)

# A easier to read part
with open(TEXT_CONFIG_PATH, 'w') as data:
    for key, value in config.items():
        data.write('%s: %s\n' % (key, value))

