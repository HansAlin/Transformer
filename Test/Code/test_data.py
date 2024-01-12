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

from dataHandler.datahandler import get_data, prepare_data, find_hyperparameters, get_data_binary_class, get_test_data, collect_config_to_df, get_model_config, get_predictions, save_data, save_model, create_model_folder
from models.models import build_encoder_transformer, get_n_params
from evaluate.evaluate import get_MMac
from plots.plots import get_area_under_curve, get_noise_reduction, get_roc


#train_data, val_data, test_data = get_data_binary_class(save_test_set=True, subset=True)
# train_data, val_data, test_data = get_test_data()
# df = collect_config_to_df(model_numbers=[2,3,4,5,6,7,8,9,10,11,12], save_path='/home/halin/Master/Transformer/Test/ModelsResults/collections/')
# print(df[['model_num', 'embed_type', 'pos_enc_type', 'loss_function', 'seq_len', 'd_model', 'd_ff', 'N', 'h', "num_parms", "current_epoch", 'Accuracy', 'Efficiency', 'Precission']])

model_numbers = [13,14,15]
print('{:<20} {:>10} {:>10} {:>10} {:>20}'.format('Model number', 'MACs', 'Params', 'ROC area', 'Noise reduction area'))
for model_number in model_numbers:
    config = get_model_config(model_num=model_number)
    model = build_encoder_transformer(config)
    macs, params = get_MMac(model=model, 
                batch_size=config['batch_size'], 
                seq_len=config['seq_len'], 
                channels=config['n_ant'])
    y_true, y_pred = get_predictions(model_number=model_number)
    x, y = get_roc(y_true, y_pred, bins=1000)
    roc_area = get_area_under_curve(x, y)
    x, y = get_noise_reduction(y_true, y_pred, bins=1000)
    nr_area = get_area_under_curve(x, y)
    config['MACs'] = macs
    config['num_parms'] = params
    config['roc_area'] = roc_area
    config['nr_area'] = nr_area
    save_data(config)
    print('{:<20} {:>10.2e} {:>10.2e} {:>10.6f} {:>20.4e}'.format(model_number, macs, params, roc_area, nr_area))





