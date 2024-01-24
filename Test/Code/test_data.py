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
from evaluate.evaluate import get_MMac, count_parameters
from evaluate.evaluate import get_MMac, count_parameters
from plots.plots import get_area_under_curve, get_noise_reduction, get_roc, get_NSE_AT_NRF


#train_data, val_data, test_data = get_data_binary_class(save_test_set=True, subset=True)
# train_data, val_data, test_data = get_test_data()
# df = collect_config_to_df(model_numbers=[2,3,4,5,6,7,8,9,10,11,12], save_path='/home/halin/Master/Transformer/Test/ModelsResults/collections/')
# print(df[['model_num', 'embed_type', 'pos_enc_type', 'loss_function', 'seq_len', 'd_model', 'd_ff', 'N', 'h', "num_parms", "current_epoch", 'Accuracy', 'Efficiency', 'Precission']])
#get_data_binary_class(seq_len=256, batch_size=32, subset=True, save_test_set=True)
def change_key(dict_obj, old_key, new_key):
    dict_obj[new_key] = dict_obj.pop(old_key)

def add_key_if_not_exists(dict_obj, key, value):
    if key not in dict_obj:
        dict_obj[key] = value

# ###################################################################
# # Add data to config file                                         #
# ###################################################################
# model_numbers = [1,2]#[0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,16,17,18,19,20,21,22,23,24,25]
# bins = 10000
# print('{:<20} {:>20} {:>20} {:>20} '.format('Model number', 'Encoder type','Embed type',  'Final type' ))
# for model_number in model_numbers:
#     config = get_model_config(model_num=model_number)
#     model = build_encoder_transformer(config)
#     # results = count_parameters(model, verbose=False)
#     # config['encoder_param'] = results['encoder_param']
#     # config['input_param'] = results['src_embed_param']
#     # config['final_param'] = results['final_param']
#     # config['pos_param'] = results['buf_param']
#     # config['num_param'] = results['total_trainable_param']
#     # macs, params = get_MMac(model=model, 
#     #             batch_size=config['batch_size'], 
#     #             seq_len=config['seq_len'], 
#     #             channels=config['n_ant'])
#     y_true, y_pred = get_predictions(model_number=model_number)
#     # # x, y = get_roc(y_true, y_pred, bins=10000)
#     # # roc_area = get_area_under_curve(x, y)
#     x, y = get_noise_reduction(y_true, y_pred, bins=bins)
#     # nr_area = get_area_under_curve(x, y)
#     # config['MACs'] = macs
#     # config['num_parms'] = params
#     # config['roc_area'] = roc_area
#     # config['nr_area'] = nr_area
#     # nse = get_NSE_AT_NRF(TP=x, noise_reduction=y,  number_of_noise=10000)
#     # config['NSE_AT_10KNRF'] = nse
#     # change_key(config, 'num_parms', 'num_param')
#     # results = count_parameters(model, verbose=False)
#     # config['num_param'] = results['total_param']
#     # config['encoder_param'] = results['encoder_param']
#     # config['input_param'] = results['src_embed_param']
#     # config['final_param'] = results['final_param']
#     # config['pos_param'] = results['buf_param']
#     add_key_if_not_exists(config, 'data_type', 'chunked')
#     if config['data_type'] == 'classic':
#         config['data_type'] = 'trigger'
      

#     print('{:<20} {:>20} {:>20} {:>20}  '.format(model_number, config['encoder_type'], config['embed_type'], config['final_type']))        

#     save_data(config)

###################################################################
# Print config file                                               #
###################################################################
model_num = [13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,99,100,101,102,103,104,105,106,107,108,109,110,111,108,112,113,114,115,116,117,118,119,120,121] # 101,105,106,107,108,109,110   
col_1 = 'model_num'
# col_2 = 'encoder_type'
# col_3 = 'embed_type'#
# col_4 = 'final_type' 
col_5 = 'num_param'
col_6 = 'encoder_param'
col_7 = 'input_param'
col_8 = 'final_param'
col_9 = 'seq_len'
col_10 = 'd_model'
col_11 = 'd_ff'
col_12 = 'N'
col_13 = 'h' 
# col_14 = 'data_type'
col_15 = 'NSE_AT_10KNRF'
df = collect_config_to_df(model_numbers=model_num, save_path='/home/halin/Master/Transformer/Test/ModelsResults/collections/', save=True)
# col_2, col_3, col_4,col_14,
df = df.sort_values('NSE_AT_10KNRF', ascending=False)
print(df[[col_1, col_5, col_6, col_7, col_8, col_9, col_10, col_11, col_12, col_13,  col_15]])



###################################################################
# Test count of parameters                                        # 
###################################################################
# model_num = 21
# config = get_model_config(model_num=model_num)
# model = build_encoder_transformer(config)
# count_parameters(model, verbose=True)        


