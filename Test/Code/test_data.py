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
import pandas as pd
import yaml
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
# # ###################################################################
# model_numbers = [99,100,101,102,103,104,105,106,107,108,109,110,111,108,112,113,114,115,116,117,118,119,120,121,122,123,124,125,127] 
# bins = 10000
# col_1 = 'model_num'
# col_2 = 'encoder_type'
# col_3 = 'embed_type'#
# col_4 = 'final_type' 
# col_5 = 'num_param'
# col_6 = 'encoder_param'
# col_7 = 'input_param'
# col_8 = 'final_param'
# col_9 = 'seq_len'
# col_10 = 'd_model'
# col_11 = 'd_ff'
# col_12 = 'N'
# col_13 = 'h' 
# # col_14 = 'data_type'
# col_15 = 'NSE_AT_10KNRF'
# # print('{:<20} {:>20} {:>20} {:>20}  '.format('Model number', 'Prev NSE_AT_10KNRF','Nev ROC NSE_AT_10KNR', 'ROC Threshold' ))
# for model_number in model_numbers:
#     config = get_model_config(model_num=model_number)
#     # model = build_encoder_transformer(config)
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
#     # y_true, y_pred = get_predictions(model_number=model_number)
#     # x, y, nse_roc, threshold_roc  = get_roc(y_true, y_pred, bins=10000)
#     if config['final_param'] == config['seq_len'] + 1 :
#         config['final_type'] = 'seq_average_linear'
#     elif config['final_param'] == config['d_model'] + 1:
#         config['final_type'] = 'd_model_average_linear'    
#     # # roc_area = get_area_under_curve(x, y)
#     #x, y, nse, threshold = get_noise_reduction(y_true, y_pred, bins=bins)
#     # nr_area = get_area_under_curve(x, y)
#     # config['MACs'] = macs
#     # config['num_parms'] = params
#     # config['roc_area'] = roc_area
#     # config['nr_area'] = nr_area
#     # config['TRESH_AT_10KNRF'] = threshold_roc
#     # nse = get_NSE_AT_NRF(TP=x, noise_reduction=y,  number_of_noise=10000)
#     # config['NSE_AT_10KNRF'] = nse
#     # config['final_type'] = 'd_model_average_linear'
#     # change_key(config, 'num_parms', 'num_param')
#     # results = count_parameters(model, verbose=False)
#     # config['num_param'] = results['total_param']
#     # config['encoder_param'] = results['encoder_param']
#     # config['input_param'] = results['src_embed_param']
#     # config['final_param'] = results['final_param']
#     # config['pos_param'] = results['buf_param']
#     add_key_if_not_exists(config, 'normalization', 'layer')
#     # if config['data_type'] == 'classic':
#     #     config['data_type'] = 'trigger'
      

#     # print('{:<20} {:>20} {:>20} {:>20} '.format(model_number, config[col_15], nse_roc, threshold_roc, ))        

#     save_data(config)

###################################################################
# Print config file                                               #
###################################################################
        # model_numbers = [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14, 15, 16, 17, 18, 19, 20, 21, 22, 23,24,25,26,27]
# model_numbers = [99,100,101,102,103,104,105,106,107,108,109,110,111,108,112,113,114,115,116,117,118,119,120,121,122,123,124,125,127]
        
model_num = [13,14, 15, 16, 17, 18, 19, 20, 21, 22, 23,24,25,26,27,99,100,101,102,103,104,105,106,107,108,109,110,111,112,113,114,115,116,117,118,119,120,121,122,123,124,125,126,127,128,129,130,131] # 101,105,106,107,108,109,110   
col_1 = 'model_num'
col_2 = 'encoder_type'
col_3 = 'embed_type'#
col_4 = 'final_type' 
col_6 = 'num_param'
col_5 = 'normalization'
col_7 = 'pos_enc_type'
col_9 = 'seq_len'
col_10 = 'd_model'
col_11 = 'd_ff'
col_12 = 'N'
col_13 = 'h' 
col_14 = 'data_type'
col_15 = 'NSE_AT_10KNRF'
#col_16 = 'TRESH_AT_10KNRF'
col_17 = 'data_type'
#col_18 = 'MACs'
df = collect_config_to_df(model_numbers=model_num, save_path='/home/halin/Master/Transformer/Test/ModelsResults/collections/', save=True)

df = df.sort_values('NSE_AT_10KNRF', ascending=False)
print(df[[col_1,col_2,col_3, col_4, col_5,col_6, col_7,  col_9, col_10, col_11, col_12, col_13,  col_15, col_17, ]])
# df.plot.scatter(x=col_15, y=col_6, c='DarkBlue')
# # plt.savefig('/home/halin/Master/Transformer/Test/Code/plots/NRF_vs_param.png')
# df_2 = df[[col_1,col_2,col_3, col_4, col_5,col_6, col_7, col_9, col_10, col_11, col_12, col_13,  col_15, col_17, ]]
# # df_2 = df_2.copy()

# df_2[col_2] = pd.factorize(df_2[col_2])[0] + 1
# df_2[col_3] = pd.factorize(df_2[col_3])[0] + 1
# df_2[col_4] = pd.factorize(df_2[col_4])[0] + 1
# df_2[col_5] = pd.factorize(df_2[col_5])[0] + 1
# df_2[col_17] = pd.factorize(df_2[col_17])[0] + 1
# df_2[col_7] = pd.factorize(df_2[col_7])[0] + 1
# print(df_2)
# correlation_matrix = df_2.corr()
# print(correlation_matrix)

# def get_similar(df, ignore_col):

#     cols = ['N', 'h', 'seq_len', 'd_model',  'd_ff', 'data_type', 'encoder_type', 'embed_type', 'encoder_type', 'final_type', 'pos_enc_type', 'normalization', 'loss_function']
#     #cols = [col for col in df.columns if col not in ['model_num', 'd_ff']]
#     cols.remove(ignore_col)
    
#     # Create a mask that identifies the rows where all columns (except 'model_num' and 'd_ff') are identical
#     mask = df.duplicated(subset=cols, keep=False)

#     # Use the mask to filter the DataFrame
#     similar_df = df[mask]
#     indices = similar_df.index
#     indices = indices.intersection(df.index)
#     selected_rows = df.loc[indices]
#     print(selected_rows[[col_1,col_2,col_3, col_4, col_5,col_6,col_7,  col_9, col_10, col_11, col_12, col_13,  col_15, col_17, ]])
    

#     return selected_rows['model_num'].tolist()

# models = get_similar(df=df, ignore_col='seq_len')

# print(models)
###################################################################
# Test count of parameters                                        # 
###################################################################
# model_num = 21
# config = get_model_config(model_num=model_num)
# model = build_encoder_transformer(config)
# count_parameters(model, verbose=True)




###################################################################
# Changing config file style                                      #
###################################################################                
# def transform_config(config):
#     def get_int(key):
#         value = config.get(key, None)
#         return int(value) if value is not None else None

#     def get_float(key):
#         value = config.get(key, None)
#         return float(value) if value is not None else None

#     newdict = {
#         'basic': {
#             'model_num': get_int('model_num'),
#             'model_name': config.get('model_name', None),
#             'model_type': config.get('model_type', None),
#             'model': config.get('model', None),
#             'model_path': config.get('model_path', None),
#             'data_path': config.get('data_path', None),
#         },
#         'architecture': {
#             'inherit_model': get_int('inherit_model'),
#             'pretrained': config.get('pretrained', None),
#             'embed_type': config.get('embed_type', None),
#             'pos_enc_type': config.get('pos_enc_type', None),
#             'encoder_type': config.get('encoder_type', None),
#             'final_type': config.get('final_type', None),
#             'normalization': config.get('normalization', None),
#             'seq_len': get_int('seq_len'),
#             'd_model': get_int('d_model'),
#             'd_ff': get_int('d_ff'),
#             'N': get_int('N'),
#             'h': get_int('h'),
#             'by_pass': config.get('by_pass', False),
#             'output_size': get_int('output_size'),
#             'output_shape': get_int('output_shape'),
#             'data_type': config.get('data_type', None),
#             'n_ant': get_int('n_ant'),
#             'omega': get_int('omega'),
#             'activation': config.get('activation', None),
#         },
#         'num of parameters': {
#             'num_param': get_int('num_param'),
#             'MACs': get_int('MACs'),
#             'input_param': get_int('input_param'),
#             'pos_param': get_int('pos_param'),
#             'encoder_param': get_int('encoder_param'),
#             'final_param': get_int('final_param'),
#             'trained_noise': get_int('trained_noise'),
#             'trained_signal': get_int('trained_signal'),
#         },
#         'training': {
#             'loss_function': config.get('loss_function', None),
#             'dropout': config.get('dropout', None),
#             'num_epochs': get_int('num_epochs'),
#             'batch_size': get_int('batch_size'),
#             'learning_rate': get_float('learning_rate'),
#             'decreas_factor': get_float('decreas_factor'),
#             'early_stop': get_int('early_stop'),
#             'metric': config.get('metric', None),
#         },
#         'results': {
#             'current_epoch': get_int('current_epoch'),
#             'global_epoch': get_int('global_epoch'),
#             'test_acc': get_float('test_acc'),
#             'Accuracy': get_float('Accuracy'),
#             'Efficiency': get_float('Efficiency'),
#             'Precission': get_float('Precission'),
#             'roc_area': get_float('roc_area'),
#             'nr_area': get_float('nr_area'),
#             'TRESH_AT_10KNRF': get_float('TRESH_AT_10KNRF'),
#             'NSE_AT_10KNRF': get_float('NSE_AT_10KNRF'),
#             'NSE_AT_10KROC': get_float('NSE_AT_10KROC'),
#             'NSE_AT_100KNRF': get_float('NSE_AT_100KNRF'),
#             'trained': config.get('trained', False),
#             'power': get_float('power'),
#             'training_time': get_float('training_time'),
#             'energy': get_float('energy'),
#         }
#     }

#     return newdict
# model_numbers = [126]
# for model_number in model_numbers:
#     config = get_model_config(model_num=model_number, type_of_file='txt')

#     new_dict = transform_config(config)
#     with open(config['model_path'] + 'config.yaml', 'w') as data:
#         yaml.dump(new_dict, data, default_flow_style=False)

        
