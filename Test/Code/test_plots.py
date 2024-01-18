import os
from os.path import dirname, abspath, join
import sys
import unittest
import pickle
import numpy as np
# Find code directory relative to our directory
CODE_DIR_1  ='/home/halin/Master/Transformer/'
sys.path.append(CODE_DIR_1)
# type(sys.path)
# for path in sys.path:
#    print(path)
from models.models import InputEmbeddings, LayerNormalization, FinalBlock, build_encoder_transformer
from dataHandler.datahandler import get_data, prepare_data, find_hyperparameters, get_model_config, collect_config_to_df
# from training.train import test_model
from plots.plots import get_area_under_curve, get_noise_reduction, get_roc, plot_performance_curve, histogram, plot_performance, plot_collections, plot_table
import torch
import subprocess
import pandas as pd
import time
import matplotlib.pyplot as plt



# path = os.getcwd()

# data_path = path + '/Test/data/test_1000_data.npy'
# x_train, x_test, x_val, y_train, y_val, y_test = get_data(path=data_path)

# train_loader, val_loader, test_loader, number_of_noise, number_of_signals = prepare_data(x_train, x_val, x_test, y_train, y_val, y_test, 32)

# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")    
# model_number = 992
# with open(f"/mnt/md0/halin/Models/model_{model_number}/config.txt", 'rb') as f:
#     config = pickle.load(f)
# path = config['model_path']
# save_path = f'/home/halin/Master/Transformer/Test/ModelsResults/test/model_{model_number}_noise_reduction.png'
# df = pd.read_pickle(path + 'y_pred_data.pkl')
# noise_reduction_factor([df['y_pred']], [df['y']], [config], save_path=save_path, x_lim=[0,1])
#histogram(df['y_pred'], df['y'], config,)

# model = build_encoder_transformer(config)

# blocks = [ 'final_binary_block']
# for block in blocks:
#     plot_weights(model, config, block=block, quiet=True)


###################################################################
#  Plot collections of noise reduction factors or roc             #
###################################################################
# models_path = '/mnt/md0/halin/Models/'
models = [13,14,15,16,17,18,19,20,21,22]
# curve = 'nr'
# parameter = 'model_num'
# bins = 10000

# str_models = '_'.join(map(str, models))
# #save_path = f"/mnt/md0/halin/Models/collections/{str_models}_{curve}.png"
# save_path = f'/home/halin/Master/Transformer/Test/presentation/model_{str_models}_{curve}.png'


# hyper_parameters = find_hyperparameters(model_number=models, 
#                                         parameter=parameter,
#                                         models_path=models_path)
# labels = {'hyper_parameters': hyper_parameters, 'name': f'Model num.,  {parameter}'}

# plot_collections(models, 
#                  labels, 
#                  save_path=save_path, 
#                  models_path=models_path,
#                  x_lim=[0.9,1],
#                  curve=curve,
#                  bins=bins)
# df = collect_config_to_df(model_numbers=models, save_path='/home/halin/Master/Transformer/Test/ModelsResults/collections/', save=False)
# df = df.sort_values('NSE_AT_10KNRF', ascending=False)
# keys = ['model_num', 'd_model', 'd_ff', 'N', 'h', "num_param", "current_epoch", 'NSE_AT_10KNRF']
# for key in keys:
#     if key not in df.columns:
#         df[key] = np.nan
# print(df[keys])

###################################################################
# Plot tabel of hyperparameters                                   #
###################################################################
models_path = '/mnt/md0/halin/Models/'

str_models = '_'.join(map(str, models))
save_path = f'/home/halin/Master/Transformer/Test/presentation/model_{str_models}_table.png'
df = collect_config_to_df(model_numbers=models, save_path=save_path)
keys = ['model_num', 'd_model', 'd_ff', 'N', 'h', "num_param", "pos_param", 'input_param', 'encoder_param', 'final_param', 'MACs', 'training_time']
plot_table(df, keys, save_path=save_path)

###################################################################
# Plot performance of a model                                     #
###################################################################
# num = 4
# plot_performance(model_num=num, lim_value=0.1, save_path=f'/home/halin/Master/Transformer/Test/ModelsResults/test/model_{num}_')


###################################################################
# Test noise reduction factor                                     #
###################################################################
# model_num = 990
# save_path=f'/home/halin/Master/Transformer/Test/ModelsResults/model_{model_num}_roc'
# data_path='/home/halin/Master/Transformer/Test/data/'
# model_path='/mnt/md0/halin/Models/'
# x_test = torch.load(data_path + 'example_x_data.pt')
# y_test = torch.load(data_path + 'example_y_data.pt')
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  
# MODEL_PATH = model_path + f'model_{model_num}/saved_model/model_{model_num}.pth'
# CONFIG_PATH = model_path + f'model_{model_num}/config.txt'
# with open(CONFIG_PATH, 'rb') as f:
#     config = pickle.load(f)

# model = build_encoder_transformer(config)
# state = torch.load(MODEL_PATH)
# model.load_state_dict(state['model_state_dict'])

# model.to(device)
# x_test, y_test = x_test.to(device), y_test.to(device)
# model.eval()
# pred = model.encode(x_test,src_mask=None)
# index = 0

# pred = pred.cpu().detach().numpy().reshape(-1,1)
# x_test = x_test.cpu().detach().numpy()
# y_test = y_test.cpu().detach().numpy().reshape(-1,1)

# plot_performance_curve(y_preds=[pred], 
#                        ys=[y_test], 
#                        configs=[config], 
#                        save_path=save_path, x_lim=[0,1], curve='roc')

###################################################################
# Plot single curves                                                     #
###################################################################
# model_num = 13
# model_path = '/mnt/md0/halin/Models/'
# df = pd.read_pickle(model_path + f'model_{model_num}/y_pred_data.pkl')
# y = df['y'].to_numpy()
# y_pred = df['y_pred'].to_numpy()
# cuda_device = 1 # If running performance plot

# plot_performance_curve(y_preds=[y_pred], 
#                        ys=[y], 
#                        configs=[get_model_config(model_num=model_num)], 
#                        save_path='', 
#                        x_lim=[0,1], 
#                        curve='roc',
#                        log_bins=False)
# plot_performance_curve(y_preds=[y_pred], 
#                        ys=[y], 
#                        configs=[get_model_config(model_num=model_num)], 
#                        save_path='', 
#                        x_lim=[0,1], 
#                        curve='nr',
#                        log_bins=False)
# histogram(y_pred=y_pred,
#           y=y,
#            config=get_model_config(model_num=model_num),
#              )
# torch.cuda.set_device(cuda_device)
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# print(f"Using device: {device}, name of GPU: {torch.cuda.get_device_name(device=device)}")
# plot_performance(config=get_model_config(model_num=model_num), device=device, lim_value=0.5)

###################################################################
# Plot multi curves                                               #
###################################################################
# model_nums = [13]
# window_pred = False
# bins = 100
# save_path = ''
# labels = None
# for model_num in model_nums:
#     plot_collections([model_num], 
#                     labels, 
#                     save_path=save_path, 
#                     models_path=models_path,
#                     x_lim=[0,1],
#                     window_pred=window_pred,
#                     curve='nr',
#                     bins=bins,
#                     log_bins=True)
    # plot_collections([model_num], 
    #                 labels, 
    #                 save_path=save_path, 
    #                 models_path=models_path,
    #                 x_lim=[0,1],
    #                 window_pred=window_pred,
    #                 curve='roc',
    #                 bins=bins,
    #                 log_bins=True)



