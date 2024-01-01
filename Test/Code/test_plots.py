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
from dataHandler.datahandler import get_data, prepare_data, find_hyperparameters
# from training.train import test_model
from plots.plots import plot_results, plot_weights, histogram, plot_performance_curve, plot_collections, plot_examples, plot_performance
import torch
import subprocess
import pandas as pd


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
models_path = '/mnt/md0/halin/Models/'
models = [1]
curve = 'roc'
window_pred = False
str_models = '_'.join(map(str, models))
save_path = f'/home/halin/Master/Transformer/Test/ModelsResults/model_{str_models}_{curve}_window_pred_{str(window_pred)}.png'

parameter = 'd_model'
hyper_parameters = find_hyperparameters(model_number=models, 
                                        parameter=parameter,
                                        models_path=models_path)
labels = {'hyper_parameters': hyper_parameters, 'name': 'H parameter (d_model)'}

plot_collections(models, 
                 labels, 
                 save_path=save_path, 
                 models_path=models_path,
                 x_lim=[0,1],
                 window_pred=True,
                 curve=curve)

###################################################################
# Plot performance of a model                                     #
###################################################################
# num = 1
# plot_performance(model_num=num, save_path=f'/home/halin/Master/Transformer/Test/ModelsResults/model_{num}_')


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
