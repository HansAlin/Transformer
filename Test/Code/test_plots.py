import os
from os.path import dirname, abspath, join
import sys
import unittest
import pickle
import numpy as np
# Find code directory relative to our directory
CODE_DIR_1  ='/home/halin/Master/Transformer/'
sys.path.append(CODE_DIR_1)
type(sys.path)
for path in sys.path:
   print(path)
from models.models import TimeInputEmbeddings, LayerNormalization, FinalBinaryBlock, build_encoder_transformer
from dataHandler.datahandler import get_data, prepare_data, find_hyperparameters
from training.train import test_model
from plots.plots import plot_results, plot_weights, histogram, noise_reduction_factor, plot_collections, plot_examples
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
#  Plot collections of noise reduction factors                    #
###################################################################
save_path = ''
models_path = '/mnt/md0/halin/Models/'
models = [990,991,992]
parameter = 'd_model'
hyper_parameters = find_hyperparameters(model_number=models, 
                                        parameter=parameter,
                                        models_path=models_path)
labels = {'hyper_parameters': hyper_parameters, 'name': 'H parameter (d_model)'}

plot_collections(models, 
                 labels, 
                 save_path=save_path, 
                 models_path=models_path,
                 x_lim=[0,1])

# data = np.load('/home/halin/Master/Transformer/Test/data/test_100_data.npy')
# plot_examples(data)