import os
from os.path import dirname, abspath, join
import sys
import unittest
import pickle
import numpy as np
# Find code directory relative to our directory
THIS_DIR = dirname(__file__)
CODE_DIR = abspath(join(THIS_DIR, '..', ''))
sys.path.append(CODE_DIR)

from models.models_1 import TimeInputEmbeddings, LayerNormalization, FinalBinaryBlock, build_encoder_transformer
from dataHandler.datahandler import get_data, prepare_data
from training.train import test_model
from plots.plots import plot_results, plot_weights, histogram, noise_reduction_factor, plot_collections, plot_examples
import torch
import subprocess


# path = os.getcwd()

# data_path = path + '/Test/data/test_1000_data.npy'
# x_train, x_test, x_val, y_train, y_val, y_test = get_data(path=data_path)

# train_loader, val_loader, test_loader, number_of_noise, number_of_signals = prepare_data(x_train, x_val, x_test, y_train, y_val, y_test, 32)

# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")    
# model_number = 998
# with open(f"Test/ModelsResults/model_{model_number}/config.txt", 'rb') as f:
#     config = pickle.load(f)

# model = build_encoder_transformer(config)

# blocks = [ 'final_binary_block']
# for block in blocks:
#     plot_weights(model, config, block=block, quiet=True)


# models = [997,998,999]

# labels = {'hyper_parameters': [0.01, 2, 10000], 'name': 'Omega'}

# plot_collections(models, labels, save_path='')

data = np.load('/home/halin/Master/Transformer/Test/data/test_100_data.npy')
plot_examples(data)