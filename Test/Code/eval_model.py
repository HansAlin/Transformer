from os.path import dirname, abspath, join
import sys
THIS_DIR = dirname(__file__)
CODE_DIR = abspath(join(THIS_DIR, '..', ''))
sys.path.append(CODE_DIR)
import pickle
import torch

import matplotlib.pyplot as plt
import numpy as np
from models.models_1 import build_encoder_transformer
from training.train import test_model
from dataHandler.datahandler import get_data, prepare_data
from plots.plots import histogram, noise_reduction_factor, plot_results

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")           
print(f"Using device: {device}, name of GPU: {torch.cuda.get_device_name(device=device)}")

model_num = 993
MODEL_PATH = f'/home/halin/Master/Transformer/Test/ModelsResults/model_{model_num}/saved_model/model_{model_num}.pth'
CONFIG_PATH = f'/home/halin/Master/Transformer/Test/ModelsResults/model_{model_num}/config.txt'

with open(CONFIG_PATH, 'rb') as f:
    config = pickle.load(f)

x_train, x_test, x_val, y_train, y_val, y_test = get_data()

train_loader, val_loader, test_loader = prepare_data(x_train, x_val, x_test, y_train, y_val, y_test, config['batch_size'], multi_channel=True)
del x_train
del x_test
del x_val
del y_train
del y_test
del y_val

model = build_encoder_transformer(config)

arr, y_pred_data = test_model(model, test_loader, device, config)

histogram(y_pred_data['y_pred'], y_pred_data['y'], config)
noise_reduction_factor([y_pred_data['y_pred']], [y_pred_data['y']], [config])

