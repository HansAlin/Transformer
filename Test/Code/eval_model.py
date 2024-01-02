from os.path import dirname, abspath, join
import sys
CODE_DIR_1  ='/home/halin/Master/Transformer/'
sys.path.append(CODE_DIR_1)
import pickle
import torch

import matplotlib.pyplot as plt
import numpy as np
from models.models import build_encoder_transformer
from training.train import test_model
from dataHandler.datahandler import get_data, prepare_data, save_data
from plots.plots import histogram, plot_performance_curve, plot_results

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")           
print(f"Using device: {device}, name of GPU: {torch.cuda.get_device_name(device=device)}")

model_num = 1
harddrive = True
if harddrive:
    MODEL_PATH = f'/mnt/md0/halin/Models/model_{model_num}/saved_model/model_{model_num}.pth'
    CONFIG_PATH = f'/mnt/md0/halin/Models/model_{model_num}/config.txt'
else:
    MODEL_PATH = f'/home/halin/Master/Transformer/Test/ModelsResults/model_{model_num}/saved_model/model_{model_num}.pth'
    CONFIG_PATH = f'/home/halin/Master/Transformer/Test/ModelsResults/model_{model_num}/config.txt'   


with open(CONFIG_PATH, 'rb') as f:
    config = pickle.load(f)

train_loader, val_loader, test_loader = get_data(batch_size=config['batch_size'], 
                                                 seq_len=config['seq_len'],
                                                 subset=False,
                                                 )

model = build_encoder_transformer(config)

y_pred_data, accuracy, efficiency, precission = test_model(model, test_loader, device, config)
save_data(config=config, df=None, y_pred_data=y_pred_data  )
save_hist_path = f'/home/halin/Master/Transformer/Test/ModelsResults/test/hist_{model_num}.png'
histogram(y_pred_data['y_pred'], y_pred_data['y'], config, save_path=save_hist_path)
save_noise_reduction_path = f'/home/halin/Master/Transformer/Test/ModelsResults/test/noise_reduction_{model_num}.png'
plot_performance_curve([y_pred_data['y_pred']], [y_pred_data['y']], [config], bins=1000, save_path=save_noise_reduction_path, x_lim=[0,1], window_pred=True)

