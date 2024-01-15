from os.path import dirname, abspath, join
import sys
#import numpy as np
#from sklearn.preprocessing import StandardScaler
import torch 
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from torch.optim.lr_scheduler import ReduceLROnPlateau
import matplotlib.pyplot as plt
from os.path import dirname, abspath, join
import sys
import argparse


type(sys.path)
for path in sys.path:
   print(path)

import os
myhost = os.uname()[1]

print("Host name: ", myhost)

import dataHandler.datahandler as dh
import models.models as md
from training.train import training
from plots.plots import plot_collections
from model_configs.config import get_config

# TODO implement loading and saving of model
# TODO Implement train, val, test set use pytorch randomsplit()
# TODO implement unity test?
# TODO read Visualiz... and  Checklist....
# TODO use Pathlib?
# TODO save the optimaizer also
# TODO FLOP's from fvcore.nn import FlopCountAnalysis
# TODO Check final bloch down sizing if it is correct
# TODO Do I have to concern about float 32 and float 64

# TODO Add som ekind of timer
# TODO Scheck if scheduler works!
# TODO secure no model gets overwritten when running a new model

def main(start_model_num, batch_size, epochs, test, cuda_device, config_number): 
  models_path = '/mnt/md0/halin/Models/'
  hyper_paramters = [128,256,512]
  hyper_param_key = 'd_ff'
  labels = {'hyper_parameters': hyper_paramters, 'name': 'Number of heads: ({hyper_param_key}})'}
  
  if start_model_num == None:
    start_model_num = input("Enter start model number: ")
    start_model_num = int(start_model_num)
  model_num = start_model_num

  configs = []
  for i in range(len(hyper_paramters)):

    old_config = get_config(config_number)
    config = old_config.copy()
    config['batch_size'] = batch_size
    config['model_num'] = model_num
    config['num_epochs'] = epochs 
    
    # Copy the config from the model to inherit from
    if config['inherit_model'] != None:
      inherit_model = config['inherit_model']
      old_config = dh.get_model_config(inherit_model)
      config = old_config.copy()
      config['model_num'] = model_num
      config['model_path'] = ''
      config['inherit_model'] = inherit_model
      config['num_epochs'] = epochs
      config['Accuracy'] = 0
      config['Efficiency'] = 0
      config['Precission'] = 0
      config['training_time'] = 0

    # Update the hyper parameter
    config[hyper_param_key] = hyper_paramters[i]

    configs.append(config)
    model_num += 1

  training(configs=configs, 
          cuda_device=cuda_device,
          batch_size=configs[0]['batch_size'], 
          channels=configs[0]['n_ant'],
          model_folder=models_path,
          test=test,)
  

if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  parser.add_argument('--start_model_num', help='Check for no interference', type=int)
  parser.add_argument('--batch_size', type=int, help='Default 32', default=32)
  parser.add_argument('--epochs', type=int, help='Default 100', default=100)
  parser.add_argument('--test', type=bool, help='Default False', default=False)
  parser.add_argument('--cuda_device', type=int,help='Default 0', default=0)
  parser.add_argument('--config_number', type=int,help='Default 1', default=1)
  args = parser.parse_args()
  main(args.start_model_num, args.batch_size, args.epochs, args.test, args.cuda_device, args.config_number)


