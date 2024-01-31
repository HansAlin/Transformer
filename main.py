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


# type(sys.path)
# for path in sys.path:
#    print(path)

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

def main(start_model_num, batch_size, epochs, test, cuda_device, config_number, inherit_model, pretrained): 
  models_path = '/mnt/md0/halin/Models/'
  # 'double_linear', 'single_linear', 'seq_average_linear'
  hyper_paramters = ['Relative']
  hyper_param_key = 'pos_enc_type'
  labels = {'hyper_parameters': hyper_paramters, 'name': 'Encoder type: ({hyper_param_key}})'}
  
  if start_model_num == None:
    if test:
      print("Test mode")
    else:
      print("Training mode") 
    start_model_num = input("Enter start model number: ")
    check_path = models_path + f"model_{start_model_num}"

    if check_directory_exists(check_path):
        print("File exists")
        over_write = input("Do you want to overwrite the model? (y/n): ")
        if over_write == 'y':
          print("Overwriting model")
        else:
          print("Not overwriting model")
          sys.exit()  
    else:
      print('Model not exisiting')
 
    start_model_num = int(start_model_num)
  
  old_config = get_config(config_number)
  while hyper_param_key not in old_config['architecture'].keys():
    hyper_param_key = input("Enter hyper parameter key: ")
    hyper_param_key = str(hyper_param_key)   
  
  model_num = start_model_num

  configs = []
  for i in range(len(hyper_paramters)):

    
    config = old_config.copy()
    config['training']['batch_size'] = batch_size
    config['basic']['model_num'] = model_num
    config['training']['num_epochs'] = epochs
    # config['batch_size'] = batch_size
    # config['model_num'] = model_num
    # config['num_epochs'] = epochs 
    
    # Copy the config from the model to inherit from
    if inherit_model != None:
      old_config = dh.get_model_config(inherit_model, type_of_file='yaml' )
      config = old_config.copy()
      #config['model_num'] = model_num
      config['basic']['model_num'] = model_num
      #config['model_path'] = ''
      config['basic']['model_path'] = ''
      if pretrained:
        # config['inherit_model'] = inherit_model
        config['architecture']['inherit_model'] = inherit_model
      else:
        # config['inherit_model'] = None  
        config['architecture']['inherit_model'] = None

      config['training']['num_epochs'] = epochs
      config['results']['Accuracy'] = 0
      config['results']['Efficiency'] = 0
      config['results']['Precission'] = 0
      config['results']['training_time'] = 0
      config['results']['energy'] = 0
      config['results']['trained'] = False
      config['results']['power'] = 0
      config['results']['roc_area'] = 0
      config['results']['nr_area'] = 0
      config['results']['NSE_AT_10KNRF'] = 0
      config['results']['TRESH_AT_10KNRF'] = 0
      config['results']['NSE_AT_100KNRF'] = 0
      config['architecture']['pretrained'] = pretrained
      

    # Update the hyper parameter
    config['architecture'][hyper_param_key] = hyper_paramters[i]

    configs.append(config)
    model_num += 1

  training(configs=configs, 
          cuda_device=cuda_device,
          batch_size=configs[0]['training']['batch_size'], 
          channels=configs[0]['architecture']['n_ant'],
          model_folder=models_path,
          test=test,)
  
def check_directory_exists(file_path):
    return os.path.isdir(file_path)  

if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  parser.add_argument('--start_model_num', help='Check for no interference', type=int)
  parser.add_argument('--batch_size', type=int, help='Default 32', default=64)
  parser.add_argument('--epochs', type=int, help='Default 100', default=2)
  parser.add_argument('--test', type=bool, help='Default False', default=True)
  parser.add_argument('--cuda_device', type=int,help='Default 0', default=0)
  parser.add_argument('--config_number', type=int,help='Default 1', default=0)
  parser.add_argument('--inherit_model', type=int,help='Default 18', default=19)
  parser.add_argument('--pretrained', type=bool,help='Default False', default=False)
  
  args = parser.parse_args()
  main(args.start_model_num, args.batch_size, args.epochs, args.test, args.cuda_device, args.config_number, args.inherit_model, args.pretrained)


