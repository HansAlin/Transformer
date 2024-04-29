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
import itertools
import copy
import yaml



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



def parse_args():
  parser = argparse.ArgumentParser()
  parser.add_argument('--start_model_num', help='Check for no interference', type=int)
  parser.add_argument('--epochs', type=int, help='Number of epochs')
  parser.add_argument('--test', type=bool, help='True or False for test mode. ')
  parser.add_argument('--cuda_device', type=int,help='Which cuda device to use. ')
  parser.add_argument('--config_number', type=int,help='Which config file type to use. Recomend 0')
  parser.add_argument('--resume_training_for_model', default=None, type=int, help='Resume training for model')
  parser.add_argument('--inherit_model', default=None, type=int, help='Inherit model')
  
  
  args = parser.parse_args()

    # Check if any arguments were provided
  if all(value is None for value in vars(args).values()):
    raise Exception("No arguments provided")
  
  return args

def compare_dicts(dict1, dict2, exclude_keys):
    diff_keys = [k for k in dict1 if k not in exclude_keys and dict1[k] != dict2.get(k)]
    diff_keys.extend([k for k in dict2 if k not in exclude_keys and dict2[k] != dict1.get(k)])
    return diff_keys

def main(): 

  # start_model_num, epochs, test, cuda_device, config_number, inherit_model, retrain

  try:
    args = parse_args()
  except Exception as e:
    print(e)  
    args = argparse.Namespace()
    args.start_model_num = None
    args.epochs = 100
    args.test = False
    args.cuda_device = 2
    args.config_number = 0
    args.resume_training_for_model = None
    args.inherit_model = 265

  compare_model = 265
  compare_config = dh.get_model_config(compare_model)


  models_path = '/mnt/md0/halin/Models/'
  # 'double_linear', 'single_linear', 'seq_average_linear'

  if args.resume_training_for_model != None:
    old_config = dh.get_model_config(args.resume_training_for_model, type_of_file='yaml' )

    config = old_config.copy()
    config['transformer']['training']['num_epochs'] = args.epochs
    configs = [config]
    retraining = True
  else:
    config = get_config(args.config_number)
    retraining = False
    #config = old_config.copy()
    hyper_param = {
                # 'N':[2]
                # 'pos_enc_type':['Relative'],
                #'max_relative_position': [None],
      # 'antenna_type': ['LPDA'],
      # 'data_type': ['chunked'],
    'd_model': [8],
    'd_ff': [32, 128],
    'h': [4, 8],
    'N': [2, 3],  

                  }

    # Get all combinations
    combinations = list(itertools.product(*hyper_param.values()))
    # TODO remove this after running 256, ...
    # combinations = combinations[2:]
    if args.start_model_num == None:
      if args.test:
        print("Test mode")
        args.start_model_num = 1000
      else:
        print("Training mode") 
        args.start_model_num = input("Enter start model number: ")
        check_path = models_path + f"model_{args.start_model_num}"
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




  
      args.start_model_num = int(args.start_model_num)
    

    model_num = args.start_model_num

    configs = []
    
        
    # Copy the config from the model to inherit from
    if args.inherit_model != None:
      old_config = dh.get_model_config(args.inherit_model, type_of_file='yaml' )
      config = old_config.copy()
      if 'transformer' not in config:
        config = {'transformer': config}


      config['transformer']['architecture']['inherit_model'] = args.inherit_model
      config['transformer']['architecture']['pretrained'] = False 
      config['transformer']['basic']['model_path'] = ''
      config['transformer']['training']['num_epochs'] = args.epochs
      config['transformer']['results'] = {}
      config['transformer']['num of parameters'] = {}
      config['transformer']['results']['current_epoch'] = 0
      config['transformer']['results']['global_epoch'] = 0

 
    config['transformer']['basic']['model_num'] = model_num
    config['transformer']['training']['num_epochs'] = args.epochs

    for combination in combinations:
      params = dict(zip(hyper_param.keys(), combination))
      print(f"Model number: {model_num}, and parameters: {params}")
      for hyper_param_key, hyper_paramter in params.items():
        config['transformer']['architecture'][hyper_param_key] = hyper_paramter
      config['transformer']['basic']['model_num'] = model_num
      configs.append(copy.deepcopy(config))
      model_num += 1

  for config in configs:
    print(f"Batch size: {dh.get_value(config, 'batch_size'):>5}, d_model: {dh.get_value(config, 'd_model'):>5}, d_ff: {dh.get_value(config, 'd_ff'):>5}, h: {dh.get_value(config, 'h'):>5}, N: {dh.get_value(config, 'N'):>5}, Loss function: {dh.get_value(config, 'loss_function'):>5}, Embedding: {dh.get_value(config, 'embed_type'):>5}, Positional encoding: {dh.get_value(config, 'pos_enc_type'):>5}, Projection: {dh.get_value(config, 'projection_type'):>5}, Antenna type: {dh.get_value(config, 'antenna_type'):>5}, Data type: {dh.get_value(config, 'data_type'):>5}" )

    exclude_keys = {'basic', 'num_of_parameters', 'results'}
    diff_keys = compare_dicts(config, compare_config, exclude_keys)
    print(f"Different keys: {diff_keys}")
  answer = input("Do you want to continue? (y/n): ")
  if answer == 'n':
    sys.exit()

  training(configs=configs, 
          cuda_device=args.cuda_device,
          model_folder=models_path,
          test=args.test,
          retraining=retraining,)
  
def check_directory_exists(file_path):
    return os.path.isdir(file_path)  

if __name__ == "__main__":
  
  main()
