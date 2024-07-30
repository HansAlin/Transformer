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
  parser.add_argument('--save_configs', default=None, type=bool, help='Save configs to file.')
  
  
  args = parser.parse_args()

    # Check if any arguments were provided
  if all(value is None for value in vars(args).values()):
    raise Exception("No arguments provided")
  
  return args

def compare_dicts(dict1, dict2, exclude_keys, parent_key=''):
    diff_keys = []

    for k in dict1:
        full_key = f'{parent_key}.{k}' if parent_key else k
        if k not in exclude_keys:
            if isinstance(dict1[k], dict) and isinstance(dict2.get(k), dict):
                diff_keys.extend(compare_dicts(dict1[k], dict2.get(k), exclude_keys, full_key))
            elif dict1[k] != dict2.get(k):
                diff_keys.append((full_key, dict1[k], dict2.get(k)))

    for k in dict2:
        if k not in exclude_keys and k not in dict1:
            full_key = f'{parent_key}.{k}' if parent_key else k
            diff_keys.append((full_key, None, dict2[k]))

    return diff_keys

def save_configs(configs):
  for config in configs:
    model_num = config['transformer']['basic']['model_num']
    model_path = f'/home/halin/Master/nuradio-analysis/data/models/fLow_0.08-fhigh_0.23-rate_0.5/config_{model_num}.yaml'
    config['transformer']['basic']['model_path'] = model_path
    save_path = f'/home/halin/Master/nuradio-analysis/configs/chunked/config_{model_num}.yaml'
    with open(save_path, 'w') as file:
      yaml.dump(config, file)


def update_nested_dict(d, key, value):
    for k, v in d.items():
        if isinstance(v, dict):
            update_nested_dict(v, key, value)
        if k == key:
            d[k] = value

def main(): 

  # start_model_num, epochs, test, cuda_device, config_number, inherit_model, retrain

  try:
    args = parse_args()
  except Exception as e:
    print(e)  
    args = argparse.Namespace()
    args.start_model_num = None           # Whcih model number to start from
    args.epochs = 20                       # Number of epochs
    args.test = True                      # Test mode
    args.cuda_device = 1                  # Which cuda device to use
    args.resume_training_for_model = None # Resume training for a specific model
    args.inherit_model = 2                # Inherit model number 0, 1, 2 means standard configs were 2 is a basic config specified in model_configs/config.py
    args.save_configs = False             # Save configs to file in stead of training if True
    alt_combination = 'combi'             # option to choose between single = one item from each, 
                                          # combi  = all combinations
                                          # restrict = only combinations that satisfy FLOPs constraints
    subset = None # None
  # Uncomment this if you want to compare the configs with a specific model
  # compare_model = 320
  # compare_config = dh.get_model_config(compare_model)


  #models_path = '/mnt/md0/halin/Models/'
  models_path = '/home/halin/Master/Transformer/trained_models/'
  # 'double_linear', 'single_linear', 'seq_average_linear'

  if args.resume_training_for_model != None:
    old_config = dh.get_model_config(args.resume_training_for_model, type_of_file='yaml' )

    config = old_config.copy()
    config['training']['num_epochs'] = args.epochs
    configs = [config]
    retraining = True
  else:
    # config = get_config(args.config_number)
    retraining = False
    # #config = old_config.copy()

    cnn_configs = [
            {'kernel_size': 3, 'stride': 1},
           # {'kernel_size': 3, 'stride': 2},
        ]
    vit_configs = [
            # {'kernel_size': 2, 'stride': 2},
            {'kernel_size': 4, 'stride': 4},
        ]


    hyper_param = {
            'd_model': [16],
            'd_ff': [32],
            'h': [4],
            'N': [2], 
            'batch_size': [1024],
            'max_pool': [True],
            'embed_type': ['ViT'],
            'max_relative_position': [16],
            'pos_enc_type': ['Relative'],
    }


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


    model_num = int(args.start_model_num)

    configs = dh.config_production(base_config_number=args.inherit_model,
                                   test_dict=hyper_param,
                                   cnn_configs=cnn_configs,
                                   vit_configs=vit_configs,
                                   alt_combination=alt_combination,
                                   subset=subset,)


  for config in configs:
    config['transformer']['basic']['model_num'] = model_num
    config['training']['num_epochs'] = args.epochs
    print(f"Model number: {model_num}, Batch size: {dh.get_value(config, 'batch_size'):>4}, d_model: {dh.get_value(config, 'd_model'):>3}, d_ff: {dh.get_value(config, 'd_ff'):>3}, h: {dh.get_value(config, 'h'):>2}, N: {dh.get_value(config, 'N'):>2}, Loss function: {dh.get_value(config, 'loss_fn'):>14}, Embedding: {dh.get_value(config, 'embed_type'):>6}, Positional encoding: {dh.get_value(config, 'pos_enc_type'):>12}, Projection: {dh.get_value(config, 'projection_type'):>8}, Antenna type: {dh.get_value(config, 'antenna_type'):>7}, Data type: {dh.get_value(config, 'data_type'):>7}" )
    model_num += 1

  # Uncomment this if you want to compare the configs with a specific model
  # exclude_keys = {'basic', 'results', 'num of parameters'}
  # diff_keys = compare_dicts(configs[0], compare_config, exclude_keys)
  # print(f"Different keys: {diff_keys}")
  # for key, val1, val2 in diff_keys:
  #     print(f'Key: {key}\nDict1 Value: {val1}\nDict2 Value: {val2}\n---')
  # answer = input("Do you want to continue? (y/n): ")
  # if answer == 'n':
  #   sys.exit()

  if args.save_configs:
    save_configs(configs)
  else:
    training(configs=configs, 
          cuda_device=args.cuda_device,
          model_folder=models_path,
          test=args.test,
          retraining=retraining,)
  
def check_directory_exists(file_path):
    return os.path.isdir(file_path)  

if __name__ == "__main__":
  
  main()