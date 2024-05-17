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
    args.start_model_num = None
    args.epochs = 100
    args.test = False
    args.cuda_device = 2
    args.config_number = 0
    args.resume_training_for_model = None
    args.inherit_model = 256
    args.save_configs = False
    alt_combination = True

  compare_model = 256
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

    cnn_configs = [
            {'kernel_size': 3, 'stride': 1},
           # {'kernel_size': 3, 'stride': 2},
        ]
    vit_configs = [
            # {'kernel_size': 2, 'stride': 2},
            {'kernel_size': 5, 'stride': 5},
        ]


    hyper_param = {
        'N': [1, 2, 3, 4,5][::-1],
        'd_model': [8,16,32,64,128][::-1],
        'd_ff': [16,32, 64, 128,256][::-1],
        'h':[2,4,8,16,32][::-1], 
        'batch_size': [1024,1024,1024,1024,1024][::-1],
    }

    # Get all combinations
    if alt_combination:
      combinations = list(zip(*hyper_param.values()))
    else:
      combinations = list(itertools.product(*hyper_param.values()))
    # TODO remove this after running 256, ...
    #combinations = combinations[5:]
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

      if params.get('embed_type') == 'linear' and params.get('max_pool') == True:
        continue

      print(f"Model number: {model_num}, and parameters: {params}")

      # Have to update kernels differently for cnn and ViT
      if params.get('embed_type') == 'cnn':

        if config['transformer']['architecture']['n_ant'] == 5 or params.get('n_ant') == 5:
            if params.get('d_model') != None:
                if params.get('d_model') % 5 == 0:
                    pass
                elif config['transformer']['architecture']['d_model']  % 5 == 0:
                    pass
            elif config['transformer']['architecture']['d_model']  % 5 == 0:
                pass    
            else:
                assert False, "d_model must be divisible by 5"


        for cnn_config in cnn_configs:
              params_copy = params.copy()
              params_copy.update(cnn_config)
              
              for (test_key, value) in params_copy.items():
                  update_nested_dict(config, test_key, value)

      elif params.get('embed_type') == 'ViT':

        for vit_config in vit_configs:
            params_copy = params.copy()
            params_copy.update(vit_config)
            params.update(vit_config)
            for (test_key, value) in params_copy.items():
                update_nested_dict(config, test_key, value)  
            config['transformer']['architecture']['max_relative_position'] = config['transformer']['architecture']['max_relative_position'] // params['stride']    

      else:
            for (test_key, value) in params.items():
                update_nested_dict(config, test_key, value)

      if params.get('max_pool') == True:
            config['transformer']['architecture']['max_relative_position'] = config['transformer']['architecture']['max_relative_position'] // 2    

      config['transformer']['basic']['model_num'] = model_num
      config['training']['batch_size'] = config['transformer']['training']['batch_size']
      configs.append(copy.deepcopy(config))
      model_num += 1

  for config in configs:
    print(f"Batch size: {dh.get_value(config, 'batch_size'):>4}, d_model: {dh.get_value(config, 'd_model'):>3}, d_ff: {dh.get_value(config, 'd_ff'):>3}, h: {dh.get_value(config, 'h'):>2}, N: {dh.get_value(config, 'N'):>2}, Loss function: {dh.get_value(config, 'loss_function'):>14}, Embedding: {dh.get_value(config, 'embed_type'):>6}, Positional encoding: {dh.get_value(config, 'pos_enc_type'):>12}, Projection: {dh.get_value(config, 'projection_type'):>8}, Antenna type: {dh.get_value(config, 'antenna_type'):>7}, Data type: {dh.get_value(config, 'data_type'):>7}" )

  exclude_keys = {'basic', 'results', 'num of parameters'}
  diff_keys = compare_dicts(configs[0], compare_config, exclude_keys)
  print(f"Different keys: {diff_keys}")
  for key, val1, val2 in diff_keys:
      print(f'Key: {key}\nDict1 Value: {val1}\nDict2 Value: {val2}\n---')
  answer = input("Do you want to continue? (y/n): ")
  if answer == 'n':
    sys.exit()

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