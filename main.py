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

def main(start_model_num, epochs, test, cuda_device, config_number, inherit_model, retrain): 
  models_path = '/mnt/md0/halin/Models/'
  # 'double_linear', 'single_linear', 'seq_average_linear'

  if inherit_model != None and retrain == True:
    old_config = dh.get_model_config(inherit_model, type_of_file='yaml' )

    config = old_config.copy()
    configs = [config]
  else:
    config = get_config(config_number)

    #config = old_config.copy()
    hyper_param = {
                #  'N':[15]
                # 'pos_enc_type':['Relative'],
                'max_relative_position': [64],
                #   'h': [2,4,8],
                # 'd_model': [32],
                # 'h': [16],
                # 'd_model': [512],
                # 'd_ff': [256],
                  }

    # Get all combinations
    combinations = list(itertools.product(*hyper_param.values()))

    if start_model_num == None:
      if test:
        print("Test mode")
        start_model_num = 1000
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
    

    model_num = start_model_num

    configs = []
    
        
    # Copy the config from the model to inherit from
    if inherit_model != None:
      old_config = dh.get_model_config(inherit_model, type_of_file='yaml' )
      config = old_config.copy()


      config['transformer']['architecture']['inherit_model'] = inherit_model
      config['transformer']['architecture']['pretrained'] = False 
      config['transformer']['basic']['model_path'] = ''
      config['transformer']['training']['num_epochs'] = epochs
      config['transformer']['results']['Accuracy'] = 0
      config['transformer']['results']['Efficiency'] = 0
      config['transformer']['results']['Precission'] = 0
      config['transformer']['results']['training_time'] = 0
      config['transformer']['results']['energy'] = 0
      config['transformer']['results']['trained'] = False
      config['transformer']['results']['power'] = 0
      config['transformer']['results']['roc_area'] = 0
      config['transformer']['results']['nr_area'] = 0
      config['transformer']['results']['NSE_AT_10KNRF'] = 0
      config['transformer']['results']['TRESH_AT_10KNRF'] = 0
      config['transformer']['results']['NSE_AT_100KNRF'] = 0
      config['transformer']['num of parameters']['MACs'] = 0
      config['transformer']['num of parameters']['encoder_param'] = 0
      config['transformer']['num of parameters']['final_param'] = 0
      config['transformer']['num of parameters']['input_param'] = 0
      config['transformer']['num of parameters']['num_param'] = 0
      config['transformer']['num of parameters']['pos_param'] = 0  
      config['transformer']['results']['current_epoch'] = 0
      config['transformer']['results']['global_epoch'] = 0

 
    config['transformer']['basic']['model_num'] = model_num
    config['transformer']['training']['num_epochs'] = epochs

    for combination in combinations:
      params = dict(zip(hyper_param.keys(), combination))
      print(params)
      for hyper_param_key, hyper_paramter in params.items():
        config['transformer']['architecture'][hyper_param_key] = hyper_paramter
      config['transformer']['basic']['model_num'] = model_num
      configs.append(copy.deepcopy(config))
      model_num += 1

  training(configs=configs, 
          cuda_device=cuda_device,
          second_device=1,
          batch_size=configs[0]['transformer']['training']['batch_size'], 
          channels=configs[0]['transformer']['architecture']['n_ant'],
          model_folder=models_path,
          test=test,
          retrained=retrain)
  
def check_directory_exists(file_path):
    return os.path.isdir(file_path)  

if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  parser.add_argument('--start_model_num', help='Check for no interference', type=int)

  parser.add_argument('--epochs', type=int, help='Default 100', default=2)
  parser.add_argument('--test', type=bool, help='Default False', default=True)
  parser.add_argument('--cuda_device', type=int,help='Default 0', default=0)
  parser.add_argument('--config_number', type=int,help='Default 1', default=0)
  parser.add_argument('--inherit_model', type=int,help='Default 18', default=None)
  parser.add_argument('--retrain', type=bool,help='Default False', default=False)
  
  args = parser.parse_args()
  main(args.start_model_num, args.epochs, args.test, args.cuda_device, args.config_number, args.inherit_model, args.retrain)


