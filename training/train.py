import numpy as np
import torch
import torch.nn as nn
import os
import matplotlib.pyplot as plt
from itertools import zip_longest

from plots.plots import histogram, plot_performance_curve, plot_results, plot_examples, plot_performance
import tqdm as tqdm
from torch.utils.tensorboard import SummaryWriter
from torch.optim.lr_scheduler import ReduceLROnPlateau
import pandas as pd
import sys
import time
from tqdm import tqdm
import subprocess

import tensorboard
from models.models import build_encoder_transformer
from dataHandler.datahandler import save_data, save_model, create_model_folder, get_model_path, get_chunked_data, get_trigger_data
from evaluate.evaluate import test_model, validate, get_energy, get_MMac, count_parameters


def get_least_utilized_gpu():
    gpu_utilization = []
    gpu_output = subprocess.check_output("nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits", shell=True)
    for x in gpu_output.decode('utf-8').split('\n')[:-1]:
        gpu_utilization.append(int(x))

    return gpu_utilization

def training(configs, cuda_device, second_device=None, batch_size=32, channels=4, model_folder='', test=False, retrained=False):

  data_type = configs[0]['transformer']['architecture'].get('data_type', 'chunked')
  if data_type == 'chunked':
    train_loader, val_loader, test_loader = get_chunked_data(batch_size=batch_size, config=configs[0], subset=test) # 
  else:
    train_loader, val_loader, test_loader = get_trigger_data(configs[0], subset=test) # 


  item = next(iter(train_loader))
  output_size = item[1].shape[-1]
  print(f"Output shape: {output_size}")



  for config in configs:

 
    df = pd.DataFrame([], columns= ['Train_loss', 'Val_loss', 'metric', 'Epochs', 'lr'])
    if not retrained:
      
      config['transformer']['results']['power'  ] = 0 # get_energy(cuda_device) # 
      # config['transformer']['architecture']['output_size'] = output_size #
      # config['transformer']['architecture']['out_put_shape'] = output_size # config['transformer']['architecture']['out_put_shape']
      
      if config['transformer']['basic']['model_type'] == "base_encoder": # config['transformer']['basic']['model_type']
        model = build_encoder_transformer(config['transformer']) #

        optimizer = torch.optim.Adam(model.parameters(), lr=config['transformer']['training']['learning_rate']) #
      
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=config['transformer']['training']['step_size'], gamma=config['transformer']['training']['decreas_factor']) #

      else:
          print("No model found")
          return None
      
      
      config['transformer']['num of parameters']['MACs'], config['transformer']['num of parameters']['num_param'] = get_MMac(model, config) # 
      config['transformer']['basic']['model_path'] = create_model_folder(config['transformer']['basic']['model_num'], path=model_folder) # 
      results = count_parameters(model, verbose=False)
      config['transformer']['num of parameters']['encoder_param'] = results['encoder_param'] # 
      config['transformer']['num of parameters']['input_param'] = results['src_embed_param'] # 
      config['transformer']['num of parameters']['final_param'] = results['final_param'] # 
      config['transformer']['num of parameters']['pos_param'] = results['buf_param'] 
      print(f"Number of paramters: {config['transformer']['num of parameters']['num_param']} input: {config['transformer']['num of parameters']['input_param']} encoder: {config['transformer']['num of parameters']['encoder_param']} final: {config['transformer']['num of parameters']['final_param']} pos: {config['transformer']['num of parameters']['pos_param']}")
      initial_epoch = 1
    else:
      model = build_encoder_transformer(config['transformer']) 
      optimizer = torch.optim.Adam(model.parameters(), lr=config['transformer']['training']['learning_rate']) #
      scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=config['transformer']['training']['step_size'], gamma=config['transformer']['training']['decreas_factor']) #

      state = torch.load(model_folder + f"model_{config['transformer']['basic']['model_num']}/saved_model/model_{config['transformer']['basic']['model_num']}early_stop.pth") # config['transformer'][architecture]['inherit_model']
      model.load_state_dict(state['model_state_dict'])

      scheduler.load_state_dict(state['scheduler_state_dict'])
      initial_epoch = config['transformer']['results']['current_epoch']
      optimizer.load_state_dict(state['optimizer_state_dict'])
      # Move optimizer state to the same device
      for state in optimizer.state.values():
          for k, v in state.items():
              if isinstance(v, torch.Tensor):
                  state[k] = v.to(device)

      
    
    writer = SummaryWriter(config['transformer']['basic']['model_path'] + '/trainingdata')
    print(f"Follow on tensorboard: python3 -m tensorboard.main --logdir={config['transformer']['basic']['model_path']}trainingdata")
    #  python3 -m tensorboard.main --logdir=/mnt/md0/halin/Models/model_1/trainingdata
    

    #######################################################################
    #  Set up the GPU's to use                                           #
    #######################################################################
    if torch.cuda.is_available(): 
      device = torch.device(f'cuda:{cuda_device}')
      if second_device == None:
        print(f"Let's use GPU {cuda_device}!")

      else:  
        if torch.cuda.device_count() > 1 and get_least_utilized_gpu()[second_device] == 0 :
            print(f"Let's use GPU {cuda_device} and {second_device}!")
            model = nn.DataParallel(model, device_ids=[cuda_device, second_device])


        else:
            print(f"Let's use GPU {cuda_device}!")
            device = torch.device(f'cuda:{cuda_device}')


    else:
        device = torch.device("cpu")

    model = model.to(device)

 
    
    
    loss_type = config['transformer']['training'].get('loss_function', 'BCE')
    if loss_type == 'BCE':
      criterion = nn.BCELoss().to(device)
    elif loss_type == 'BCEWithLogits':
      criterion = nn.BCEWithLogitsLoss().to(device)
    else:
      print("No loss function found")
      return None
   
    

    early_stop_count = 0
    min_val_loss = float('inf')
    total_time = time.time()

    for epoch in range(initial_epoch, config['transformer']['training']['num_epochs']):

      epoch_time = time.time()
      
      # set the model in training mode
      model.train()
      
      train_loss = []
      val_loss = []
      metric = []
      # Training
      batch_num = 1
      #############################################
      # Training                                  #
      #############################################
        
      num_of_bathes = int(len(train_loader))
      for istep in tqdm(range(len(train_loader))):

        print(f"Epoch {epoch}/{config['transformer']['training']['num_epochs']} Batch {batch_num}/{num_of_bathes}", end="\r") # config['transformer']['training']['num_epochs']
  
        x_batch, y_batch = train_loader.__getitem__(istep)
        if data_type == 'chunked':
           y_batch = y_batch.max(dim=1)[0]
        x_batch, y_batch = x_batch.to(device), y_batch.to(device)
        
        outputs = model(x_batch)

        loss = criterion(outputs, y_batch.squeeze())

        loss.backward()

        optimizer.step()
        optimizer.zero_grad()

        train_loss.append(loss.item())
        batch_num += 1

 
      
      #############################################
      # Validation                                #
      #############################################
      model.eval()
      with torch.no_grad():

        for istep in tqdm(range(len(val_loader))):

          x_batch, y_batch = val_loader.__getitem__(istep)
          
          if data_type == 'chunked':
            y_batch = y_batch.max(dim=1)[0]

          x_batch, y_batch = x_batch.to(device), y_batch.to(device)
          outputs = model(x_batch)
          loss = criterion(outputs, y_batch.squeeze())
          if  config['transformer']['training']['loss_function']== 'BCEWithLogits':      #
            outputs = torch.sigmoid(outputs)
          pred = outputs.round()
          met = validate(y_batch.cpu().detach().numpy(), pred.cpu().detach().numpy(), config['transformer']['training']['metric'])  # config['transformer']['training']['metric']
            
          val_loss.append(loss.item())
          metric.append(met)

      train_loss = np.mean(train_loss)
      val_loss = np.mean(val_loss)    
      metric = np.mean(metric)

      current_lr = optimizer.state_dict()['param_groups'][0]['lr']
      print(f"Learning rate: {current_lr}", end=" ")
      scheduler.step()

      #############################################
      # Data saving
      #############################################
      
      temp_df = pd.DataFrame([[train_loss, val_loss, metric, epoch, current_lr]], 
                            columns= ['Train_loss', 'Val_loss', 'metric', 'Epochs', 'lr'])
      df = pd.concat([df, temp_df], ignore_index=True)
      # TODO maybe use best_val_loss instead of best_accuracy
      if val_loss < min_val_loss:
        save_model(model, optimizer, scheduler, config, epoch, text='early_stop')
        print(f"Model saved at epoch {epoch + 1}")
      save_data(config, df)

      ############################################
      #  Tensorboard
      ############################################
      writer.add_scalar('Training Loss' , train_loss, epoch)
      writer.add_scalar('Validation Loss' , val_loss, epoch)
      writer.add_scalar(config['transformer']['training']['metric'], metric, epoch)    # config['transformer']['training']['metric']
      writer.flush()

      print(f"Training Loss: {train_loss:.4f}, Validation Loss: {val_loss:.4f}, Val {config['transformer']['training']['metric']}: {metric:.6f}, Time: {time.time() - epoch_time:.2f} s")

      #############################################
      # Early stopping
      #############################################
      if val_loss < min_val_loss:
        min_val_loss = val_loss
        early_stop_count = 0
      else:
        early_stop_count += 1
      if early_stop_count >= config['transformer']['training']['early_stop']: # config['transformer']['training']['early_stop']
        print("Early stopping!")
        break
      
      config['transformer']['results']['current_epoch'] += 1
      config['transformer']['results']['global_epoch'] += 1

      config['transformer']['results']['power'] = ((config['transformer']['results']['current_epoch'])*config['transformer']['results']['power'] + get_energy(cuda_device))/(config['transformer']['results']['current_epoch'] + 1)
    ###########################################
    # Training done                           #
    ###########################################  
    save_model(model, optimizer, scheduler, config, epoch, text='final')  
    writer.close()   
    total_training_time = time.time() - total_time   
    print(f"Total time: {total_training_time} s")

    config['transformer']['results']['trained'] = True
    config['transformer']['results']['training_time'] = total_training_time
    config['transformer']['results']['energy'] = config['transformer']['results']['power']*total_training_time

    model_path = get_model_path(config)
    
    print(f'Preloading model {model_path}')
    state = torch.load(model_path)
    model.load_state_dict(state['model_state_dict'])

    if torch.cuda.is_available(): 

      device = torch.device(f'cuda:{cuda_device}')
    else:
      device = torch.device("cpu")

    y_pred_data, accuracy , efficiency, precision = test_model(model=model, 
                                                                test_loader=test_loader,
                                                                device=device, 
                                                                config=config['transformer'])    
    config['transformer']['results']['Accuracy'] = float(accuracy)
    config['transformer']['results']['Efficiency'] = float(efficiency)
    config['transformer']['results']['Precission'] = float(precision)

    print(f"Test efficiency: {config['transformer']['results']['Efficiency']:.4f}")

    histogram(y_pred_data['y_pred'], y_pred_data['y'], config['transformer'])
    nr_area, nse, threshold = plot_performance_curve([y_pred_data['y_pred']], [y_pred_data['y']], [config], curve='nr', x_lim=[0,1], bins=10000)
    config['transformer']['results']['nr_area'] = float(nr_area)
    config['transformer']['results']['NSE_AT_10KNRF'] = float(nse)
    roc_area, nse, threshold = plot_performance_curve([y_pred_data['y_pred']], [y_pred_data['y']], [config], curve='roc', bins=10000)
    config['transformer']['results']['roc_area'] = float(roc_area)
    config['transformer']['results']['NSE_AT_10KROC'] = float(nse)
    config['transformer']['results']['TRESH_AT_10KNRF'] = float(threshold)
    plot_results(config['transformer']['basic']['model_num'], config['transformer'])


    x_batch, y_batch = train_loader.__getitem__(0)
    data = x_batch.cpu().detach().numpy()
      
    plot_examples(data, config=config['transformer'])
    plot_performance(config['transformer'], device, x_batch=x_batch, y_batch=y_batch, lim_value=0.5, )
    save_data(config['transformer'], df, y_pred_data)
