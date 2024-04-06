import numpy as np
import torch
import torch.nn as nn
import os
import matplotlib.pyplot as plt
from itertools import zip_longest
import subprocess

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
from models.models import build_encoder_transformer, get_FLOPs, count_parameters, load_model
from dataHandler.datahandler import save_data, save_model, create_model_folder, get_model_path, get_chunked_data, get_trigger_data, get_value
from evaluate.evaluate import test_model, validate, get_energy
import lossFunctions.lossFunctions as ll


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

  precision = torch.float32

  for config in configs:

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
      
      
      config['transformer']['num of parameters']['FLOPs'] = get_FLOPs(model, config) #
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

      
    
    writer = SummaryWriter(config['transformer']['basic']['model_path'] + 'trainingdata')
    print(f"Follow on tensorboard: python3 -m tensorboard.main --logdir={config['transformer']['basic']['model_path']}trainingdata")
    #  python3 -m tensorboard.main --logdir=/mnt/md0/halin/Models/model_1/trainingdata
    



    model = model.to(device).to(precision)

    n_ant = config['transformer']['architecture']['n_ant']
   
    
    loss_type = config['transformer']['training'].get('loss_function', 'BCE')
    if loss_type == 'BCE':
      criterion = nn.BCELoss()
    elif loss_type == 'BCEWithLogits':
      criterion = nn.BCEWithLogitsLoss()
    elif loss_type == 'hinge' or loss_type == 'hinge_max':
      criterion = ll.HingeLoss(loss_type)
    else:
      print("No loss function found")
      return None
   
    criterion.to(device).to(precision)

    early_stop_count = 0
    min_val_loss = float('inf')
    total_time = time.time()

    for epoch in range(initial_epoch, config['transformer']['training']['num_epochs'] + 1):

      epoch_time = time.time()
      
      # set the model in training mode
      model.train()
      first_param = next(model.parameters())
      print(f"Type: {first_param.dtype} Device: {first_param.device}")
      
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
        x_batch, y_batch = x_batch.to(device).to(precision), y_batch.to(device).to(precision)
        if data_type == 'phased':
          x_batch = x_batch[:, :, :n_ant]
        outputs = model(x_batch)

        if data_type == 'chunked':
          outputs = outputs.squeeze()

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
        preds = []
        ys = []

        for istep in tqdm(range(len(val_loader))):

          x_batch, y_batch = val_loader.__getitem__(istep)
          
          if data_type == 'chunked':
            y_batch = y_batch.max(dim=1)[0]

          x_batch, y_batch = x_batch.to(device).to(precision), y_batch.to(device).to(precision)
          outputs = model(x_batch)
          val_loss.append(criterion(outputs, y_batch.squeeze()).item())
          pred = outputs.cpu().detach().numpy()
          preds.append(pred)
          ys.append(y_batch.cpu().detach().numpy())

        threshold, accuracy, efficiency, precission, recall, F1 = validate(np.asarray(ys), np.asarray(preds))  # config['transformer']['training']['metric']
        if config['transformer']['training']['metric'] == 'Accuracy':
          met = accuracy
        elif config['transformer']['training']['metric'] == 'Efficiency':
          met = efficiency
        elif config['transformer']['training']['metric'] == 'Precision':
          met = precission
        elif config['transformer']['training']['metric'] == 'recall':
          met = recall
        elif config['transformer']['training']['metric'] == 'F1':
          met = F1  
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
        save_model(model, optimizer, scheduler, config, epoch, text='best_loss', threshold=threshold, )
        print(f"Model saved at epoch {epoch}")
      save_data(config, df)
      save_model(model, optimizer, scheduler, config, epoch, text=f'{epoch}', threshold=threshold,)
      histogram(preds, ys, config['transformer'], threshold=threshold, text=f'_{epoch}')
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
    save_model(model, optimizer, scheduler, config, epoch, text='final', threshold=threshold, )  
    writer.close()   
    total_training_time = time.time() - total_time   
    print(f"Total time: {total_training_time} s")

    config['transformer']['results']['trained'] = True
    config['transformer']['results']['training_time'] = total_training_time
    config['transformer']['results']['energy'] = config['transformer']['results']['power']*total_training_time

    # model_path = get_model_path(config)
    # model
    # print(f'Preloading model {model_path}')
    # state = torch.load(model_path)
    # model.load_state_dict(state['model_state_dict'])

    if torch.cuda.is_available(): 

      device = torch.device(f'cuda:{cuda_device}')
    else:
      device = torch.device("cpu")


    best_efficiency = 0
    best_accuracy = 0
    best_precision = 0
    best_model_epoch = None
    best_threshold = None
    best_model = None
    best_y_pred_data = None

    df = pd.DataFrame([], columns= ['Epoch', 'Accuracy', 'Precission', 'Efficiency', 'Threshold'])

    for model_epoch in range(1, config['transformer']['training']['num_epochs'] + 1):  
      model_path = get_model_path(config, f'{model_epoch}')
      model = load_model(config=config, text=f'{model_epoch}')

      y_pred_data, accuracy , efficiency, precision, threshold = test_model(model=model, 
                                                                  test_loader=test_loader,
                                                                  device=device, 
                                                                  config=config, 
                                                                  extra_identifier=f'_{model_epoch}',
                                                                  plot_attention=True)  
      temp_df = pd.DataFrame([[model_epoch, accuracy, precision, efficiency, threshold]],
                            columns= ['Epoch', 'Accuracy', 'Precission', 'Efficiency', 'Threshold'])
      df = pd.concat([df, temp_df], ignore_index=True)
      state_dict = torch.load(model_path)

      try:
          state_dict['model_state_dict']['threshold'] = threshold
      except:
          state_dict['threshold'] = threshold
      torch.save(state_dict, model_path)

      if efficiency >= best_efficiency:
        best_efficiency = efficiency
        best_accuracy = accuracy
        best_precision = precision
        best_model_epoch = model_epoch
        best_threshold = threshold
        best_model = model  
        best_y_pred_data = y_pred_data


    config['transformer']['results']['Accuracy'] = float(best_accuracy)
    config['transformer']['results']['Efficiency'] = float(best_efficiency)
    config['transformer']['results']['Precission'] = float(best_precision)
    config['transformer']['results']['best_epoch'] = int(best_model_epoch)
    config['transformer']['results']['best_threshold'] = float(best_threshold)

    print(f"Test efficiency for best model: {config['transformer']['results']['Efficiency']:.4f}")

    histogram(y_pred_data['y_pred'], y_pred_data['y'], config, text='_best', threshold=best_threshold)
    nr_area, nse, threshold = plot_performance_curve([y_pred_data['y_pred']], [y_pred_data['y']], [config], curve='nr', x_lim=[0,1], bins=1000, text='best')
    config['transformer']['results']['nr_area'] = float(nr_area)
    config['transformer']['results']['NSE_AT_10KNRF'] = float(nse)
    roc_area, nse, threshold = plot_performance_curve([y_pred_data['y_pred']], [y_pred_data['y']], [config], curve='roc', bins=1000, text='best')
    config['transformer']['results']['roc_area'] = float(roc_area)
    config['transformer']['results']['NSE_AT_10KROC'] = float(nse)
    config['transformer']['results']['TRESH_AT_10KNRF'] = float(threshold)
    plot_results(config['transformer']['basic']['model_num'], config['transformer'])


    x_batch, y_batch = train_loader.__getitem__(0)
    x = x_batch.cpu().detach().numpy()
    y = y_batch.cpu().detach().numpy()
      
    plot_examples(x, y, config=config['transformer'])
    plot_performance(config['transformer'], device, x_batch=x_batch, y_batch=y_batch, lim_value=best_threshold, )
    save_data(config, df, y_pred_data)
    save_data(config, df, y_pred_data, text='best')
    save_model(best_model, optimizer, scheduler, config, epoch, text='best', threshold=threshold, )
    # model_num = [str(get_value(config, 'model_num'))]
    # model_path = get_value(config, 'model_path') + '/plot/'
    # subprocess.call(['python3', '/home/halin/Master/Transformer/QuickVeffRatio.py', '--models'] + model_num + ['--save_path', model_path])