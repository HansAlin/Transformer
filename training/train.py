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
from models.models import build_encoder_transformer, get_FLOPs, count_parameters, load_model, get_state
from dataHandler.datahandler import save_data, save_model, create_model_folder, get_model_path, get_chunked_data, get_trigger_data, get_value
from evaluate.evaluate import test_model, validate, get_energy, find_best_model
import lossFunctions.lossFunctions as ll


def get_least_utilized_gpu():
    gpu_utilization = []
    gpu_output = subprocess.check_output("nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits", shell=True)
    for x in gpu_output.decode('utf-8').split('\n')[:-1]:
        gpu_utilization.append(int(x))

    return gpu_utilization



def training(configs, cuda_device, model_folder='', test=False, retraining=False):

  data_type = configs[0]['transformer']['architecture'].get('data_type', 'chunked')
  if data_type == 'chunked':
    train_loader, val_loader, test_loader = get_chunked_data(config=configs[0], subset=False) # 
  else:
    train_loader, val_loader, test_loader = get_trigger_data(configs[0], subset=test) # 


  item = next(iter(train_loader))
  output_size = item[1].shape[-1]
  print(f"Output shape: {output_size}")

  precision = torch.float32

  #######################################################################
  #  Set up the GPU's to use                                           #
  #######################################################################
  if torch.cuda.is_available(): 
    device = torch.device(f'cuda:{cuda_device}')
    print(f"Current GPU usage: {device}")

  else:
      print('No GPU available training stops.')
      return None


  for config in configs:


 
    df = pd.DataFrame([], columns= ['Train_loss', 'Val_loss', 'metric', 'Epochs', 'lr'])
    if not retraining:
      
      if config['transformer']['basic']['model_type'] == "base_encoder": # config['transformer']['basic']['model_type']
        model = build_encoder_transformer(config) #

        optimizer = torch.optim.Adam(model.parameters(), lr=config['training']['learning_rate']) #
      
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=config['training']['step_size'], gamma=config['training']['decreas_factor']) #
        


      else:
          print("No model found")
          return None
      
      warmup = config['training'].get('warm_up', False)

      if warmup:
        lr = config['training']['learning_rate']
        warmup_epochs = 10
        def lr_lambda(epoch):
          if epoch < warmup_epochs:
              return epoch / warmup_epochs 
          else:
              return 1
           
        warmup_scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

      config['transformer']['num of parameters']['FLOPs'] = get_FLOPs(model, config) #
      config['transformer']['basic']['model_path'] = create_model_folder(config['transformer']['basic']['model_num'], path=model_folder) # 
      results = count_parameters(model, verbose=False)
      config['transformer']['num of parameters']['num_param'] = results['total_param'] #
      config['transformer']['num of parameters']['encoder_param'] = results['encoder_param'] # 
      config['transformer']['num of parameters']['input_param'] = results['src_embed_param'] # 
      config['transformer']['num of parameters']['final_param'] = results['final_param'] # 
      config['transformer']['num of parameters']['pos_param'] = results['buf_param'] 
      config['transformer']['results'] = {}
      config['transformer']['results']['current_epoch'] = 0
      config['transformer']['results']['global_epoch'] = 0
      print(f"Number of paramters: {config['transformer']['num of parameters']['num_param']} input: {config['transformer']['num of parameters']['input_param']} encoder: {config['transformer']['num of parameters']['encoder_param']} final: {config['transformer']['num of parameters']['final_param']} pos: {config['transformer']['num of parameters']['pos_param']}")
      initial_epoch = 1
    else:
      model = load_model(config=config, text='last')
      state = get_state(config=config, text='last')
      optimizer = torch.optim.Adam(model.parameters(), lr=config['training']['learning_rate']) #
      scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=config['training']['step_size'], gamma=config['training']['decreas_factor']) #

      warmup = config['training'].get('warm_up', False)

      if warmup:
        lr = config['training']['learning_rate']
        warmup_epochs = 10
        def lr_lambda(epoch):
          if epoch < warmup_epochs:
              return epoch / warmup_epochs 
          else:
              return 1
           
        warmup_scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


      scheduler.load_state_dict(state['scheduler_state_dict'])
      initial_epoch = config['transformer']['results']['current_epoch'] + 1
      optimizer.load_state_dict(state['optimizer_state_dict'])
      # Move optimizer state to the same device
      for state in optimizer.state.values():
          for k, v in state.items():
              if isinstance(v, torch.Tensor):
                  state[k] = v.to(device)

      
    
    writer = SummaryWriter(config['transformer']['basic']['model_path'] + 'trainingdata')
    print(f"Follow on tensorboard: python3 -m tensorboard.main --logdir={config['transformer']['basic']['model_path']}trainingdata")
    #  python3 -m tensorboard.main --logdir=/mnt/md0/halin/Models/model_1/trainingdata
    



    model.to(device).to(precision)

    n_ant = config['transformer']['architecture']['n_ant']
   
    
    loss_type = config['training'].get('loss_fn', 'BCE')
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

    x_batch, y_batch = test_loader.__getitem__(0)
    x = x_batch.cpu().detach().numpy()
    y = y_batch.cpu().detach().numpy()
    
    plot_examples(x, y, config=config['transformer'], data_type=data_type)

    for epoch in range(initial_epoch, config['training']['num_epochs'] + 1):

      epoch_time = time.time()
      
      # set the model in training mode
      model.train()
      first_param = next(model.parameters())
      print(f"Type: {first_param.dtype} Device: {first_param.device}, GPU: {device}")
      
      train_loss = []
      val_loss = []
      metric = []
      # Training
      batch_num = 1
      #############################################
      # Training                                  #
      #############################################
        
      num_of_bathes = int(len(train_loader))
      for istep in tqdm(range(len(train_loader)), disable=False):

        print(f"Epoch {epoch}/{config['training']['num_epochs']} Batch {batch_num}/{num_of_bathes}, GPU: {device} ", end="\r") # config['training']['num_epochs']
  
        x_batch, y_batch = train_loader.__getitem__(istep)
        if data_type == 'chunked':
           y_batch = y_batch.max(dim=1)[0]
        x_batch, y_batch = x_batch.to(device).to(precision), y_batch.to(device).to(precision)

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

          if data_type == 'chunked':
            outputs = outputs.squeeze()

          val_loss.append(criterion(outputs, y_batch.squeeze()).item())
          pred = outputs.cpu().detach().numpy()
          preds.append(pred)
          ys.append(y_batch.cpu().detach().numpy())
        if data_type == 'chunked':
          noise_rejection = config['sampling']['rate']*1e9/config['input_length']
        else:
          noise_rejection = 10000
        threshold, accuracy, efficiency, precission, recall, F1 = validate(np.asarray(ys), np.asarray(preds), noise_rejection=noise_rejection)  # config['training']['metric']
        if config['training']['metric'] == 'Accuracy':
          met = accuracy
        elif config['training']['metric'] == 'Efficiency':
          met = efficiency
        elif config['training']['metric'] == 'Precision':
          met = precission
        elif config['training']['metric'] == 'recall':
          met = recall
        elif config['training']['metric'] == 'F1':
          met = F1  
        metric.append(met)

      train_loss = np.mean(train_loss)
      val_loss = np.mean(val_loss)    
      metric = np.mean(metric)

      current_lr = optimizer.state_dict()['param_groups'][0]['lr']
      print(f"Learning rate: {current_lr}", end=" ")

      if warmup:
        if epoch < warmup_epochs:
          warmup_scheduler.step()
        else:
          scheduler.step()  
      else:
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
      writer.add_scalar(config['training']['metric'], metric, epoch)    # config['training']['metric']
      writer.flush()

      print(f"Training Loss: {train_loss:.4f}, Validation Loss: {val_loss:.4f}, Val {config['training']['metric']}: {metric:.6f}, Time: {time.time() - epoch_time:.2f} s")

      #############################################
      # Early stopping
      #############################################
      if val_loss < min_val_loss:
        min_val_loss = val_loss
        early_stop_count = 0
      else:
        early_stop_count += 1
      if early_stop_count >= config['training']['early_stop']: # config['training']['early_stop']
        print("Early stopping!")
        break
      
      config['transformer']['results']['current_epoch'] += 1
      config['transformer']['results']['global_epoch'] += 1

    ###########################################
    # Training done                           #
    ###########################################  
  
    save_model(model, optimizer, scheduler, config, epoch, text='final', threshold=threshold, )  
    writer.close()   
    total_training_time = time.time() - total_time   
    print(f"Total time: {total_training_time} s")

    config['transformer']['results']['trained'] = True
    config['transformer']['results']['training_time'] = total_training_time


    save_data(config, df)
    
    del model

    find_best_model(config, device, test=test, test_loader=test_loader)