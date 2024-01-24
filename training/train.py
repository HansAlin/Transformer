import numpy as np
import torch
import torch.nn as nn
import os
import matplotlib.pyplot as plt
from dataHandler.datahandler import get_data, get_data_binary_class, prepare_data, get_test_data

from plots.plots import histogram, plot_performance_curve, plot_results, plot_examples, plot_performance
import tqdm as tqdm
from torch.utils.tensorboard import SummaryWriter
from torch.optim.lr_scheduler import ReduceLROnPlateau
import pandas as pd
import sys
import time
from tqdm import tqdm

from models.models import build_encoder_transformer
from dataHandler.datahandler import save_data, save_model, create_model_folder
from evaluate.evaluate import test_model, validate, get_energy, get_MMac, count_parameters



def training(configs, cuda_device, batch_size=32, channels=4, model_folder='', test=False):
  if not test:
    data_type = configs[0].get('data_type', 'chunked')
    if data_type == 'chunked':
      train_loader, val_loader, test_loader = get_data(batch_size=batch_size, seq_len=configs[0]['seq_len'], subset=test)
    else:
      train_loader, val_loader, test_loader = get_data_binary_class(batch_size=batch_size, seq_len=configs[0]['seq_len'], subset=test)
  else:  
    train_loader, val_loader, test_loader = get_test_data(batch_size=batch_size, seq_len=configs[0]['seq_len'], n_antennas=channels)


  item = next(iter(train_loader))
  out_put_shape = item[0].shape[-2]
  print(f"Output shape: {out_put_shape}")

  torch.cuda.set_device(cuda_device)
  device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
  print(f"Using device: {device}, name of GPU: {torch.cuda.get_device_name(device=device)}")
  
  for config in configs:
    df = pd.DataFrame([], columns= ['Train_loss', 'Val_loss', 'metric', 'Epochs', 'lr'])
    config['power'] = get_energy(cuda_device)
    config['out_put_shape'] = out_put_shape
    if config['model_type'] == "base_encoder":
      model = build_encoder_transformer(config)

      if config['inherit_model'] != None:

        new_state_dic = model.state_dict()
        old_state = torch.load(model_folder + f"model_{config['inherit_model']}/saved_model/model_{config['inherit_model']}.pth")
        
        for name, param in old_state.items():
          if name not in new_state_dic:
            print(f"Parameter {name} not in model")
            continue
          if new_state_dic[name].shape != param.shape:
            print(f"Parameter {name} not in model")
            continue
          new_state_dic[name].copy_(param)  

        model.load_state_dict(new_state_dic)
        config['global_epoch'] = config['current_epoch']
        
        config['current_epoch'] = 0
      else:
        config['global_epoch'] = 0  

    else:
      print("No model found")
      return None
    
    
    config['MACs'], config['num_param'] = get_MMac(model, batch_size=batch_size,  seq_len=config['seq_len'], channels=channels)
    config['model_path'] = create_model_folder(config['model_num'], path=model_folder)
    results = count_parameters(model, verbose=False)
    config['encoder_param'] = results['encoder_param']
    config['input_param'] = results['src_embed_param']
    config['final_param'] = results['final_param']
    config['pos_param'] = results['buf_param']
    print(f"Number of paramters: {config['num_param']} input: {config['input_param']} encoder: {config['encoder_param']} final: {config['final_param']} pos: {config['pos_param']}")
    writer = SummaryWriter(config['model_path'] + '/trainingdata')
    print(f"Follow on tensorboard: python3 -m tensorboard.main --logdir={config['model_path']}trainingdata")
    #  python3 -m tensorboard.main --logdir=/mnt/md0/halin/Models/model_1/trainingdata
    
    loss_type = config.get('loss_function', 'BCE')
    if loss_type == 'BCE':
      criterion = nn.BCELoss().to(device)
    elif loss_type == 'BCEWithLogits':
      criterion = nn.BCEWithLogitsLoss().to(device)
    else:
      print("No loss function found")
      return None
   
    optimizer = torch.optim.Adam(model.parameters(), lr=config['learning_rate'])
    scheduler = ReduceLROnPlateau(optimizer=optimizer,
                                mode='min',
                                factor=config['decreas_factor'],
                                patience=4,
                                verbose=True)
  
    model.to(device)
    
    initial_epoch = 0
    

    early_stop_count = 0
    min_val_loss = float('inf')
    total_time = time.time()

    for epoch in range(initial_epoch, config['num_epochs']):

      epoch_time = time.time()

      config['current_epoch'] = epoch + 1
      
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

        print(f"Epoch {epoch + 1}/{config['num_epochs']} Batch {batch_num}/{num_of_bathes}", end="\r") 
  
        x_batch, y_batch = train_loader.__getitem__(istep)
        x_batch, y_batch = x_batch.to(device), y_batch.to(device)
        
        outputs = model.encode(x_batch, src_mask=None)

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
          x_batch, y_batch = x_batch.to(device), y_batch.to(device)
          outputs = model.encode(x_batch,src_mask=None)
          loss = criterion(outputs, y_batch.squeeze())
          if config['loss_function'] == 'BCEWithLogits':
            outputs = torch.sigmoid(outputs)
          pred = outputs.round()
          met = validate(y_batch.cpu().detach().numpy(), pred.cpu().detach().numpy(), config['metric'])
            
          val_loss.append(loss.item())
          metric.append(met)

      train_loss = np.mean(train_loss)
      val_loss = np.mean(val_loss)    
      metric = np.mean(metric)

      current_lr = optimizer.state_dict()['param_groups'][0]['lr']
      print(f"Learning rate: {current_lr}", end=" ")
      scheduler.step(metrics=val_loss)

      #############################################
      # Data saving
      #############################################
      
      temp_df = pd.DataFrame([[train_loss, val_loss, metric, epoch, current_lr]], 
                            columns= ['Train_loss', 'Val_loss', 'metric', 'Epochs', 'lr'])
      df = pd.concat([df, temp_df], ignore_index=True)
      # TODO maybe use best_val_loss instead of best_accuracy
      if val_loss < min_val_loss:
        save_model(model, optimizer, config, epoch)
      save_data(config, df)

      ############################################
      #  Tensorboard
      ############################################
      writer.add_scalar('Training Loss' , train_loss, epoch)
      writer.add_scalar('Validation Loss' , val_loss, epoch)
      writer.add_scalar(config['metric'], metric, epoch)
      writer.flush()

      print(f"Training Loss: {train_loss:.4f}, Validation Loss: {val_loss:.4f}, Val {config['metric']}: {metric:.6f}, Time: {time.time() - epoch_time:.2f} s")

      #############################################
      # Early stopping
      #############################################
      if val_loss < min_val_loss:
        min_val_loss = val_loss
        early_stop_count = 0
      else:
        early_stop_count += 1
      if early_stop_count >= config['early_stop']:
        print("Early stopping!")
        break
      
      config['global_epoch'] += 1
      config['power'] = ((config['current_epoch'])*config['power'] + get_energy(cuda_device))/(config['current_epoch'] + 1)

    ###########################################
    # Training done                           #
    ###########################################  
    writer.close()   
    total_training_time = time.time() - total_time   
    print(f"Total time: {total_training_time} s")
    config['trained'] = True
    config['training_time'] = total_training_time
    config['energy'] = config['power']*total_training_time

    y_pred_data, config['Accuracy'], config['Efficiency'], config['Precission'] = test_model(model=model, 
                                                                                             test_loader=test_loader,
                                                                                             device=device, 
                                                                                             config=config)    
    print(f"Test efficiency: {config['Efficiency']:.4f}")

    histogram(y_pred_data['y_pred'], y_pred_data['y'], config)
    nr_area, nse = plot_performance_curve([y_pred_data['y_pred']], [y_pred_data['y']], [config], curve='nr', x_lim=[0,1], bins=10000)
    config['nr_area'] = nr_area
    config['NSE_AT_10KNRF'] = nse
    roc_area, _ = plot_performance_curve([y_pred_data['y_pred']], [y_pred_data['y']], [config], curve='roc', bins=10000)
    config['roc_area'] = roc_area
    plot_results(config['model_num'], config)


    x_batch, y_batch = train_loader.__getitem__(0)
    data = x_batch.cpu().detach().numpy()
      
    plot_examples(data, config=config)
    plot_performance(config, device, x_batch=x_batch, y_batch=y_batch, lim_value=0.5, )
    save_data(config, df, y_pred_data)
