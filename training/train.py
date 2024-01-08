import numpy as np
import torch
import torch.nn as nn
import os
import matplotlib.pyplot as plt
from dataHandler.datahandler import get_data, prepare_data, get_test_data

from plots.plots import histogram, plot_performance_curve, plot_results, plot_examples, plot_performance
import tqdm as tqdm
from torch.utils.tensorboard import SummaryWriter
from torch.optim.lr_scheduler import ReduceLROnPlateau
import pandas as pd
import sys
import time
from tqdm import tqdm

from models.models import build_encoder_transformer, get_n_params, set_max_split_size_mb 
from dataHandler.datahandler import save_data, save_model, create_model_folder
from evaluate.evaluate import test_model, validate



def training(configs, data_path, batch_size=32, channels=4, save_folder='', test=False):
  if data_path == '':
    train_loader, val_loader, test_loader = get_data(batch_size=batch_size, seq_len=configs[0]['seq_len'], subset=test)
  else:  
    x_train, x_test, x_val, y_train, y_val, y_test = get_test_data(path=data_path)
    if channels == 1:
      train_loader, val_loader, test_loader, config['trained_noise'], config['trained_signal'] = prepare_data(x_train, x_val, x_test, y_train, y_val, y_test, batch_size)
    else:
      train_loader, val_loader, test_loader = prepare_data(x_train, x_val, x_test, y_train, y_val, y_test, batch_size, multi_channel=True)
    
    del x_train
    del x_test
    del x_val
    del y_train
    del y_test
    del y_val


  for config in configs:
    df = pd.DataFrame([], 
                            columns= ['Train_loss', 'Val_loss', 'metric', 'Epochs', 'lr'])
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")           
    print(f"Using device: {device}, name of GPU: {torch.cuda.get_device_name(device=device)}")
  
    if config['model_type'] == "base_encoder":
      model = build_encoder_transformer(config)
    else:
      print("No model found")
      return None
    
    
    config['num_parms'] = get_n_params(model)
    config['model_path'] = create_model_folder(config['model_num'], path=save_folder)
    
    print(f"Number of paramters: {config['num_parms']}") 
    writer = SummaryWriter(config['model_path'] + '/trainingdata')
    print(f"Follow on tensorboard: python3 -m tensorboard.main --logdir={config['model_path']}trainingdata")
    #  python3 -m tensorboard.main --logdir=/mnt/md0/halin/Models/model_1/trainingdata
    
    criterion = nn.BCELoss().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config['learning_rate'])
    scheduler = ReduceLROnPlateau(optimizer=optimizer,
                                mode='min',
                                factor=config['decreas_factor'],
                                patience=4,
                                verbose=True)
    


    model.to(device)
    
    initial_epoch = 0
    global_step = 0

    
    best_accuracy = 0
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
      if isinstance(train_loader, torch.utils.data.DataLoader):
        #############################################
        # Torch Dataloader                          #
        #############################################
        print("train_loader is a DataLoader")
        num_of_bathes = int(len(train_loader.dataset)/batch_size)
        for batch in test_loader:
          print(f"Epoch {epoch + 1}/{config['num_epochs']} Batch {batch_num}/{num_of_bathes}", end="\r")
          x_batch, y_batch = batch
          x_batch, y_batch = x_batch.to(device), y_batch.to(device)
          y_test = y_test.squeeze() 

          outputs = model.encode(x_batch, src_mask=None)

          loss = criterion(outputs, y_batch.squeeze())

          loss.backward()

          optimizer.step()
          optimizer.zero_grad()

          train_loss.append(loss.item())
          batch_num += 1

      else:
        #############################################
        # Alan's Dataloader                         #
        #############################################
        print("train_loader is not a DataLoader")    
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

      print(f"\rEpoch {epoch + 1}/{config['num_epochs']} Done ", end="  ")
      
      #############################################
      # Validation                                #
      #############################################
      model.eval()
      with torch.no_grad():
        if isinstance(val_loader, torch.utils.data.DataLoader):
          print("train_loader is a DataLoader")
          for batch in val_loader:
            x_batch, y_batch = batch
            x_batch, y_batch = x_batch.to(device), y_batch.to(device)
            outputs = model.encode(x_batch,src_mask=None)
            loss = criterion(outputs, y_batch.squeeze())
            
            pred = outputs.round()
            

        else:
            print("train_loader is not a DataLoader")
       
            for istep in tqdm(range(len(val_loader))):

              x_batch, y_batch = val_loader.__getitem__(istep)
              x_batch, y_batch = x_batch.to(device), y_batch.to(device)
              outputs = model.encode(x_batch,src_mask=None)
              loss = criterion(outputs, y_batch.squeeze())

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
        save_model(model, optimizer, config, global_step)
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
      
      global_step += 1

    ###########################################
    # Training done                           #
    ###########################################  
    writer.close()   
    total_training_time = time.time() - total_time   
    print(f"Total time: {total_training_time} s")
    config['pre_trained'] = True
    config['training_time'] = total_training_time
    y_pred_data, config['Accuracy'], config['Efficiency'], config['Precission'] = test_model(model=model, 
                                                                                             test_loader=test_loader,
                                                                                             device=device, 
                                                                                             config=config)    
    print(f"Test efficiency: {config['Efficiency']:.4f}")
    save_data(config, df, y_pred_data)

    histogram(y_pred_data['y_pred'], y_pred_data['y'], config)
    plot_performance_curve([y_pred_data['y_pred']], [y_pred_data['y']], [config], curve='nr', x_lim=[0,1])
    plot_performance_curve([y_pred_data['y_pred']], [y_pred_data['y']], [config], curve='roc')
    plot_results(config['model_num'], config)

    if isinstance(train_loader, torch.utils.data.DataLoader):
      dataiter = iter(train_loader)
      x_batch, y_batch = dataiter.next()
      data = x_batch.cpu().detach().numpy()
    else:
      x_batch, y_batch = train_loader.__getitem__(0)
      data = x_batch.cpu().detach().numpy()
    plot_examples(data, config=config)
    plot_performance(config['model_num'], x_batch=x_batch, y_batch=y_batch, lim_value=0.5, )

