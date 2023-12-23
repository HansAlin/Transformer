import numpy as np
import torch
import torch.nn as nn
import os
import matplotlib.pyplot as plt
from dataHandler.datahandler import get_data, prepare_data, get_test_data
from config.config import getweights_file_path
from plots.plots import histogram, noise_reduction_factor, plot_results, plot_examples
import tqdm as tqdm
from torch.utils.tensorboard import SummaryWriter
from torch.optim.lr_scheduler import ReduceLROnPlateau
import pandas as pd
import sys
import time
from tqdm import tqdm

from models.models_1 import build_encoder_transformer, get_n_params, set_max_split_size_mb 
from dataHandler.datahandler import save_data, save_model, create_model_folder



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
                            columns= ['Train_loss', 'Val_loss', 'metric', 'Epochs'])
    
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

    
    print(f"Number of paramters: {config['num_parms']}")
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

      
      print(f"Learning rate: {optimizer.state_dict()['param_groups'][0]['lr']}", end=" ")
      scheduler.step(metrics=val_loss)

      #############################################
      # Data saving
      #############################################
      
      temp_df = pd.DataFrame([[train_loss, val_loss, metric, epoch]], 
                            columns= ['Train_loss', 'Val_loss', 'metric', 'Epochs'])
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
    print(f"Total time: {time.time() - total_time:.2f} s")
    config['pre_trained'] = True
    y_pred_data, config['Accuracy'], config['Efficiency'], config['Precission'] = test_model(model=model, 
                                                                                             test_loader=test_loader,
                                                                                             device=device, 
                                                                                             config=config)    
    print(f"Test efficiency: {config['Efficiency']:.4f}")
    save_data(config, df, y_pred_data)

    histogram(y_pred_data['y_pred'], y_pred_data['y'], config)
    noise_reduction_factor([y_pred_data['y_pred']], [y_pred_data['y']], [config])
    plot_results(config['model_num'], config)

    if isinstance(train_loader, torch.utils.data.DataLoader):
      dataiter = iter(train_loader)
      x_batch, y_batch = dataiter.next()
      data = x_batch.cpu().detach().numpy()
    else:
      x_batch, y_batch = train_loader.__getitem__(0)
      data = x_batch.cpu().detach().numpy()
    plot_examples(data, config=config)

 

def test_model(model, test_loader, device, config):
  """
  This function tests the model on the test set
  and returns the accuracy and true positive rate
  true negative rate, false positive rate and false negative rate
  Arg:
    model : the trained model
    test_loader : the test data loader
    device : the device to use
  Return:
    arr : (acc, TP, TN, FP, FN ) 
  """
  model_path = config['model_path'] + 'saved_model' + f'/model_{config["model_num"]}.pth'
 
  print(f'Preloading model {model_path}')
  state = torch.load(model_path)
  model.load_state_dict(state['model_state_dict'])

  model.to(device)
  model.eval()
  acc = 0
  TP = 0
  TN = 0
  FP = 0
  FN = 0
  count = 0
  y = []
  y_pred = []
  pred_round = []
  with torch.no_grad():
    
    if isinstance(test_loader, torch.utils.data.DataLoader):
      print("train_loader is a DataLoader")
      for batch in test_loader:
        x_test, y_test = batch
        y_test = y_test.squeeze() 
        outputs = model.encode(x_test,src_mask=None)
        y_pred.append(outputs.cpu().detach().numpy())
        y.append(y_test.cpu().detach().numpy()) 
        pred_round.append(outputs.cpu().detach().numpy().round())
    else:
      print("train_loader is not a DataLoader")
      for istep in tqdm(range(len(test_loader))):

        x_test, y_test = test_loader.__getitem__(istep)
        x_test, y_test = x_test.to(device), y_test.to(device)
        y_test = y_test.squeeze() 
        outputs = model.encode(x_test,src_mask=None)
        y_pred.append(outputs.cpu().detach().numpy())
        y.append(y_test.cpu().detach().numpy()) 
        pred_round.append(outputs.cpu().detach().numpy().round())

      
          





  y = np.asarray(y).flatten()
  y_pred = np.asarray(y_pred).flatten()
  pred_round = np.asarray(pred_round).flatten()
  true_signal = np.logical_and(y == 1, pred_round == 1)
  true_noise = np.logical_and( y == 0, pred_round == 0)
  false_signal = np.logical_and(y == 0, pred_round == 1)
  false_noise = np.logical_and(y == 1, pred_round == 0)

  TP += np.sum(true_signal)
  TN += np.sum(true_noise)
  FP += np.sum(false_signal)
  FN += np.sum(false_noise)
  accuracy = (TP + TN) / len(y)
  
  efficiency = TP / np.count_nonzero(y)

  if TP + FP == 0:
    precission = 0
  else:  
    precission = TP / (TP + FP) 
 
  y_pred_data = pd.DataFrame({'y_pred': y_pred, 'y': y})

  return y_pred_data, accuracy, efficiency, precission

def validate(y, y_pred, metric='Accuracy'):
  y = y.flatten()
  y_pred = y_pred.flatten()
  TP = np.sum(np.logical_and(y == 1, y_pred == 1))
  TN = np.sum(np.logical_and(y == 0, y_pred == 0))
  FP = np.sum(np.logical_and(y == 0, y_pred == 1))
  FN = np.sum(np.logical_and(y == 1, y_pred == 0))
  
  if metric == 'Accuracy':
    metric = (TP + TN) / len(y)
  elif metric == 'Efficiency':
    metric = TP / np.count_nonzero(y)
  elif metric == 'Precision':
    metric = TP / (TP + FP)  


  return metric
