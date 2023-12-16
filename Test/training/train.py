import numpy as np
import torch
import torch.nn as nn
import os
import matplotlib.pyplot as plt
from dataHandler.datahandler import get_data, prepare_data
from config.config import getweights_file_path
from plots.plots import histogram, noise_reduction_factor, plot_results
import tqdm as tqdm
from torch.utils.tensorboard import SummaryWriter
from torch.optim.lr_scheduler import ReduceLROnPlateau
import pandas as pd
import sys

from models.models_1 import build_encoder_transformer, get_n_params, set_max_split_size_mb 
from dataHandler.datahandler import save_data, save_model, create_model_folder


def training(config, data_path):
  df = pd.DataFrame([], 
                           columns= ['Train_loss', 'Val_loss', 'Val_acc', 'Epochs'])
  device = torch.device("cuda" if torch.cuda.is_available() else "cpu")           
  print(f"Using device: {device}, name of GPU: {torch.cuda.get_device_name(device=device)}")
  if config['model_type'] == "base_encoder":
    model = build_encoder_transformer(config,
                                      embed_size=config['embed_size'], 
                                      seq_len=config['seq_len'], 
                                      d_model=config['d_model'], 
                                      N=config['N'], 
                                      h=config['h'], 
                                      dropout=config['dropout'],
                                      omega=config['omega'])
  else:
    print("No model found")
    return None
  
  
  config['num_parms'] = get_n_params(model)
  config['model_path'] = create_model_folder(config['model_num'])
  
  print(f"Number of paramters: {config['num_parms']}") 
  writer = SummaryWriter(config['model_path']+'/trainingdata')
  print(f"Follow on tensorboard: python3 -m tensorboard.main --logdir=Test/ModelsResults/model_{config['model_num']}/trainingdata")

  
  print(f"Number of paramters: {config['num_parms']}")
  criterion = nn.BCELoss().to(device)
  optimizer = torch.optim.Adam(model.parameters(), lr=config['learning_rate'])
  scheduler = ReduceLROnPlateau(optimizer=optimizer,
                               mode='max',
                               factor=config['decreas_factor'],
                               patience=5,
                               verbose=True)
  
  x_train, x_test, x_val, y_train, y_val, y_test = get_data(path=data_path)

  train_loader, val_loader, test_loader, config['trained_noise'], config['trained_signal'] = prepare_data(x_train, x_val, x_test, y_train, y_val, y_test, config['batch_size'])

  #########################################################################
  #  Visulizing graphs doesn't work satisfactory                          #
  #########################################################################
  # writer.add_graph(model=model.encoder, input_to_model=torch.ones(32,100,1))
  # writer.close()
  # sys.exit()

  # print(model)



  model.to(device)
  
  initial_epoch = 0
  global_step = 0

  
  best_accuracy = 0
  min_val_loss = float('inf')
  for epoch in range(initial_epoch, config['num_epochs']):
    config['current_epoch'] = epoch + 1
    #print(f"Epoch {epoch + 1}/{config['num_epochs']}, Batch: ", end="             ")
    # set the model in training mode
    model.train()

    
    train_loss = []
    val_loss = []
    val_acc = []
    # Training
    batch_num = 1
    num_of_bathes = len(train_loader.dataset)
    

    for batch in train_loader:
      #print(f"{batch_num}/{num_of_bathes}")
      x_batch, y_batch = batch
      x_batch, y_batch = x_batch.to(device), y_batch.to(device)
     
      
      outputs = model.encode(x_batch, src_mask=None)

      loss = criterion(outputs, y_batch)

      # Backpropagation
      loss.backward()

      # Update the weights
      optimizer.step()
      optimizer.zero_grad()

      train_loss.append(loss.item())
    
    
    # writer.flush()
    # validation
    # set model in evaluation mode
    # This part should not be part of the model thats the reason for .no_grad()
    model.eval()
    with torch.no_grad():
      
      for batch in val_loader:
        x_batch, y_batch = batch
        x_batch, y_batch = x_batch.to(device), y_batch.to(device)
        outputs = model.encode(x_batch,src_mask=None)
        loss = criterion(outputs, y_batch)

        # TODO put in function
        pred = outputs.round()
        acc = pred.eq(y_batch).sum() / float(y_batch.shape[0])
        
        

        # Log the loss
        # writer.add_scalar("Val loss", loss.item())  
        val_loss.append(loss.item())
        val_acc.append(acc.item())

    train_loss = np.mean(train_loss)
    val_loss = np.mean(val_loss)    
    val_acc = np.mean(val_acc)

    scheduler.step(metrics=val_acc)
    print(f"Learning rate: {optimizer.state_dict()['param_groups'][0]['lr']}")

    #############################################
    # Data saving
    #############################################
    
    temp_df = pd.DataFrame([[train_loss, val_loss, val_acc, epoch]], 
                           columns= ['Train_loss', 'Val_loss', 'Val_acc', 'Epochs'])
    df = pd.concat([df, temp_df], ignore_index=True)
    # TODO maybe use best_val_loss instead of best_accuracy
    if val_acc > best_accuracy:
      save_model(model, optimizer, config, global_step)
      best_accuracy = val_acc
    save_data(config, df)

    ############################################
    #  Tensorboard
    ############################################
    writer.add_scalar('Training Loss' , train_loss, epoch)
    writer.add_scalar('Validation Loss' , val_loss, epoch)
    writer.add_scalar('Accuracy', val_acc, epoch)
    writer.flush()

    print(f"Epoch {epoch + 1}/{config['num_epochs']}, Training Loss: {train_loss:.4f}, Validation Loss: {val_loss:.4f}, Val acc: {val_acc:.6f}")

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

  config['pre_trained'] = True
  (config['acc'], config['TP'], config['TN'], config['FP'], config['FN']) = test_model(model, test_loader, device, config)    
  print(f"Test acc: {config['acc']:.4f}")
  save_data(config, df)


  writer.close()

 

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
  count = 1
  y = []
  y_pred = []
  with torch.no_grad():
    for batch in test_loader:
      x_test, y_test = batch
      x_test, y_test = x_test.to(device), y_test.to(device)
      y.append(y_test.detach().item())  
      outputs = model.encode(x_test,src_mask=None)
      y_pred.append(outputs.detach().item())

      pred = outputs.round()
      if pred == y_test:
        acc += 1
        if pred == 1:
          TP += 1
        else:
          TN += 1
      if pred != y_test:
        if pred == 1:
          FP += 1
        else:
          FN += 1      
      count += 1

  
  y = np.asarray(y)  
  y_pred = np.asarray(y_pred)
  # TODO save y and y_pred
  np.save
  arr = (acc/count, TP/(TP + FN), TN/(TN + FP), FP/(FP + TN), FN/(FN+TP))


  histogram(y_pred, y, config)
  noise_reduction_factor([y_pred], y, config)
  
  return arr 
