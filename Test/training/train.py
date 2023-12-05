import numpy as np
import torch
import torch.nn as nn
import os
import matplotlib.pyplot as plt
from dataHandler.datahandler import get_data, prepare_data
from config.config import getweights_file_path
import tqdm as tqdm
from torch.utils.tensorboard import SummaryWriter
from torch.optim.lr_scheduler import ReduceLROnPlateau
import pandas as pd


from models.models_1 import build_encoder_transformer, get_n_params, set_max_split_size_mb 
from dataHandler.datahandler import save_data, save_model, create_model_folder


def training(config, data_path):
  df = pd.DataFrame([], 
                           columns= ['Train_loss', 'Val_loss', 'Val_acc', 'Epochs'])
  device = torch.device("cuda" if torch.cuda.is_available() else "cpu")           
  print(f"Using device: {device}")
  if config['model_name'] == "base_encoder":
    model = build_encoder_transformer(embed_size=config['embed_size'], 
                                      seq_len=config['seq_len'], 
                                      d_model=config['d_model'], 
                                      N=config['N'], 
                                      h=config['h'], 
                                      dropout=config['dropout'])
  else:
    print("No model found")
    return None
  
  model.to(device)
  
  config['num_parms'] = get_n_params(model)
  config['model_path'] = create_model_folder(config['model_num'])
  print(f"Number of paramters: {config['num_parms']}") 
  writer = SummaryWriter(config['model_path']+'/trainingdata')
  writer.add_graph(model, images)
  writer.close()

  os.system('tensorboard --logdir=' + config['model_path']+'/trainingdata' )

  
  
  print(f"Number of paramters: {config['num_parms']}")
  criterion = nn.BCELoss().to(device)
  optimizer = torch.optim.Adam(model.parameters(), lr=config['learning_rate'])
  scheduler = ReduceLROnPlateau(optimizer=optimizer,
                               mode='max',
                               factor=config['decreas_factor'],
                               patience=5,
                               verbose=True)
  
  x_train, x_test, y_train, y_test = get_data(path=data_path)

  train_loader, test_loader = prepare_data(x_train, x_test, y_train, y_test, config['batch_size'])

  # TODO can I use this
  
  # writer.add_graph(model.encode, torch.tensor([32, 100, 1]))
  initial_epoch = 0
  global_step = 0

  val_losses = []
  train_losses = []
  val_accs = []
  best_accuracy = 0
  for epoch in range(initial_epoch, config['num_epochs']):
    config['current_epoch'] = epoch
    #print(f"Epoch {epoch + 1}/{config['num_epochs']}, Batch: ", end="             ")
    # set the model in training mode
    model.train()

    
    train_loss = []
    val_loss = []
    val_acc = []
    # Training
    batch_num = 1
    num_of_bathes = len(train_loader.dataset)
    min_val_loss = float('inf')

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
      
      for batch in test_loader:
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
      save_model(model, config, df)
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


  #config['epochs'] = range(1,len(train_losses) + 1)
  writer.close()
  # TODO save model
  # save_model_data(trained_model=model,
  #                 config=config)
  # TODO save config
  # TODO save training values
 


def plot_results(model_number, path=''):
  if path == '':
    path = os.getcwd() + f'/Test/ModelsResults/model_{model_number}/'

  df = pd.read_pickle(path + 'dataframe.pkl')

  # Loss plot 
  plot_path = path + f'plot/' 
  isExist = os.path.exists(plot_path)
  if not isExist:
    os.makedirs(plot_path)
    print("The new directory is created!")
  loss_path = plot_path + f'model_{model_number}_loss_plot.png'  
  df.plot(x='Epochs', y=['Train_loss','Val_loss'], kind='line', figsize=(7,7))
  
  plt.title("Loss")
  plt.legend()
  plt.savefig(loss_path)
  plt.cla()
  plt.clf()
  # Accuracy plot
  acc_path = plot_path + f'model_{model_number}_acc_plot.png'
  df.plot('Epochs', 'Val_acc', label='Accuracy')
  plt.title("Accuracy")
  plt.ylim([0,1])
  plt.legend()
  plt.savefig(acc_path)
  plt.cla()
  plt.clf()
