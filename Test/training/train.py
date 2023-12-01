import numpy as np
import torch
import torch.nn as nn
import os
import matplotlib.pyplot as plt
from dataHandler.datahandler import get_test_data, prepare_data
from config.config import getweights_file_path
import tqdm as tqdm
from torch.utils.tensorboard import SummaryWriter

from models.models_1 import build_encoder_transformer, get_n_params
from dataHandler.datahandler import save_model_data


def training(config):

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
  print(f"Number of paramters: {config['num_parms']}") 
  
  

  # print(f"Number of paramters: {model.get_n_parms(model)}")
  criterion = nn.BCELoss().to(device)
  optimizer = torch.optim.Adam(model.parameters(), lr=config['learning_rate'])


  x_train, x_test, y_train, y_test = get_test_data(path=config['data_path'])
  train_loader, test_loader = prepare_data(x_train, x_test, y_train, y_test, config['batch_size'])

  # TODO can I use this
  #writer = SummaryWriter(config['experiment_name'])

  initial_epoch = 0
  global_step = 0

  val_losses = []
  train_losses = []
  val_accs = []

  for epoch in range(initial_epoch, config['num_epochs']):
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
      
      # Only for umar_jamil.py
      
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

    config['train_loss'].append(train_loss)
    config['val_loss'].append(val_loss)
    config['val_acc'].append(val_acc)

    print(f"Epoch {epoch + 1}/{config['num_epochs']}, Training Loss: {train_loss:.4f}, Validation Loss: {val_loss:.4f}, Val acc: {val_acc:.6f}")
  config['epochs'] = range(1,len(train_losses) + 1)

  # TODO save model
  save_model_data(trained_model=model,
                  config=config)
  # TODO save config
  # TODO save training values
 


def plot_results(model_number, path=''):
  if path == '':
    path = os.getcwd() + f'/Test/ModelsResults/model_{model_number}/'

  with open(path + 'training_values.npy', 'rb') as f:
    epochs = np.load(f)
    train_loss = np.load(f)
    val_loss = np.load(f)
    val_acc = np.load(f)

  # Loss plot 
  loss_path = path + f'model_{model_number}_loss_plot.png'
  plt.plot(epochs, train_loss, label='Training') 
  plt.plot(epochs, val_loss, label='Validation')
  plt.legend()
  plt.savefig(loss_path)

  # Accuracy plot
  acc_path = path + f'model_{model_number}_acc_plot.png'
  plt.plot(epochs, val_acc, label='Accuracy')
  plt.legend()
  plt.savefig(acc_path)  