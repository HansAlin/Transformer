import numpy as np
import torch
import torch.nn as nn
import os
import matplotlib.pyplot as plt
from dataHandler.datahandler import get_test_data, prepare_data
from config.config import getweights_file_path
import tqdm as tqdm
from torch.utils.tensorboard import SummaryWriter

def training(model, config):

  device = torch.device("cuda" if torch.cuda.is_available() else "cpu")           
  print(f"Using device: {device}")
  model = model.to(device)
  criterion = nn.BCELoss().to(device)
  optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
  print("test")

  x_train, x_test, y_train, y_test = get_test_data(path='/home/halin/Master/Transformer/Test/data/test_data.npy')
  train_loader, test_loader = prepare_data(x_train, x_test, y_train, y_test, config['batch_size'])


  writer = SummaryWriter(config['experiment_name'])

  initial_epoch = 0
  global_step = 0


  for epoch in range(initial_epoch, config['num_epochs']):
    # set the model in training mode
    model.train()

    # Training
    for batch in train_loader:
      x_batch, y_batch = batch
      x_batch, y_batch = x_batch.to(device), y_batch.to(device)
      
      # Only for umar_jamil.py
      src_mask = torch.ones(100,1).to(device)
      outputs = model.encode(x_batch, src_mask)
      loss = criterion(outputs, y_batch)

      # Log the loss
      writer.add_scalar("Train loss", loss.item())  

      # Backpropagation
      loss.backward()

      # Update the weights
      optimizer.step()
      optimizer.zero_grad()

    
    writer.flush()
    # validation
    # set model in evaluation mode
    # This part should not be part of the model thats the reason for .no_grad()
    model.eval()
    with torch.no_grad():
      
      for batch in test_loader:
        x_batch, y_batch = batch
        x_batch, y_batch = x_batch.to(device), y_batch.to(device)
        outputs = model.encode(x_batch,src_mask)
        loss = criterion(outputs, y_batch)

        # Log the loss
        writer.add_scalar("Train loss", loss.item())  



  #   print(f"Epoch {epoch + 1}/{epochs}, Training Loss: {train_loss}, Validation Loss: {val_loss:.4f}, Val acc: {acc:.2f}")
  # train_length = range(1,len(train_losses) + 1)
  # return (model, train_length, train_losses, val_losses, val_accs)

def save_data(trained_model, path='', model_number=999):
  if path == '':
    path = os.getcwd() 
    path += f'/Test/ModelsResults/model_{model_number}/' 

    isExist = os.path.exists(path)
    if not isExist:
      os.makedirs(path)
      print("The new directory is created!")

  # TODO have to save the model as well!!
  torch.save(trained_model[0].state_dict(), path + f'model_{model_number}.pth')

  with open(path + 'training_values.npy', 'wb') as f:
    np.save(f, np.array(trained_model[1]))
    np.save(f, np.array(trained_model[2]))
    np.save(f, np.array(trained_model[3]))
    np.save(f, np.array(trained_model[4]))  

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