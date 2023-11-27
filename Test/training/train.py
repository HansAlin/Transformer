import numpy as np
import torch
import os
#import matplotlib.pyplot as plt

def training(model,
          train_loader,
          test_loader,
          device,
          optimizer,
          criterion,
          
          epochs = 100,
          early_stop_count = 0,
          min_val_loss = float('inf'),
        ):
  val_losses = []
  train_losses = []
  val_accs = []

  for epoch in range(epochs):
    # set the model in training mode
    model.train()
    train_loss = []
    val_loss = []
    val_acc = []

    # Training
    for batch in train_loader:
      x_batch, y_batch = batch
      if device != None:
        x_batch, y_batch = x_batch.to(device), y_batch.to(device)
      optimizer.zero_grad()
      outputs = model(x_batch)
      loss = criterion(outputs, y_batch)
      train_loss.append(loss.item()) 
      loss.backward()
      optimizer.step()

    train_loss = np.mean(train_loss)
    
    # validation
    # set model in evaluation mode
    # This part should not be part of the model thats the reason for .no_grad()
    model.eval()
    with torch.no_grad():
      
      for batch in test_loader:
        x_batch, y_batch = batch
        if device != None:
          x_batch, y_batch = x_batch.to(device), y_batch.to(device)
        outputs = model(x_batch)
        loss = criterion(outputs, y_batch)
        val_loss.append(loss.item())   

        pred = outputs.round()
        acc = pred.eq(y_batch).sum() / float(y_batch.shape[0])
        acc = acc.cpu()
        val_acc.append(acc)


    val_loss = np.mean(val_loss)    
    val_acc = np.mean(val_acc)
    
    train_losses.append(train_loss)
    val_losses.append(val_loss)
    val_accs.append(val_acc)
          
    
    # scheduler.step(val_loss)

    print(f"Epoch {epoch + 1}/{epochs}, Training Loss: {train_loss}, Validation Loss: {val_loss:.4f}, Val acc: {acc:.2f}")
  train_length = range(1,len(train_losses) + 1)
  return (model, train_length, train_losses, val_losses, val_accs)

def save_data(trained_model, path='', model_number=999):
  if path == '':
    path = os.getcwd + f'/Test/ModelsResults/model_{model_number}/'

  # TODO have to save the model as well!!
  torch.save(trained_model[0].state_dict(), path)

  with open(path + 'training_values', 'wb') as f:
    np.save(f, np.array(trained_model[1]))
    np.save(f, np.array(trained_model[2]))
    np.save(f, np.array(trained_model[3]))
    np.save(f, np.array(trained_model[4]))  

def plot_results(model_number, path=''):
  if path == '':
    path = os.getcwd + f'/Test/ModelsResults/model_{model_number}/'

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