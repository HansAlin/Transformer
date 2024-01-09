import torch 
import numpy as np
from tqdm import tqdm
import pandas as pd

def test_model(model, test_loader, device, config):
  """
  This function tests the model on the test set
  and returns the accuracy and true positive rate
  true negative rate, false positive rate and false negative rate
  Arg:
    model : the trained model
    test_loader : the test data loader
    device : the device to use
    config: config file for the model
  Return:
    y_pred_data, accuracy, efficiency, precission
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
        if config['loss_function'] == 'BCEWithLogits':
          outputs = torch.sigmoid(outputs)
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
        if config['loss_function'] == 'BCEWithLogits':
          outputs = torch.sigmoid(outputs)
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
  
  if np.count_nonzero(y) != 0:
    efficiency = TP / np.count_nonzero(y)
  else:
    efficiency = 0  

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
    metric = TP / np.count_nonzero(y) if np.count_nonzero(y) != 0 else 0
  elif metric == 'Precision':
    metric = TP / (TP + FP) if TP + FP != 0 else 0 


  return metric