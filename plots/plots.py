import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import pickle
import torch

from models.models import build_encoder_transformer

def histogram(y_pred, y, config, bins=100, save_path=''):
    """
          This function plots the histogram of the predictions of a given model.
          Args:
              y_pred (array like): array of y_pred
              y (array like): array of y
              config (dict): config dict of the model
              bins (int): number of bins in the histogram
              savefig_path (str): path to save the plot, optional
    """

    if save_path == '':
      save_path = config['model_path'] + 'plot/' 
      isExist = os.path.exists(save_path)
      if not isExist:
        os.makedirs(save_path)
        print("The new directory is created!")


    plt.rcParams['text.usetex'] = True
    fig, ax = plt.subplots()
    signal_mask = y == 1
    y_pred_signal = y_pred[signal_mask]
    y_pred_noise = y_pred[~signal_mask]
    signal_wights = np.ones_like(y_pred_signal) / len(y_pred_signal)
    noise_weights = np.ones_like(y_pred_noise) / len(y_pred_noise)

    ax.set_title(f"Model {config['model_num']}")
    ax.hist(y_pred_signal, bins=bins, label='Pred signal', weights=signal_wights)
    ax.hist(y_pred_noise, bins=bins, label='Pred noise', weights=noise_weights)
    ax.set_yscale('log')
    ax.legend()
    ax.set_xlabel(r'noise $\leftrightarrow$ signal')
    plt.savefig(save_path + f"model_{config['model_num']}_histogram.png")
    plt.clf()



def plot_performance_curve(y_preds, ys, configs, bins=100, save_path='', labels=None, x_lim=[0.8,1], window_pred=False, curve='roc'):
    """
            This function plots the noise reduction factor curve for a given model or models. Note that
            the y_pred, y and config must be a list of arrays and dicts respectively.
            If labels is not None, they also have to be in a list. 
            Args:
                y_preds (list of np.arrays): list of y_pred arrays
                ys (list of np.arrays): list of y arrays
                configs (list of dicts): list of config dicts 
                bins (int): number of bins in the histogram
                savefig_path (str): path to save the plot, optional
    """
    if window_pred:
      text = ' per window'
    else:
      text = ' per timestep'  
    fig, ax = plt.subplots()
    length = len(y_preds)

    if save_path == '' and length == 0:
      save_path = config['model_path'] + 'plot/' 
      isExist = os.path.exists(save_path)
      if not isExist:
        os.makedirs(save_path)
        print("The new directory is created!")

    if labels == None:
      ax.set_title(f"Model {configs[0]['model_num']}{text}")
    else:
        if curve == 'roc':
          ax.set_title(f"ROC curve for {length} models,{text}") 
        elif curve == 'nr':  
          ax.set_title(f"Noise reduction factor for {length} models,{text}")
      
    nr_y_noise = 0    

    for i in range(length): 
        y_pred = y_preds[i]
        y = ys[i]
        config = configs[i]
        if window_pred:
          seq_len = config['seq_len']
          y_pred = np.asarray(y_pred)
          y = np.asarray(y)
          y_pred = np.reshape(y_pred, (-1, seq_len))
          
          y = np.reshape(y, (-1, seq_len))
          y_window = np.count_nonzero(y==1, axis=-1)
          y = np.where(y_window > 0, 1, y_window)
          nr_y_signal = np.count_nonzero(y==1)
          nr_y_noise = np.count_nonzero(y == 0)
        else:
          nr_y_signal = np.count_nonzero(y==1)
          nr_y_noise = np.count_nonzero(y == 0)

        if curve == 'roc':
          y, x = get_roc(y, y_pred)
        elif curve == 'nr':  
          x, y = get_noise_reduction(y, y_pred, bins, window_pred=window_pred)

        if labels == None:
          ax.plot(x, y)   
        else:
            ax.plot(x, y, label=labels['hyper_parameters'][i])
        
            
    if labels != None:
       ax.legend(title=labels['name'])        
    
    if curve == 'roc':
      ax.set_xlabel('False positive rate')
      ax.set_ylabel('True positive rate')
    elif curve == 'nr':
      ax.set_xlabel('True positive rate') 
      ax.set_ylabel(f'Noise reduction factor (nr. noise {nr_y_noise})')
      ax.set_yscale('log')
      ax.set_xlim(x_lim)

    if save_path == '':
        save_path = config['model_path'] + 'plot/' + f'model_{config["model_num"]}_{curve}_{text.replace(" ", "_")}.png'
    plt.savefig(save_path)

def plot_results(model_number, config, path=''):
  if path == '':
    path = config['model_path']

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
  df.plot('Epochs', 'metric', label=config['metric'])
  plt.title("Metric")
  plt.ylim([0,1])
  plt.legend()
  plt.savefig(acc_path)
  plt.cla()
  plt.clf()

def plot_weights(model, config, save_path='', block='self_attention_block', quiet=True):
  """
    This function plots the weights of a given block in the model
    Args:
      model : the trained model
      config : the config file of the model
      save_path : the path to save the plot, optional
      block : options: 'self_attention_block', 'final_binary_block', 'embedding_block', 'feed_forward_block'
      quiet : if True, no print statements
  """
  if not quiet:
    for name, param in model.named_parameters():
      if 'weight' in name:
        print(f'Layer: {name}, Shape: {param.shape}')

  if save_path == '':
      save_path = config['model_path'] + 'plot/' 
      isExist = os.path.exists(save_path)
      if not isExist:
        os.makedirs(save_path)
        print("The new directory is created!")

    
  fig, ax = plt.subplots()
  if block == 'self_attention_block':
    weight = model.encoder.layers[config['h'] - 1].self_attention_block.W_0.weight.data.numpy()
    x = ax.imshow(weight, cmap='coolwarm', interpolation='nearest')
    cbar = ax.figure.colorbar(x, ax=ax)
  elif block == 'final_binary_block':
     weight = model.final_block.linear_1.weight.data.numpy()[0] 
     x =  np.arange(len(weight))
     ax.plot(x, weight)
  elif block == 'embedding_block':
     weight = model.src_embed.embedding.weight.data.numpy()
     ax.plot(range(len(weight)), weight)   
  elif block == 'feed_forward_block':
     weight = model.encoder.layers[config['h'] - 1].feed_forward_block.linear_2.weight.data.numpy()
     x = ax.imshow(weight, cmap='coolwarm', interpolation='nearest')
     cbar = ax.figure.colorbar(x, ax=ax)
  if not quiet:
    print(weight)
  # writer.add_image("weight_image",weight )

  
  plt.title(f'Layer: {block}  - Weights')
  
  if save_path == '':
    save_path = config['model_path'] + f'plot/model_{config["model_num"]}_{block}_weights.png'
  plt.savefig(save_path)
#   writer.add_histogram('weights', weight)
#   writer.close()  

def plot_collections(models, labels, save_path='', models_path='Test/ModelsResults/', x_lim=[0.8,1], window_pred=False, curve='roc'): 
  y_pred = []
  y = []
  configs = []

  for i, model_num in enumerate(models):
    config_path = models_path + f'model_{model_num}/config.txt'
    with open(config_path, 'rb') as f:
        config = pickle.load(f)
    y_data_path = models_path + f'model_{model_num}/y_pred_data.pkl'    
    with open(y_data_path, 'rb') as f:
        y_data = pickle.load(f)
    
    y_pred.append(np.asarray(y_data['y_pred']))
    y.append(np.asarray(y_data['y']))
    configs.append(config)
  if save_path == '':  
    model_name = models_path + f'collections/{labels["name"].replace(" ", "_")}_models'
    for model_num in models:
      model_name += f'_{model_num}'
    save_path =  model_name + '.png'  

  
  plot_performance_curve(ys=y, 
                        y_preds=y_pred, 
                        configs=configs, 
                        save_path=save_path, 
                        labels=labels,
                        x_lim=x_lim,
                        window_pred=window_pred,
                        curve=curve)


def plot_examples(data, config=None, save_path=''):

  if config != None:
    save_path = config['model_path'] + 'plot/examples.png'
  else:
    if save_path == '':
      save_path = os.getcwd() + 'examples.png'

  data = data[0]
  fig, ax = plt.subplots(4,1, figsize=(10,10))
  for i in range(4):
    ax[i].plot(data[:,i])
  plt.savefig(save_path)  

def plot_performance(model_num, data_path='/home/halin/Master/Transformer/Test/data/', model_path='/mnt/md0/halin/Models/', save_path=''):
  # TODO
  x_test = torch.load(data_path + 'example_x_data.pt')
  y_test = torch.load(data_path + 'example_y_data.pt')
  device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  
  MODEL_PATH = model_path + f'model_{model_num}/saved_model/model_{model_num}.pth'
  CONFIG_PATH = model_path + f'model_{model_num}/config.txt'
  with open(CONFIG_PATH, 'rb') as f:
    config = pickle.load(f)

  model = build_encoder_transformer(config)
  state = torch.load(MODEL_PATH)
  model.load_state_dict(state['model_state_dict'])

  model.to(device)
  x_test, y_test = x_test.to(device), y_test.to(device)
  model.eval()
  pred = model.encode(x_test,src_mask=None)
  index = 0

  pred = pred.cpu().detach().numpy()
  x_test = x_test.cpu().detach().numpy()
  y_test = y_test.cpu().detach().numpy()
  index = 0
  lim_value = 0.2
  count = 1
  for i in range(pred.shape[0]):
    pred_value = np.max(pred[i])
    y_value = np.max(y_test[i])
    if pred_value >= lim_value and y_value >= lim_value:
      
      index = i


      max_value = np.max(pred[index])  
      print(f"Max value: {max_value}")
      pred = pred[index]
      x_test = x_test[index]
      y_test = y_test[index]



      fig, ax = plt.subplots(4,1, figsize=(10,10))
      for i in range(4):
        ax[i].plot(x_test[:,i], color='grey')
        ax[i].plot(pred, label='Prediction')
        ax[i].plot(y_test, label='True signal')
      plt.legend()
      if save_path == '':
        save_path = config['model_path'] + f'plot/performance_{count}.png'
      else:
        save_path = save_path + f'performance_{count}.png'  
      plt.savefig(save_path)
      count += 1
  return None

def get_roc(y, y_pred, bins=100):
  TPR = []
  FPR = []
  binnings = np.logspace(np.log10(np.amin(y_pred)), np.log10(np.amax(y_pred)), num=bins)
  for i, limit in enumerate(binnings):
    
    sub_arr = np.zeros_like(y_pred)
    sub_arr[y_pred <= limit] = 0
    sub_arr[y_pred > limit] = 1
    sub_arr = np.max(sub_arr, axis=-1)
    true_pos = np.sum(np.logical_and(y == 1, sub_arr == 1))
    false_pos = np.sum(np.logical_and(y == 0, sub_arr == 1))
    true_neg = np.sum(np.logical_and( y == 0, sub_arr == 0))
    false_neg = np.sum(np.logical_and(y == 1, sub_arr == 0))
    TPR.append(true_pos/(true_pos + false_neg))
    FPR.append(false_pos/(false_pos + true_neg))

  return TPR, FPR

def get_noise_reduction(y, y_pred, bins, window_pred=False):
  smask = y == 1

  binnings = np.logspace(np.log10(np.amin(y_pred)), np.log10(np.amax(y_pred)), num=bins)
  backround  = float(len(y[~smask]))
  signal = float(len(y[smask]))
  noise_reduction = []
  TP = []
  
  for i, limit in enumerate(binnings):

    if window_pred:
      sub_arr = np.zeros_like(y_pred)
      sub_arr[y_pred <= limit] = 0
      sub_arr[y_pred > limit] = 1
      pred_per_window = np.max(sub_arr, axis=1)

      pred_pos = ((pred_per_window[smask] > limit) == True).astype(int)
      pred_neg = ((pred_per_window[~smask] <= limit) == True).astype(int)

      nr_true_pos = np.sum(pred_pos)
      nr_true_neg = np.sum(pred_neg)
    else:
      nr_true_pos = np.sum((y_pred[smask] > limit) == True)
      nr_true_neg = np.sum((y_pred[~smask] <= limit) == True)


    true_pos = nr_true_pos/ signal
    TP.append(true_pos)

    true_neg = nr_true_neg / backround 

    if (true_neg < 1):
      noise_reduction.append(1 / (1 - true_neg))
    else:
      noise_reduction.append(backround)

  return TP, noise_reduction      
