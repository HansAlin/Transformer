import matplotlib.pyplot as plt
import matplotlib.style as style
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

import numpy as np
import os
import pandas as pd
import pickle
import torch

import matplotlib.cm as cm
import itertools
import torch.nn as nn





import models.models as mm
from dataHandler.datahandler import get_trigger_data, get_model_path, get_model_config
#from evaluate.evaluate import get_model_path

def histogram(y_pred, y, config, bins=100, save_path='', text='', threshold=None, example_plot=False, alternative_title=None):
    """
          This function plots the histogram of the predictions of a given model.
          Args:
              y_pred (array like): array of y_pred
              y (array like): array of y
              config (dict): config dict of the model
              bins (int): number of bins in the histogram
              savefig_path (str): path to save the plot, optional
    """
    y_pred = np.array(y_pred).flatten()
    y = np.array(y).flatten()
    if max(y) == 0:
      return None
    if 'transformer' in config:
      config = config['transformer']
    if save_path == '':
      save_path = config['basic']['model_path'] + 'plot/' 
      isExist = os.path.exists(save_path)
      if not isExist:
        os.makedirs(save_path)
        print("The new directory is created!")
      save_path =    save_path + f"model_{config['basic']['model_num']}_histogram{text}.png"

    text = text.replace('_', '')
    plt.rcParams['text.usetex'] = True
    plt.rcParams['font.size'] = 15
    fig, ax = plt.subplots(figsize=(10, 5))
    signal_mask = y == 1
    y_pred_signal = y_pred[signal_mask]
    y_pred_noise = y_pred[~signal_mask]
    median_signal = np.median(y_pred_signal)
    median_noise = np.median(y_pred_noise)
    quantile_signal = np.quantile(y_pred_signal, 0.1)
    mean_signal = np.mean(y_pred_signal)
    mean_noise = np.mean(y_pred_noise)
    # signal_wights = np.ones_like(y_pred_signal) / len(y_pred_signal)
    # noise_weights = np.ones_like(y_pred_noise) / len(y_pred_noise)
    if not example_plot:
      ax.set_title(f"Model {config['basic']['model_num']} {text}")
    else:
      if alternative_title != None:
        ax.set_title(alternative_title)
      else:
        ax.set_title("Example")  
    ax.hist(y_pred_signal, bins=bins, label='True signal', alpha=0.5) # weights=signal_wights, 
    ax.hist(y_pred_noise, bins=bins, label='True noise', alpha=0.5) # weights=noise_weights, 
    ax.set_yscale('log')
    
    ax.set_xlabel(r'noise $\leftrightarrow$ signal')
    ax.set_ylabel('Counts')
    if threshold is not None:
        ax.axvline(x=threshold, color='r', linestyle='--', label=f'Threshold at {threshold:.2f}')
    if not example_plot:    
      ax.axvline(x=median_signal, color='g', linestyle='--', label=f'Median signal {median_signal:.2f}')
      ax.axvline(x=median_noise, color='b', linestyle='--', label=f'Median noise {median_noise:.2f}')
      ax.axvline(x=mean_signal, color='g', linestyle='-', label=f'Mean signal {mean_signal:.2f}')
      ax.axvline(x=mean_noise, color='b', linestyle='-', label=f'Mean noise {mean_noise:.2f}')  
    text_string = f'{(quantile_signal):.2f}' 
    text_string = '90 proc. signal at ' + text_string
    ax.axvline(x=quantile_signal, color='g', linestyle='-.', label=text_string)  
    ax.legend()

    plt.savefig(save_path)
    plt.clf()
    plt.close()



def plot_performance_curve(y_preds, ys, configs, bins=1000, save_path='', text = '', labels=None, x_lim=[0.8,1], curve='roc', log_bins=False, reject_noise=1e4):
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
                text (str): text to add to the title, optional
                labels (list of dicts): list of dicts with labels and hyper_parameters
                x_lim (list): list of x limits, optional
                curve (str): options: 'roc', 'nr'
                log_bins (bool): if True, the bins are log spaced
            Returns: area, nse
                area (float): area under the curve for both noise reduction factor and roc
                nse (float): noise reduction factor at 100k noise only for noise reduction factor curve
                              for roc curve, nse is None
                threshold (float): threshold at rejection  of reject_noise              
    """
 
    fig, ax = plt.subplots()
    length = len(y_preds)
    new_configs = []
    for config in configs:
      if 'transformer' not in config:
        config = {'transformer': config}
      new_configs.append(config)  
    
    configs = new_configs

    if save_path == '' and length == 1:
      save_path = configs[0]['transformer']['basic']['model_path'] + 'plot/' 
      isExist = os.path.exists(save_path)
      if not isExist:
        os.makedirs(save_path)
        print("The new directory is created!")
   
      save_path = save_path + f'model_{config["transformer"]["basic"]["model_num"]}_{curve}_{text.replace(" ", "_")}.png'  
    
    if labels == None:
      if configs[0] != None:
        ax.set_title(f"Model {configs[0]['transformer']['basic']['model_num']} {text}")
      else:
        ax.set_title(f"Model {text}")  
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

        nr_y_signal = np.count_nonzero(y==1)
        nr_y_noise = np.count_nonzero(y == 0)

        if curve == 'roc':
          x, y, nse, threshold = get_roc(y, y_pred,bins=bins, log_bins=log_bins, number_of_noise=reject_noise)
         
    
        elif curve == 'nr':  
          x, y, nse, threshold = get_noise_reduction(y, y_pred, bins, log_bins=log_bins, number_of_noise=reject_noise)
          

        if labels == None:
          ax.plot(x, y)   
        else:
            ax.plot(x, y, label=f" {config['transformer']['basic']['model_num']},     " + str(labels['hyper_parameters'][i]))
        
            
    if labels != None:
       ax.legend(title=labels['name'])        
    
    if curve == 'roc':
      ax.set_xlabel('False positive rate')
      ax.set_ylabel('True positive rate')
      ax.set_ylim([0.8,1])
      ax.set_xscale('log')
      ax.axvline(x=1/reject_noise, color='r', linestyle='--', label=f'Reject noise {reject_noise:.0f} Hz')
    elif curve == 'nr':
      ax.set_xlabel('True positive rate') 
      ax.set_ylabel(f'Noise reduction factor (nr. noise {nr_y_noise})')
      ax.set_yscale('log')
      ax.set_xlim(x_lim)
      ax.axhline(y=reject_noise, color='r', linestyle='--', label=f'Reject noise {reject_noise:.0f}')
    ax.grid()
    style.use('seaborn-colorblind')
    plt.legend()
    plt.savefig(save_path)
    plt.close()

    return get_area_under_curve(x,y), nse, threshold

def plot_results(model_number, config, path=''):
  if path == '':
    path = config['transformer']['basic']['model_path']
    plot_path = path + f'plot/' 
    df = pd.read_pickle(path + 'dataframe.pkl')
  else:
    plot_path = path
    df = pd.read_pickle(config['transformer']['basic']['model_path'] + 'dataframe.pkl')

  

  # Loss plot 
  fig, ax = plt.subplots()
  
  isExist = os.path.exists(plot_path)
  if not isExist:
      os.makedirs(plot_path)
      print("The new directory is created!")
  loss_path = plot_path + f'model_{model_number}_loss_plot.png'  
  line1, = ax.plot(df.Epochs, df.Train_loss, label='Training')
  line2, = ax.plot(df.Epochs, df.Val_loss, label='Validation')
  ax2 = ax.twinx()
  line3, = ax2.plot(df.Epochs, df.lr, label='Learning rate', color='gray')
  ax.set_title("Loss and learning rate")

  # Create a legend for all lines in both subplots
  lines = [line1, line2, line3]
  labels = [l.get_label() for l in lines]
  ax.legend(lines, labels)

  plt.savefig(loss_path)
  plt.cla()
  plt.clf()
  plt.close()

  # Accuracy plot
  fig,ax = plt.subplots()
  acc_path = plot_path + f'model_{model_number}_{config["training"]["metric"]}_plot.png'
  ax.plot(df.Epochs, df.metric, label=config['training']['metric'])
  ax.set_title("Metric")
  #plt.ylim([0.9,1])
  plt.grid()
  plt.legend()
  plt.savefig(acc_path)
  plt.cla()
  plt.clf()
  plt.close()

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
      save_path = config['basic']['model_path'] + 'plot/' 
      isExist = os.path.exists(save_path)
      if not isExist:
        os.makedirs(save_path)
        print("The new directory is created!")

    
  fig, ax = plt.subplots()
  if block == 'self_attention_block':
    weight = model.layers[config['h'] - 1].self_attention_block.W_0.weight.data.numpy()
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
     weight = model.layers[config['h'] - 1].feed_forward_block.linear_2.weight.data.numpy()
     x = ax.imshow(weight, cmap='coolwarm', interpolation='nearest')
     cbar = ax.figure.colorbar(x, ax=ax)
  if not quiet:
    print(weight)
  # writer.add_image("weight_image",weight )

  
  plt.title(f'Layer: {block}  - Weights')
  
  if save_path == '':
    save_path = config['basic']['model_path'] + f'plot/model_{config["basic"]["model_num"]}_{block}_weights.png'
  plt.savefig(save_path)
  plt.close()


def plot_features(model, save_path='/home/halin/Master/Transformer/figures/', quiet=True, plot_feature='attention_score', identyfier=''):

  matrix_to_plot = None  
  title = ''

  for name, module in model.named_modules():

    if isinstance(module, mm.MultiHeadAttentionBlock) and plot_feature == 'attention_score':
      attention_scores = module.attention_scores
      matrix_to_plot = attention_scores[0][0].cpu().detach().numpy() # just first item in batch
      title = 'Attention score'
      save_path = save_path + f'attention_score{identyfier}.png'
      break

  fig, ax = plt.subplots()
  if matrix_to_plot is not None:
    x = ax.imshow(matrix_to_plot, cmap='coolwarm', interpolation='nearest')
    ax.invert_yaxis()  # This line inverts the y-axis
    cbar = ax.figure.colorbar(x, ax=ax)
    plt.title(f'{title}')  
    plt.savefig(save_path)
    plt.close()


def plot_collections(models, labels, bins=100, save_path='', models_path='Test/ModelsResults/', x_lim=[0.8,1], window_pred=False, curve='roc', log_bins=False, reject_noise=1e4): 
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
    if labels == None:
      save_path = ''
    else:  
      model_name = models_path + f'collections/{labels["name"].replace(" ", "_")}_models'
      for model_num in models:
        model_name += f'_{model_num}'
      save_path =  model_name + '.png'  

  
  area, nse, threshold = plot_performance_curve(ys=y, 
                        y_preds=y_pred, 
                        configs=configs, 
                        save_path=save_path, 
                        labels=labels,
                        x_lim=x_lim,
                        bins=bins,
                        curve=curve,
                        log_bins=log_bins,
                        reject_noise=reject_noise)


def plot_examples(x, y, config=None, save_path='', data_type='trigger', antenna_type='LPDA'):

  plt.rcParams.update({'font.size': 15})  # Change the font size here
  
  if data_type == 'chunked':
    x = x.permute(0,2,1)
    y = y.max(dim=-1)
    if antenna_type == 'phased':
      sample_rate = 1 # GHz
    else:
      sample_rate = 0.5 # GHz
  else:
    sample_rate = 0.5 # GHz    

  signals = np.where(y == 1)[0]
  noise = np.where(y == 0)[0]
  n_ant = x.shape[2]
  seq_len = x.shape[1]

  x_signals = x[signals][:,:,:4] # For guarrante that we only use 4 antennas
  x_noise = x[noise][:,:,:4]    # For guarrante that we only use 4 antennas

  input_length = seq_len

  time_window = input_length / (sample_rate)
  time_window = f' ({time_window:.0f} ns)'

 # Find highest signal value
  try:
    max_index_flat = np.argmax(np.abs(x_signals))
  except ValueError:
    print('No signal found')
    return None
  batch_size_index, seq_len_index, n_ant_index = np.unravel_index(max_index_flat, x_signals.shape)
  max_signal_event = x_signals[batch_size_index, :, n_ant_index]

  # Find highest noise value
  max_index_flat = np.argmax(x_noise)
  batch_size_index, seq_len_index, n_ant_index = np.unravel_index(max_index_flat, x_noise.shape)
  max_noise_event = x_noise[batch_size_index, :, n_ant_index]

  x = [max_signal_event, max_noise_event]
  title = ['Signal', 'Noise']

  if config != None:
    save_path = config['basic']['model_path'] + 'plot/'
    isExist = os.path.exists(save_path)
    if not isExist:
        os.makedirs(save_path)
        print("The new directory is created!")

    save_path = save_path + 'examples.png'
  else:
    if save_path == '':
      save_path = os.getcwd() + '/figures/examples/examples.png'


  

  fig, ax = plt.subplots(1,2, figsize=(20,7), sharex=True, sharey=True)
  for j in range(2):

    ax[j].plot(x[j])
    # tixs1 = np.arange(0, seq_len + 1, seq_len//8)
    # tix_labels = tixs1/2
    # ax[j].set_xticks(tixs1)  # Set x-ticks at these locations
    # ax[j].set_xticklabels(tix_labels)  # Label x-ticks with these values
    
    ax[j].set_xlabel(f'Samples {time_window}')  # Set x-label
    
    ax[j].set_title(f'{title[j]}')  # Set title
    ax[j].axhline(0, color='gray', linestyle='-')  # Highlight y=0 line  
    ax[j].grid(True)   
  fig.text(0.005, 0.5, 'V / rms', va='center', rotation='vertical')  # Add common y-label
 
  plt.tight_layout()  # Make tight layout    
  plt.savefig(save_path)
  plt.clf()
  plt.close()  

def plot_performance(config, device, x_batch=None, y_batch=None,lim_value=0.2, model_path='/mnt/md0/halin/Models/', save_path='', text=''):
  """ This function plots the performance of a given model. If no data is given, the test data is used.
      Args:
        config (dict): config dict of the model
        device (torch.device): device to use
        x_batch (torch.tensor): batch of x data, optional
        y_batch (torch.tensor): batch of y data, optional
        lim_value (float): value to use as treshold
        model_path (str): path to model, optional
        save_path (str): path to save the plot, optional
  
  """
  
  if x_batch == None and y_batch == None:
    train_data, val_data, test_data = get_trigger_data(config, seq_len=config['input_length'], 
                                                            batch_size=config['training']['batch_size'], 
                                                            )
    del train_data
    del val_data
    x_batch, y_batch = test_data.__getitem__(0)
    x_test = x_batch
    y_test = y_batch
    
  else:
    x_test = x_batch
    y_test = y_batch

  device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  
  model = mm.load_model(config, text=text)
  model.to(device)
  x_test, y_test = x_test.to(device), y_test.to(device)
  model.eval()
  pred = model(x_test)
  index = 0

  pred = pred.cpu().detach().numpy()
  x_test = x_test.cpu().detach().numpy()
  y_test = y_test.cpu().detach().numpy()
  index = 0

  count = 1
  text = ''
  plot_true_pos = False
  plot_false_pos = False
  plot_true_neg = False
  plot_false_neg = False
  plot = False
  for i in range(pred.shape[0]):
    pred_value = np.max(pred[i])
    y_value = np.max(y_test[i])

    if y_value >= lim_value:
      if pred_value >= lim_value and not plot_true_pos:
        text = 'true positive'
        plot_true_pos = True
        plot = True
      elif pred_value < lim_value and not plot_false_neg:
        text = 'false negative' 
        plot_false_neg = True 
        plot = True
      else:
        plot = False 
    else:
      if pred_value >= lim_value and not plot_false_pos:
        text = 'false positive'
        plot_false_pos = True
        plot = True
      elif pred_value < lim_value and not plot_true_neg:
        text = 'true negative'
        plot_true_neg = True  
        plot = True
      else:
        plot = False  

    index = i

    if plot:
      max_value = np.max(pred[index])  
      print(f"Max value: {max_value} {text}")
      pred_y = pred[index]
      x = x_test[index]
      y = y_test[index]

      seq_len = x.shape[0]

      fig, ax = plt.subplots(4,1, figsize=(10,10), sharex=True, sharey=True)
      for i in range(4):
        ax[i].plot(x[:,i])
        ax[i].plot(pred_y, label='Prediction')
        ax[i].plot(y, label='True signal')
        tixs1 = np.arange(0, seq_len + 1, seq_len//8)
        tix_labels = tixs1/2
        ax[i].set_xticks(tixs1)  # Set x-ticks at these locations
        ax[i].set_xticklabels(tix_labels)  # Label x-ticks with these values
        ax[i].grid(True)
      if config['transformer']["architecture"]["data_type"] == "chunked":
        plt.legend()
      
      fig.suptitle(f"Model {config['transformer']['basic']['model_num']} - {text} treshold: {lim_value}")
      if save_path == '':
        path = config['transformer']['basic']['model_path'] + f'plot/performance_{text.replace(" ", "_")}.png'
      else:
        path = save_path + f'performance_{text.replace(" ", "_")}.png'  
      fig.text(0.005, 0.5, 'V / rms', va='center', rotation='vertical')   
      plt.tight_layout()  
      plt.savefig(path)
      count += 1
      plt.clf()
      plt.close()
  return None

def get_roc(y_true, y_score, bins=100, log_bins=False, number_of_noise=1e4):
    """
        This function calculates the ROC curve for a given model.
        Args:
            y_true (array like): array of y_true
            y_score (array like): array of y_score
            bins (int): number of bins in the histogram
            log_bins (bool): if True, the bins are log spaced
        Returns: fpr, tpr
            fpr (array like): false positive rate
            tpr (array like): true positive rate    
        """
    if log_bins:
      binnings = np.logspace(np.log10(np.amin(y_score)), np.log10(np.amax(y_score)), num=bins)
    else:
      binnings = np.linspace(np.amin(y_score), np.amax(y_score), num=bins)   

    tpr = []
    fpr = []

    for threshold in binnings:
        # Predict class labels based on the threshold
        y_pred = np.where(y_score >= threshold, 1, 0)
        
        # Calculate True Positives (TP), False Positives (FP), True Negatives (TN), and False Negatives (FN)
        TP = np.sum((y_pred == 1) & (y_true == 1))
        FP = np.sum((y_pred == 1) & (y_true == 0))
        TN = np.sum((y_pred == 0) & (y_true == 0))
        FN = np.sum((y_pred == 0) & (y_true == 1))
        


        # Calculate TPR and FPR
        if (TP + FN) == 0:
          TPR = 0
        else:  
          TPR = TP / (TP + FN)

        if (FP + TN) == 0:
          FPR = 0
        else:    
          FPR = FP / (FP + TN)
        
        tpr.append(TPR)
        fpr.append(FPR)
    index = np.argmax(ensure_numpy_array(fpr) < 1/number_of_noise)
    nse = tpr[index]
    threshold = binnings[index]
    return fpr, tpr, nse, threshold

def get_noise_reduction(y, y_pred, bins=1000,  log_bins=False, number_of_noise=1e4):
    """ 
      This function calculates the noise reduction factor for a given model.
      Args:
          y (array like): array of y true values
          y_pred (array like): array of y_pred models predictions
          bins (int): number of bins in the histogram
          log_bins (bool): if True, the bins are log spaced
      Returns: TP, noise_reduction
          TP (array like): true positive rate
          noise_reduction (array like): noise reduction factor
    """
    smask = y == 1
    n = len(y_pred)

    if log_bins:
        binnings = np.logspace(np.log10(np.amin(y_pred)), np.log10(np.amax(y_pred)), num=bins)
    else:
        binnings = np.linspace(np.amin(y_pred), np.amax(y_pred), num=bins)

    background = float(len(y[~smask]))
    signal = float(len(y[smask]))

    TP = np.zeros_like(binnings)
    noise_reduction = np.zeros_like(binnings)

    for i, limit in enumerate(binnings):
       
        pred_pos = (y_pred[smask] > limit).astype(int)
        pred_neg = (y_pred[~smask] <= limit).astype(int)

        nr_true_pos = np.sum(pred_pos)
        nr_true_neg = np.sum(pred_neg)

        TP[i] = nr_true_pos / signal if signal != 0 else 0
        true_neg = nr_true_neg / background if background != 0 else 0

        noise_reduction[i] = 1 / (1 - true_neg) if true_neg < 1 else background

    nse, index = get_NSE_AT_NRF(TP, noise_reduction, number_of_noise=number_of_noise)

    return TP, noise_reduction, nse, binnings[index]
 
def get_NSE_AT_NRF(TP, noise_reduction, number_of_noise=1e4):
  """ 
    This function calculates the noise reduction factor at a given number of noise
    Args:
        TP (array like): array of true positive values
        noise_reduction (array like): array of noise reduction values
        number_of_noise (int): number of noise to calculate the noise reduction factor at
    Returns: nse
        nse (float): noise reduction factor at number_of_noise
  """
  TP = ensure_numpy_array(TP)
  noise_reduction = ensure_numpy_array(noise_reduction)
  index = np.argmax(noise_reduction >= number_of_noise)
  nse = TP[index]
  return nse, index


def ensure_numpy_array(variable):
    """ From ChatGPT
    This function ensures that the input is a numpy array.
    Args:
        variable (list or numpy array): input variable  
    Returns: variable
    """
    if isinstance(variable, list):
        return np.array(variable)
    elif isinstance(variable, np.ndarray):
        return variable
    else:
        raise ValueError("Input should be a list or a numpy array.")
    
def get_area_under_curve(x,y):
  """
    This function calculates the area under a curve. 
    And it can be used for both ROC and noise reduction factor curves.
    Args:
        x (array like): array of x values
        y (array like): array of y values
    Returns: area
        area (float): area under the curve    
  """
  n = len(x)

  y_max_value = np.amax(y)
  # If max value is larger than 1, we assuming the data origin from a
  # noise reduction curve, otherwise we assume it origin from a ROC curve
  if y_max_value > 1:
    x = np.append(x, 0)
    y = np.append(y, y_max_value)
    if x[0] > x[-1]:
      x = x[::-1]
      y = y[::-1]

  area = 0
  for i in range(1,n):
    delta_x = x[i] - x[i-1]
    y_mean = (y[i] + y[i-1]) / 2
    area += delta_x * y_mean
  return np.abs(area)


def plot_common_values(df, keys, save_path=''):
    print(df.dtypes)
    common_keys = ['N', 'activation', 'd_ff', 'd_model', 'embed_type', 'final_type', 'h', 'location', 'max_relative_position', 'n_ant', 'normalization', 'pos_enc_type', 'seq_len', 'batch_size', 'decrease_factor', 'dropout', 'learning_rate', 'loss_function', 'step_size']
    common_keys = [common_key for common_key in common_keys if not common_key in keys]
    common = find_constant_columns(df, column_names=['batch_size', 'seq_len', 'learning_rate' ])
    print('1')
    print(common.dtypes)
    # Transpose and reset index
    common = common.transpose().reset_index()
    print('2')
    print(common.dtypes)
    # Rename columns
    common.columns = ['Parameter', 'Value']
    print('3')
    print(common.dtypes)
    fig, ax = plt.subplots() 
    ax.axis('off')
    ax.axis('tight')
    common = pd.DataFrame(common)
    common = change_format_units(common)
    common = change_headers(common)
    for col in common.columns:
      try:
          common[col] = common[col].astype(int)
      except ValueError:
          pass
    print('4')
    print(common.dtypes)
    table = ax.table(cellText=common.values, colLabels=common.columns, loc='center', cellLoc = 'center')
    table.auto_set_font_size(False)
    table.set_fontsize(14)
    table.scale(1.5, 1.5)
    plt.savefig(save_path.replace('.png', '_common_values.png'))
    plt.close()
    print(common)

def plot_table(df, keys, save_path='', print_common_values=False):
  """ This function plots a table of the dataframe df
      Args:
        df (pd.DataFrame): dataframe to plot
        keys (list): list of keys that are being ploted
  """ 

  
  df_copy = df.copy()

  if print_common_values:
    plot_common_values(df_copy, keys, save_path=save_path.replace('.png', '_common_values.png'))

  

  for key in keys:
    if key not in df.columns:
        df[key] = np.nan
  num_of_rows = len(df)
  num_of_cols = len(df.columns)
  fig_hight = 0.5 * num_of_rows + 0.4
  fig_width = 0.23 * num_of_cols   # Add this line

  df = df[keys]
  df = change_format_units(df)
  df = change_headers(df)
  # df.columns = df.columns.str.pad(10, side='both')

  # Add white spaces around the keys
  df.columns = df.columns.str.pad(10, side='both')
  fig, ax = plt.subplots(figsize=(fig_width, fig_hight)) # 
  ax.axis('off')
  ax.axis('tight')
  table = ax.table(cellText=df.values, colLabels=df.columns, loc='center', cellLoc = 'center')

  # Add colors and so
  table.auto_set_font_size(False)
  table.set_fontsize(14)
  table.scale(1.5, 1.5)
  table.auto_set_column_width(col=list(range(len(df.columns))))
  for (row, col), cell in table.get_celld().items():
      if (row == 0):
        cell.set_facecolor('darkgrey')
      if (row % 2 == 0) and (row != 0):
        cell.set_facecolor('lightgrey')
  plt.title('Hyperparameters', fontsize=20, pad=0.1)  
  ax.title.set_position([.5, 1.01])   
  plt.subplots_adjust(left=0.01, bottom=0, right=0.99, top=0.95) 

  fig.tight_layout()
  plt.savefig(save_path)
  plt.close()
  df.to_csv(save_path.replace('.png', '.csv'), index=False)


def plot_common(df, row_height=0.4, col_width=2.9, exclude_columns=['pos_param']):
  df = df.drop([col for col in exclude_columns if col in df.columns], axis=1)
  df = df.dropna(axis=1, how='all')
  df = change_format_units(df)
  df = change_headers(df)
  dfT = df.transpose().reset_index()
  dfT.columns = ["Common", "Hyperparameters"]

  rows = dfT.shape[0]
  cols = dfT.shape[1]
  height = row_height * rows + 0.6
  width = col_width * cols
  fig, ax = plt.subplots(figsize=(width,height)) # set size frame 

  ax.axis('off')
  
  
  table = plt.table(cellText=dfT.values, 
                    colLabels=dfT.columns, 
                    cellLoc = 'center', 
                    loc='center')
  table.auto_set_font_size(False)
  table.set_fontsize(12)
  table.scale(1, 2)  # Play with this to adjust the height of the cells.
  plt.subplots_adjust(left=0.2, bottom=0.1, right=0.8, top=0.8)
  plt.show()
  plt.close()

def plot_comparing_models(df, row_height=0.35, col_width=1.8, exclude_columns=['pos_param','input_param' ], sort=False):
  if sort:
    df = df.sort_values(by='NSE_AT_10KNRF', ascending=False)
  df = df.drop([col for col in exclude_columns if col in df.columns], axis=1)
  df = df.dropna(axis=1, how='all')
  df = df.loc[:, (df != '').all()]
  df = change_format_units(df)
  rows = df.shape[0]
  cols = df.shape[1]
  height = row_height * rows + 0.5
  width = col_width * cols
  fig, ax = plt.subplots(figsize=(width,height)) # set size frame

  # Hide axes
  ax.axis('off')
  # plt.title('Hyperparamters')
  # Create table and set its properties
  df = change_headers(df)
  table = plt.table(cellText=df.values, 
                    colLabels=df.columns, 
                    cellLoc = 'center', 
                    loc='center')

  table.auto_set_font_size(False)
  table.set_fontsize(12)
  table.scale(1, 2)  # Play with this to adjust the height of the cells.

  # Adjust layout to make room for the table:
    # Adjust layout to make room for the table:
  plt.subplots_adjust(left=0.2, bottom=0.1, right=0.8, top=0.8)
  plt.show()  
  plt.close()




def change_format(value, digits=2, format="e"):
  return '{:.{digits}{format}}'.format(value, digits=digits, format=format)  

def find_constant_columns(df, column_names):
    """
    This function checks whether all the columns in the given list exist in the DataFrame
    and all their values are the same. If they are, it creates a new DataFrame with these columns and their values.
    """
    constant_columns = [col for col in column_names if col in df.columns and df[col].nunique() == 1]
    if set(constant_columns) == set(column_names):
        return df[constant_columns].iloc[[0]]  # Select a single-row DataFrame
    else:
        print(f'Columns {constant_columns} are not common')
        return None

def change_energy_units(value):
  changed_value = value/3600/1000
  return f'{changed_value:.2e} kWh'

def change_prefix(value, factor=1, prefix='k'):
  try:
    if prefix == 'k':
      return f'{(value / 1e3 * factor):.1f} k'
    elif prefix == 'M':
      return f'{(value / 1e6 * factor):.1f} M'
    elif prefix == 'G':
      return f'{(value / 1e9 * factor):.1f} G'
  except:
    return value


def change_to_kilo(value):
  try:
    return f'{(value / 1e3):.1f} k'
  except:
    return value


def change_headers(df):
  """ This function changes the headers of the dataframe df
      to make it more readable:

      Args: 
        df (pd.DataFrame): dataframe to change headers of

  """
  df = df.copy()  # Make a copy of the DataFrame
  if 'model_num' in df.columns:
    df.rename(columns={'model_num': 'Model number'}, inplace=True)
  if 'pos_enc_type' in df.columns:
    df.rename(columns={'pos_enc_type': 'Pos. enc. type'}, inplace=True)
  if 'd_model' in df.columns:
    df.rename(columns={'d_model': 'Model size'}, inplace=True)  
  if 'd_ff' in df.columns:
    df.rename(columns={'d_ff': 'Feed forward'}, inplace=True)
  if 'N' in df.columns:
    df.rename(columns={'N': 'Layers (N)'}, inplace=True)
  if 'h' in df.columns:
    df.rename(columns={'h': 'Heads (h)'}, inplace=True)
  if 'num_param' in df.columns:
    df.rename(columns={'num_param': 'Tot. param.'}, inplace=True)
  if 'NSE_AT_10KNRF' in df.columns:
    df.rename(columns={'NSE_AT_10KNRF': 'NSE at 10k NRF'}, inplace=True)
  if 'activation' in df.columns:
    df.rename(columns={'activation': 'Act. function'}, inplace=True)
  if 'embed_type' in df.columns:
    df.rename(columns={'embed_type': 'Input embed. type'}, inplace=True)
  if 'final_type' in df.columns:
    df['final_type'] = df['final_type'].apply(abbreviate)
    df.rename(columns={'final_type': 'Final layer type'}, inplace=True)
  if 'location' in df.columns:
    df.rename(columns={'location': 'Residual location'}, inplace=True)
  if 'max_relative_position' in df.columns:
    df.rename(columns={'max_relative_position': 'Max. rel. pos.'}, inplace=True)
  if 'n_ant' in df.columns:
    df.rename(columns={'n_ant': 'Antennas'}, inplace=True)
  if 'normalization' in df.columns:
    df.rename(columns={'normalization': 'Normalization'}, inplace=True)
  if 'seq_len' in df.columns:
    df.rename(columns={'seq_len': 'Seq. length'}, inplace=True)
  if 'batch_size' in df.columns:
    df.rename(columns={'batch_size': 'Batch size'}, inplace=True)
  if 'dropout' in df.columns:
    df.rename(columns={'dropout': 'Dropout'}, inplace=True)
  if 'learning_rate' in df.columns:
    df.rename(columns={'learning_rate': 'Learning rate'}, inplace=True)
  if 'loss_function' in df.columns:
    df.rename(columns={'loss_function': 'Loss function'}, inplace=True)
  if 'step_size' in df.columns:
    df.rename(columns={'step_size': 'Step size'}, inplace=True)
  if 'decreas_factor' in df.columns:
    df.rename(columns={'decreas_factor': 'Decr. factor'}, inplace=True)
  if 'encoder_param' in df.columns:
    df.rename(columns={'encoder_param': 'Encoder param.'}, inplace=True)
  if 'input_param' in df.columns:
    df.rename(columns={'input_param': 'Input param.'}, inplace=True)
  if 'pos_param' in df.columns:
    df.rename(columns={'pos_param': 'Pos. param.'}, inplace=True)
  if 'final_param' in df.columns:
    df.rename(columns={'final_param': 'Final param.'}, inplace=True)
  if 'encoder_type' in df.columns:
    df.rename(columns={'encoder_type': 'Encoder type'}, inplace=True)

  return df


def change_format_units(df):
  """ This function changes the format of the dataframe df
      to make it more readable:

      Args: 
        df (pd.DataFrame): dataframe to change format of

  """
  df = df.copy()
  if 'MACs' in df.columns:
    df.loc[:,'MACs'] = df['MACs'].apply(change_prefix, factor=2, prefix='M')
    df.rename(columns={'MACs': 'FLOPs'}, inplace=True)
  if 'num_param' in df.columns:
    df.loc[:,'num_param'] = df['num_param'].apply(change_prefix, factor=1, prefix='k')
  if 'pos_param' in df.columns:
    df.loc[:,'pos_param'] = df['pos_param'].apply(change_prefix, factor=1, prefix='k')
  if 'input_param' in df.columns:
    df.loc[:,'input_param'] = df['input_param'].apply(change_prefix, factor=1, prefix='k')
  if 'encoder_param' in df.columns:
    df.loc[:,'encoder_param'] = df['encoder_param'].apply(change_prefix, factor=1, prefix='k')    
  if 'energy' in df.columns:  
    df.loc[:,'energy'] = df['energy'].apply(lambda x: change_energy_units(x))
  if 'roc_area' in df.columns:
    df.loc[:,'roc_area'] = df['roc_area'].apply(lambda x: change_format(x, digits=4))
  if 'nr_area' in df.columns:
    df.loc[:,'nr_area'] = df['nr_area'].apply(lambda x: change_format(x, digits=2))  
  if 'NSE_AT_10KNRF' in df.columns:
    df.loc[:,'NSE_AT_10KNRF'] = df['NSE_AT_10KNRF'].apply(lambda x: change_format(x, digits=3, format='f'))
  if 'training_time' in df.columns:
    df.loc[:,'training_time'] = df['training_time']/3600
    df.loc[:,'training_time'] = df['training_time'].apply(lambda x: change_format(x, digits=0, format='f'))

  return df  
 




def plot_attention_scores(model, x, y, save_path='/home/halin/Master/Transformer/figures/attention/', extra_identifier=''):
    if y == 0:
      y = 'noise'
    elif y == 1:
      y = 'signal'

    if not os.path.exists(save_path):
      os.makedirs(save_path)

    x = x.cpu().detach().numpy()
    item = 1
    j = 0
    attention_scores = []
    
    for name, module in model.named_modules():
      if isinstance(module, mm.MultiHeadAttentionBlock) or isinstance(module, mm.VanillaEncoderBlock):
        
        if isinstance(module, mm.VanillaEncoderBlock):
          first_att = module.get_attention_scores()
          
        else:
          first_att = module.attention_scores
          
        first_att = first_att.cpu().detach().numpy()
        attention_scores.append(first_att)


    number_attentions = len(attention_scores)  
    head_range = number_attentions

    fig = plt.figure(figsize=(20, 20))  # Increase the size of the figure

    # outer_grid = gridspec.GridSpec(2, 2, wspace=0.1, hspace=0.1)  # Add some space between the figures
    fig.suptitle(f'Attention Scores {item}- {y}', fontsize=16)
    
    for i in range(head_range):
      inner_grid = gridspec.GridSpecFromSubplotSpec(2, 2, 
                                                        #subplot_spec=outer_grid[i], 
                                                        width_ratios=[1, 5], 
                                                        height_ratios=[5, 1],
                                                        wspace=0,
                                                        hspace=0)

      ax0 = plt.Subplot(fig, inner_grid[0])
      ax1 = plt.Subplot(fig, inner_grid[1])
      ax3 = plt.Subplot(fig, inner_grid[3])

      fig.add_subplot(ax0)
      fig.add_subplot(ax1)
      fig.add_subplot(ax3)

      x_reduced = x[0,:,0]
      ax1.set_title(f"Head {j+1}")
      # Plot the attention scores
      attention_score = attention_scores[i][0]
      im = ax1.imshow(attention_score, cmap='hot', interpolation='none')
      plt.savefig(save_path + f'{extra_identifier}attention_scores_{item}_{y}.png')
      plt.close()
      # # Remove the x and y ticks
      # ax1.set_xticks([])
      # ax1.set_yticks([])
      # # Plot the reduced input as a curve along the x axis
      # ax3.plot(x_reduced)
      # ax3.grid()
      # ax3.set_xlim([0, len(x_reduced)]) 
      # # Plot the reduced input as a curve along the y axis
      # ax0.plot(x_reduced, range(len(x_reduced)))
      # ax0.grid()
      # ax0.set_ylim([len(x_reduced), 0]) 
      # ax0.invert_xaxis()  # Invert the y axis to align it with the imshow plot

      # # Create a new Axes object for the colorbar
      # cax = fig.add_axes([0.96, 0.2, 0.01, 0.6])

      # # Create a colorbar in the new Axes object
      # cbar = fig.colorbar(im, cax=cax)
      # fig.subplots_adjust(left=0.05, right=0.95, bottom=0.05, top=0.95, wspace=0, hspace=0)

      
      item += 1
  



def plot_hist(values, names, log=False, threshold=None, bins=100, xlim=0, ylim=10000,save_path='/home/halin/Master/Transformer/figures/noise_hist.png'):
    if len(values) != len(names):
        raise ValueError("values and names must have the same length")

    fig, ax = plt.subplots()

    min_value = min(min(v) for v in values if v is not None)
    if log:
        bins = np.logspace(np.log10(min_value), 1, num=bins)

    for value, name in zip(values, names):
        if value is not None:
            ax.hist(value, bins=bins, alpha=0.5, label=name)
            ax.axvline(x=threshold, color='red', linestyle='--', label='Threshold')
            plt.legend()

    if log:
        ax.set_xscale('log')
   
    plt.savefig(save_path)
    
    plt.close() 

def plot_tensor(tensor, name, save_folder='/home/halin/Master/Transformer/figures/'):
  """
  This function plots a table of the relative embeddings
  Args:
    embedding_table (numpy.array): the table to plot
  """    
  fig, ax = plt.subplots()
  x = ax.imshow(tensor, cmap='hot',)
  ax.figure.colorbar(x, ax=ax)

  ax.set_title(f"{name.replace('_', ' ') }")
  plt.savefig(save_folder + f'relative_embedding_table_{name}.png')

def plot_layers(state_dict, layer, extra_name='', save_folder='/home/halin/Master/Transformer/figures/'):
  item = 1
  for (k, v) in state_dict.items():
    if layer in k:
        print(k, v.shape)
        name = f"layer_{layer}_{item}_{extra_name}"
        plot_tensor(
            v.cpu().detach().numpy(),
            name=name,
            save_folder=save_folder
        )
        item += 1

def plot_threshold_efficiency(y_pred, y, save_path='/home/halin/Master/Transformer/figures/efficiency'):
  
  y = y
  y_pred = y_pred
  count = 0

  accuracys = []
  efficiencies = []
  precisions = []
  thresholds = []
  noise_rejection_rates = []
  F1s = []

  for threshold in np.linspace(min(y_pred), max(y_pred), 1000):

    TP = np.sum(np.logical_and(y == 1, y_pred > threshold))
    TN = np.sum(np.logical_and(y == 0, y_pred < threshold))
    FP = np.sum(np.logical_and(y == 0, y_pred > threshold))
    FN = np.sum(np.logical_and(y == 1, y_pred < threshold))

    if (TN + FP) != 0:
      noise_rejection_rate = FP / (TN + FP)
      
      accuracy = (TP + TN) / len(y)
      efficiency = TP / np.count_nonzero(y) if np.count_nonzero(y) != 0 else 0
      precision = TP / (TP + FP) if TP + FP != 0 else 0 
      recall = TP / (TP + FN) if TP + FN != 0 else 0
      F1 = 2 * (precision * recall) / (precision + recall) if precision + recall != 0 else 0

      # print(f"Noise rejection rate: {noise_rejection_rate:>5.2e} | Threshold: {threshold:>5.2f} | Accuracy: {accuracy:>5.2f} | Efficiency: {efficiency:>5.2f} | Precision: {precision:>5.2f} | Recall: {recall:>5.2f} | F1: {F1:>5.2f}") 
      
      accuracys.append(accuracy)
      efficiencies.append(efficiency)
      precisions.append(precision)
      thresholds.append(threshold)
      noise_rejection_rates.append(noise_rejection_rate)
      F1s.append(F1)
  
  
  fig, ax1 = plt.subplots(1, 1, figsize=(7, 7))

  color = 'tab:red'
  ax1.set_xlabel('Threshold')
  
  ax1.plot(thresholds, efficiencies, label='Efficiency', color='salmon')
  plt.legend()
  ax1.plot(thresholds, precisions, label='Precision', color='darkred')
  plt.legend()
  ax1.plot(thresholds, accuracys, label='Accuracy',color='red')
  plt.legend()
  ax1.plot(thresholds, F1s, label='F1 score', color='red')
  plt.legend()
  ax1.set_ylabel('Efficiency, Precision, Accuracy', color=color)
  ax1.tick_params(axis='y', labelcolor=color)

  ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis
  color = 'tab:blue'
  ax2.set_ylabel('Noise rejection rate', color=color)  # we already handled the x-label with ax1
  ax2.plot(thresholds, noise_rejection_rates, label='Noise rejection rate', color='blue')
  ax2.tick_params(axis='y', labelcolor=color)
  

  fig.tight_layout()  # otherwise the right y-label is slightly clipped
  plt.legend()

  plt.savefig(save_path) 
  plt.close() 

def qualitative_colors(length, darkening_factor=0.6):
    colors = [cm.Set3(i) for i in np.linspace(0, 1, length)]
    darker_colors = [(r*darkening_factor, g*darkening_factor, b*darkening_factor, a) for r, g, b, a in colors]
    return darker_colors

def plot_veff(datafiles, plot_path='/home/halin/Master/Transformer/figures/veff.png'):

  colors = itertools.cycle(qualitative_colors(7))
  markers = itertools.cycle(("s", "P", "o", "^", ">", "X"))
  linestyles = itertools.cycle(("-", "--", ":", "dashdot", (0, (3, 5, 1, 5))))

  plt.rcParams.update({'font.size': 15})  # Change the font size here
  


  nrows = 1
  ncols = 1
  fig, ax = plt.subplots(
      ncols=ncols, nrows=nrows, figsize=(ncols * 15 * 0.7, nrows * 10 * 0.7), gridspec_kw={"wspace": 0.2, "hspace": 0.2}
  )

  standard_trig_not_plotted = True

  for dta_file in datafiles:
    data = np.load(dta_file)
    lgEs = data['lgEs']
    trig_1Hz = data['trig_1Hz']
    trig_10kHz = data['trig_10kHz']
    if standard_trig_not_plotted:
      ax.plot(lgEs, trig_1Hz, label='1 Hz', color=next(colors), marker=next(markers), linestyle=next(linestyles))
      ax.plot(lgEs, trig_10kHz, label='10 kHz', color=next(colors), marker=next(markers), linestyle=next(linestyles))
      standard_trig_not_plotted = False
    i = 0
    for key, value in data.items():
      # if (key == 'lgEs') or (key == 'trig_1Hz') or (key == 'trig_10kHz'):
      #   continue
      if 'best' in key:
        ax.plot(lgEs, value, label=key, color=next(colors), marker=next(markers), linestyle=next(linestyles))
      i += 1
  style.use('seaborn-colorblind')
  ymin, ymax = ax.get_ylim()
  ax.set_ylim(0.9, ymax)
  ax.legend(prop={"size": 10})
  ax.set_xlabel(r"lg(E$_{\nu}$ / eV)")
  ax.set_ylabel(r"V$_{\rm eff}$ / (" + 'trig 1 Hz' + ")")
  ax.tick_params(axis="both", which="both", direction="in")
  ax.yaxis.set_ticks_position("both")
  ax.xaxis.set_ticks_position("both")
  ax.grid()
  fig.savefig(plot_path)

def plot_chunked_veff(datafiles, plot_path='/home/halin/Master/Transformer/figures/chunked_veff.png'):

  colors = itertools.cycle(qualitative_colors(7))
  markers = itertools.cycle(("s", "P", "o", "^", ">", "X"))
  linestyles = itertools.cycle(("-", "--", ":", "dashdot", (0, (3, 5, 1, 5))))

  nrows = 1
  ncols = 1
  fig, ax = plt.subplots(
      ncols=ncols, nrows=nrows, figsize=(ncols * 15 * 0.7, nrows * 10 * 0.7), gridspec_kw={"wspace": 0.2, "hspace": 0.2}
  )

  standard_trig_1Hz_not_plotted = True
  standard_trig_10kHz_not_plotted = True
  trig_1Hz = 0
  for dta_file in datafiles:
    data = np.load(dta_file, allow_pickle=True)
    sort = np.argsort(data['lges'])
    lges= data['lges'][sort]
    
    veff_dict = data['veff'].item()  # Get the dictionary from the numpy array
    for key, value in veff_dict.items():
        value = np.array(value)
        print(key)
        if standard_trig_1Hz_not_plotted and key == 'trig_1Hz':
            trig_1Hz = value[sort]
            print(trig_1Hz)
            ax.plot(lges, trig_1Hz/trig_1Hz, label='1 Hz', color=next(colors), marker=next(markers), linestyle=next(linestyles))
            standard_trig_1Hz_not_plotted = False
        elif standard_trig_10kHz_not_plotted and key == 'trig_10kHz':
            
            ax.plot(lges, value[sort]/trig_1Hz, label='10 kHz', color=next(colors), marker=next(markers), linestyle=next(linestyles))
            standard_trig_10kHz_not_plotted = False
        elif 'average' in key:
            label_string = key.split('_')[1] + ' ' + key.split('_')[3] + ' ' + key.split('_')[4] + ' ' + 'average'

       
            ax.plot(lges, value[sort]/trig_1Hz, label=label_string, color=next(colors), marker=next(markers), linestyle=next(linestyles))    

  style.use('seaborn-colorblind')
  ymin, ymax = ax.get_ylim()
  ax.set_ylim(0.9, ymax)
  ax.legend(prop={"size": 6})
  ax.set_xlabel(r"lg(E$_{\nu}$ / eV)")
  ax.set_ylabel(r"V$_{\rm eff}$ / (" + 'trig 1 Hz' + ")")
  ax.tick_params(axis="both", which="both", direction="in")
  ax.yaxis.set_ticks_position("both")
  ax.xaxis.set_ticks_position("both")
  fig.savefig(plot_path)