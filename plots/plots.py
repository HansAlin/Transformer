import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import pickle
import torch


from models.models import build_encoder_transformer
from dataHandler.datahandler import get_test_data, get_data_binary_class

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
    ax.hist(y_pred_signal, bins=bins, label='Pred signal', weights=signal_wights, alpha=0.5)
    ax.hist(y_pred_noise, bins=bins, label='Pred noise', weights=noise_weights, alpha=0.5)
    ax.set_yscale('log')
    ax.legend()
    ax.set_xlabel(r'noise $\leftrightarrow$ signal')
    plt.savefig(save_path + f"model_{config['model_num']}_histogram.png")
    plt.clf()
    plt.close()



def plot_performance_curve(y_preds, ys, configs, bins=1000, save_path='', text = '', labels=None, x_lim=[0.8,1], curve='roc', log_bins=False):
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

    """
 
    fig, ax = plt.subplots()
    length = len(y_preds)

    if save_path == '' and length == 0:
      save_path = config['model_path'] + 'plot/' 
      isExist = os.path.exists(save_path)
      if not isExist:
        os.makedirs(save_path)
        print("The new directory is created!")

    if labels == None:
      if configs[0] != None:
        ax.set_title(f"Model {configs[0]['model_num']} {text}")
      else:
        ax.set_title(f"Model Test")  
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
          x, y = get_roc(y, y_pred,bins=bins, log_bins=log_bins)
        elif curve == 'nr':  
          x, y = get_noise_reduction(y, y_pred, bins, log_bins=log_bins)

        if labels == None:
          ax.plot(x, y)   
        else:
            ax.plot(x, y, label=labels['hyper_parameters'][i])
        
            
    if labels != None:
       ax.legend(title=labels['name'])        
    
    if curve == 'roc':
      ax.set_xlabel('False positive rate')
      ax.set_ylabel('True positive rate')
      ax.set_ylim([0.8,1])
      ax.set_xscale('log')
    elif curve == 'nr':
      ax.set_xlabel('True positive rate') 
      ax.set_ylabel(f'Noise reduction factor (nr. noise {nr_y_noise})')
      ax.set_yscale('log')
      ax.set_xlim(x_lim)
    ax.grid()
    if save_path == '':
        save_path = config['model_path'] + 'plot/' + f'model_{config["model_num"]}_{curve}_{text.replace(" ", "_")}.png'
    plt.savefig(save_path)
    plt.close()

    return get_area_under_curve(x,y)

def plot_results(model_number, config, path=''):
  if path == '':
    path = config['model_path']

  df = pd.read_pickle(path + 'dataframe.pkl')

  # Loss plot 
  fig,ax = plt.subplots()
  plot_path = path + f'plot/' 
  isExist = os.path.exists(plot_path)
  if not isExist:
    os.makedirs(plot_path)
    print("The new directory is created!")
  loss_path = plot_path + f'model_{model_number}_loss_plot.png'  
  ax.plot(df.Epochs, df.Train_loss, label='Training')
  ax.plot(df.Epochs, df.Val_loss, label='Validation')
  ax2 = ax.twinx()
  ax2.plot(df.Epochs, df.lr, label='Learning rate', color='black')
  ax.set_title("Loss and learning rate")
  plt.legend()
  plt.savefig(loss_path)
  plt.cla()
  plt.clf()
  plt.close()

  # Accuracy plot
  fig,ax = plt.subplots()
  acc_path = plot_path + f'model_{model_number}_{config["metric"]}_plot.png'
  ax.plot(df.Epochs, df.metric, label=config['metric'])
  ax.set_title("Metric")
  plt.ylim([0,1])
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
  plt.close()


def plot_collections(models, labels, bins=100, save_path='', models_path='Test/ModelsResults/', x_lim=[0.8,1], window_pred=False, curve='roc', log_bins=False): 
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

  
  plot_performance_curve(ys=y, 
                        y_preds=y_pred, 
                        configs=configs, 
                        save_path=save_path, 
                        labels=labels,
                        x_lim=x_lim,
                        bins=bins,
                        curve=curve,
                        log_bins=log_bins)


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
  plt.close()  

def plot_performance(config, device, x_batch=None, y_batch=None,lim_value=0.2, model_path='/mnt/md0/halin/Models/', save_path=''):
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
  model_num = config['model_num']
  if x_batch == None and y_batch == None:
    train_data, val_data, test_data = get_data_binary_class(seq_len=config['seq_len'], 
                                                            batch_size=config['batch_size'], 
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



      fig, ax = plt.subplots(4,1, figsize=(10,10), sharex=True)
      for i in range(4):
        ax[i].plot(x[:,i], color='grey')
        ax[i].plot(pred_y, label='Prediction')
        ax[i].plot(y, label='True signal')
      if config["data_type"] != "classic":
        plt.legend()
      
      fig.suptitle(f"Model {config['model_num']} - {text} treshold: {lim_value}")
      if save_path == '':
        path = config['model_path'] + f'plot/performance_{text.replace(" ", "_")}.png'
      else:
        path = save_path + f'performance_{text.replace(" ", "_")}.png'  
      plt.savefig(path)
      count += 1
      plt.clf()
      plt.close()
  return None

def get_roc(y_true, y_score, bins=100, log_bins=False):
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
      binnings = np.linspace(np.amin(y_score), 0.99999, num=bins)   

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
    
    return fpr, tpr

def get_noise_reduction(y, y_pred, bins=1000,  log_bins=False):
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
        binnings = np.linspace(np.amin(y_pred), 0.99999, num=bins)

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

    return TP, noise_reduction
 

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

