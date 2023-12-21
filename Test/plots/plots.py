import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import pickle

def histogram(y_pred, y, config, bins=100, save_path=''):

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

def noise_reduction_factor(y_preds, ys, configs, bins=100, save_path='', labels=None):
    """
            This function plots the noise reduction factor curve for a given model or models.
            Args:
                y_preds (list of np.arrays): list of y_pred arrays
                ys (list of np.arrays): list of y arrays
                configs (list of dicts): list of config dicts 
                bins (int): number of bins in the histogram
                savefig_path (str): path to save the plot, optional
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
      ax.set_title(f"Model {configs[0]['model_num']}")
    else:
        ax.set_title(f"Noise reduction factor for {length} models")
      
        

    for i in range(length): 
        y_pred = y_preds[i]
        y = ys[i]
        config = configs[i]
        signal_mask = y == 1
        y_pred_signal = y_pred[signal_mask]
        y_pred_noise = y_pred[~signal_mask]
        true_pos = np.zeros(bins)
        true_neg = np.zeros(bins)
        noise_reduction_factor = np.zeros(bins)
        binnings = np.linspace(0,1,bins)
        
        for j, limit in enumerate(binnings):
            true_pos[j] = np.count_nonzero(y_pred_signal > limit)/len(y_pred_signal)
            true_neg[j] =np.count_nonzero(y_pred_noise < limit)/len(y_pred_noise)

            if (true_neg[j] != 1):
                noise_reduction_factor[j] = 1 / (1 - true_neg[j])
            else:
                noise_reduction_factor[j] = len(y_pred_noise)  

        
        if labels == None:
          ax.plot(true_pos, noise_reduction_factor)   
        else:
            ax.plot(true_pos, noise_reduction_factor, label=labels['hyper_parameters'][i])
        
            
    if labels != None:
       ax.legend(title=labels['name'])        
    
    ax.set_xlabel('True positive rate')
    ax.set_ylabel(f'Noise reduction factor (nr. noise {len(y_pred_noise)})')
    ax.set_yscale('log')
    ax.set_xlim([0.8,1])

    if save_path == '':
        save_path = config['model_path'] + 'plot/' + f'model_{config["model_num"]}_noise_reduction.png'
    plt.savefig(save_path)


def plot_results(model_number, config, path=''):
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
  df.plot('Epochs', 'metric', label=config['metric'])
  plt.title("Accuracy")
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

def plot_collections(models, labels, save_path=''): 
  y_pred = []
  y = []
  configs = []

  for i, model_num in enumerate(models):
    with open(f"Test/ModelsResults/model_{model_num}/config.txt", 'rb') as f:
        config = pickle.load(f)
    with open(f"Test/ModelsResults/model_{model_num}/y_pred_data.pkl", 'rb') as f:
        y_data = pickle.load(f)
    
    y_pred.append(np.asarray(y_data['y_pred']))
    y.append(np.asarray(y_data['y']))
    configs.append(config)
  if save_path == '':  
    model_name = f'/Test/ModelsResults/collections/{labels["name"].replace(" ", "_")}_models'
    for model_num in models:
      model_name += f'_{model_num}'
    save_path = os.getcwd() + model_name + '.png'  

  
  noise_reduction_factor(ys=y, 
                        y_preds=y_pred, 
                        configs=configs, 
                        save_path=save_path, labels=labels)


def plot_examples(data):
  data = data[0]
  fig, ax = plt.subplots(4,1, figsize=(10,10))
  for i in range(4):
    ax[i].plot(data[i][200:300])
  plt.savefig('Test/ModelsResults/collections/examples.png')  