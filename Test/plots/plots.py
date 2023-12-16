import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd

def histogram(y_pred, y,config, bins=100):
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
    plt.savefig(config['model_path'] + 'plot/' + f'model_{config["model_num"]}_hist.png')
    plt.clf()

def noise_reduction_factor(y_preds, ys, configs, bins=100, savefig_path='', labels=[]):
    """
            This function plots the noise reduction factor curve for a given model or models.
            Args:
                y_preds (list of np.arrays): list of y_pred arrays
                ys (list of np.arrays): 
    """
    fig, ax = plt.subplots()
    length = len(y_preds)
    if len(labels) > 0:
        ax.set_title(f"Noise reduction factor for {length} models")
    else:    
        ax.set_title(f"Model {configs[0]['model_num']}")

    for i in range(length): 
        y_pred = y_preds[i]
        y = ys[i]
        config = configs[i]
        signal_mask = y == 1
        y_pred_signal = y_pred[signal_mask]
        y_pred_noise = y_pred[~signal_mask]
        true_pos = np.zeros(bins)
        false_pos = np.zeros(bins)
        noise_reduction_factor = np.zeros(bins)
        binnings = np.linspace(0,1,bins)
        
        for i, limit in enumerate(binnings):
            true_pos[i] = np.count_nonzero(y_pred_signal > limit)/len(y_pred_signal)
            false_pos[i] =np.count_nonzero(y_pred_noise > limit)/len(y_pred_noise)

            if (false_pos[i] > 0):
                noise_reduction_factor[i] = 1 / ( false_pos[i])
            else:
                noise_reduction_factor[i] = len(y_pred_noise)  

        
        ax.plot(true_pos, noise_reduction_factor)
        if len(labels) > 0:
            ax.set_label(labels[i])

        ax.legend()
    ax.set_xlabel('True positive rate')
    ax.set_ylabel(f'Noise reduction factor (\# noise {len(y_pred_noise)})')
    ax.set_yscale('log')
    ax.set_xlim([0.8,1])

    if savefig_path == '':
        config['model_path'] + 'plot/' + f'model_{config["model_num"]}_noise_reduction.png'
    plt.savefig(savefig_path)


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
      
  fig, ax = plt.subplots()
  if block == 'self_attention_block':
    weight = model.encoder.layers[config['h'] - 1].self_attention_block.W_0.weight.data.numpy()
    x = ax.imshow(weight, cmap='coolwarm', interpolation='nearest')
    cbar = ax.figure.colorbar(x, ax=ax)
  elif block == 'final_binary_block':
     weight = model.final_block.linear_1.weight.data.numpy()  
     ax.plot(range(len(weight)), weight)
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
