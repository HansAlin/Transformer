import torch 
import numpy as np
from tqdm import tqdm
import pandas as pd
import subprocess
from ptflops import get_model_complexity_info

from models.models import ModelWrapper, get_n_params, build_encoder_transformer
from dataHandler.datahandler import get_model_config, get_data, save_data



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


def get_gpu_info():
    try:
        _output_to_list = lambda x: x.split('\n')[:-1]

        command = "nvidia-smi --query-gpu=index,name,utilization.gpu,memory.total,memory.used,memory.free,power.draw --format=csv"
        gpu_info = subprocess.check_output(command.split(), universal_newlines=True)
        gpu_info = _output_to_list(gpu_info)

        # the first line is the header
        gpu_info = gpu_info[1:]

        gpu_info = [x.split(', ') for x in gpu_info]
        gpu_info = [[int(x[0]), x[1], float(x[2].split(' ')[0]), x[3], x[4], x[5], x[6]] for x in gpu_info]

        return gpu_info
    except Exception as e:
        print(f"Error: {e}")
        return None


def get_energy(device_number):
    gpu_info = get_gpu_info()
    if gpu_info is None:
        return None
    try:
        energy = gpu_info[device_number][-1]
        energy = float(energy.split(' ')[0])
        return energy
    except Exception as e:
        print(f"Error: {e}")
        return None
    
def get_MMac(model, batch_size=1,  seq_len=256, channels=4, verbose=False):
  """
    This code was provided by Copilot AI
    This function calculates the number of multiply-accumulate operations (MACs)
    and the number of parameters in a model. The model must take (batch_size, seq_len, channels) as input 
    and not the more common (1, batch_size, channels, seq_len) format.
    Args:
      model: The model to calculate the MACs and parameters for.
      batch_size: The batch size to use for the model.
      seq_len: The sequence length to use for the model.
      channels: The number of channels to use for the model.

    Returns: macs, params
      macs: The number of MACs in the model.
      params: The number of parameters in the model.

  """
  wrapped_model = ModelWrapper(model, batch_size=batch_size,  seq_len=seq_len, channels=channels)

  # Specify the input size of your model
  # This should match the input size your model expects
  input_size = (batch_size,seq_len, channels)  # example input size

  # Calculate FLOPs
  macs, params = get_model_complexity_info(wrapped_model, input_size, as_strings=False,
                                          print_per_layer_stat=False, verbose=False)
  if verbose:
    print('{:<30}  {:<8}'.format('Computational complexity: ', macs))
    print('{:<30}  {:<8}'.format('Number of parameters: ', params))
  
  return macs, params    


def count_parameters(model, verbose=False):
  """ Originaly from Copilot AI
  Counts the number of parameters in a model and prints the result.
  Args:
    model: The model to count the parameters of.
    verbose: Whether to print the number of parameters or not.
    
    Returns:
        dict: A dictionary containing the following keys:
            - 'total_param': The total number of parameters in the model.
            - 'total_trainable_param': The total number of trainable parameters in the model.
            - 'encoder_param': The number of parameters in the encoder.
            - 'src_embed_param': The number of parameters in the source embedding.
            - 'final_param': The number of parameters in the final layer.
            - 'buf_param': The number of parameters in the buffer.

    Example:
      results = count_parameters(model)\n
      param_1 = results['total_param']\n
      param_2 = results['total_trainable_param']\n
      param_3 = results['encoder_param']\n
      param_4 = results['src_embed_param']\n
      param_5 = results['final_param']\n
      param_6 = results['buf_param']\n

    """
  
  total_param = 0
  total_trainable_param = 0  
  encoder_param = 0
  src_embed_param = 0
  final_param = 0
  buf_param = 0 
  for name, param in model.named_parameters():
    if param.requires_grad:
      if verbose:
        print(f"Trainable layer: {name} | Size: {param.size()} | Number of Parameters: {param.numel()}")
      total_trainable_param += param.numel()
           
    else:
      if verbose:
        print(f"Non-trainable layer: {name} | Size: {param.size()} | Number of Parameters: {param.numel()}")

    total_param += param.numel()  
    if 'encoder' in name:
      encoder_param += param.numel()
    elif 'src_embed' in name:
       src_embed_param += param.numel()
    elif 'final' in name:
      final_param += param.numel() 


  for name, buf in model.named_buffers():
    if verbose:
      print(f"Buffer: {name} | Size: {buf.size()} | Number of Parameters: {buf.numel()}")
    total_param += buf.numel()
    buf_param += buf.numel()
  if verbose:
    print(f'\nTotal Encoder Parameters: {encoder_param}')
    print(f'\nTotal src_embed Parameters: {src_embed_param}')
    print(f'\nTotal final Parameters: {final_param}')
    print(f'\nTotal Buffer Parameters: {buf_param}')  
    print(f'\nTotal Trainable Number of Parameters: {total_trainable_param}')
    print(f'\nTotal Number of Parameters: {total_param}')

  return {
        'total_param': total_param,
        'total_trainable_param': total_trainable_param,
        'encoder_param': encoder_param,
        'src_embed_param': src_embed_param,
        'final_param': final_param,
        'buf_param': buf_param
    }

def get_results(model_num, device=0):

    config = get_model_config(model_num=model_num)
    model = build_encoder_transformer(config)
    torch.cuda.set_device(device)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_loader, val_loader, test_loader = get_data(batch_size=config['batch_size'], seq_len=config['seq_len'], subset=False)
    del train_loader
    del val_loader
    y_pred_data, accuracy, efficiency, precission = test_model(model, test_loader, device, config)
    save_data(config=config, y_pred_data=y_pred_data)
    return accuracy, efficiency, precission

