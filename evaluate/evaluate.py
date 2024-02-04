import torch 
import numpy as np
from tqdm import tqdm
import pandas as pd
import subprocess
from ptflops import get_model_complexity_info
import argparse
import sys
import os
from matplotlib import pyplot as plt
import matplotlib
import itertools
import glob

from models.models import ModelWrapper, get_n_params, build_encoder_transformer
from dataHandler.datahandler import get_model_config, get_data, save_data, get_model_path

CODE_DIR_1  ='/home/acoleman/software/NuRadioMC/'
sys.path.append(CODE_DIR_1)
CODE_DIR_2 = '/home/acoleman/work/rno-g/'
sys.path.append(CODE_DIR_2)

from analysis_tools.Filters import GetRMSNoise
from NuRadioReco.utilities import units


   

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
      if config['training']['loss_function'] == 'BCEWithLogits':
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
    train_loader, val_loader, test_loader = get_data(batch_size=config['training']['batch_size'], seq_len=config['architecture']['seq_len'], subset=False)
    del train_loader
    del val_loader
    y_pred_data, accuracy, efficiency, precission = test_model(model, test_loader, device, config)
    save_data(config=config, y_pred_data=y_pred_data)
    return accuracy, efficiency, precission

def get_transformer_triggers(waveforms, trigger_times, model_name, pre_trig):
  config = model_name['config']
  triggers = np.zeros((len(waveforms)))
  target_length = config['architecture']['seq_len']
  data_config = model_name['data_config']
  sampling_rate = data_config['sampling']['rate'] 
  upsampling = data_config['training']['upsampling']
  frac_into_waveform = data_config['training']['start_frac']
  model = model_name['model']

  current_length = waveforms.shape[1]
  assert current_length >= target_length

  pct_pass = 0

  with torch.no_grad():
      for i in pre_trig:
          this_wvf = waveforms[i]

          t0 = trigger_times[i]
          trig_bin = int(t0 * sampling_rate * upsampling)
          cut_low_bin = int(trig_bin - target_length * frac_into_waveform)

          if cut_low_bin < 0:
              this_wvf.roll(cut_low_bin, dims=-1)
              cut_low_bin = 0
          cut_high_bin = cut_low_bin + target_length

          if cut_high_bin >= current_length:
              backup = cut_high_bin - current_length
              cut_high_bin -= backup
              cut_low_bin -= backup

          try:
              x = this_wvf[cut_low_bin:cut_high_bin].swapaxes(0, 1).unsqueeze(0)
              x = x.transpose(1, 2)
              yhat = model.encode(x, src_mask=None)
              yhat = torch.sigmoid(yhat)
              triggers[i] = yhat.cpu().squeeze() > config['results']['TRESH_AT_10KNRF']
              pct_pass += 1 * triggers[i]
          except Exception as e:
              print("Yhat failed for ", this_wvf[cut_low_bin:cut_high_bin].swapaxes(0, 1).unsqueeze(0).shape)
              print(trig_bin, cut_low_bin, cut_high_bin, current_length)
              continue

  pct_pass /= len(pre_trig)
  return triggers, pct_pass

def qualitative_colors(length):
    colors=[]
    for i in range(length):
        colors.append(matplotlib.cm.tab10(i))

    return colors

def LoadModel(filename, model_list, device):
    config = get_model_config(model_num=123)
    name = config['basic']["model_num"]
    model_list[name] = dict()
    model_list[name]["config"] = config
    model_list[name]["model"] = build_encoder_transformer(config)
    model_list[name]["model"].to(device)
    model_list[name]["model"].eval()
    return name

def get_quick_veff_ratio(model_list,  data_config, save_path, device=0):

  ABS_PATH_HERE = str(os.path.dirname(os.path.realpath(__file__)))
  plt.rcParams["font.family"] = "serif"
  plt.rcParams["mathtext.fontset"] = "dejavuserif"

  # parser = argparse.ArgumentParser()
  # parser.add_argument("input", nargs="+", help="Input numpy files")
  # parser.add_argument("--cuda_core", type=str, default="0", help="Which core to run on")
  # args = parser.parse_args()

  torch.cuda.set_device(device)
  device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
  print(f"Using device: {device}, name of GPU: {torch.cuda.get_device_name(device=device)}")
  
  # Get noise
  data_path = '/home/acoleman/data/rno-g/signal-generation/data/npy-files/veff/fLow_0.08-fhigh_0.23-rate_0.5/CDF_0.7/'
  # file_list = glob.glob(data_path+'VeffData_nu_*.npz')
  file_list =[data_path + 'VeffData_nu_mu_cc_17.00eV.npz',
              data_path + 'VeffData_nu_mu_cc_17.25eV.npz',
              data_path + 'VeffData_nu_mu_cc_17.50eV.npz',]
  sampling_string = data_path.split("/")[-3]
  band_flow = float(sampling_string.split("-")[0].split("_")[1])
  band_fhigh = float(sampling_string.split("-")[1].split("_")[1])
  sampling_rate = float(sampling_string.split("-")[2].split("_")[1])
  rms_noise = GetRMSNoise(band_flow, band_fhigh, sampling_rate, 300 * units.kelvin)
  print(f"Scaling all values by 1 / {rms_noise / (1e-6 * units.volt):0.4f} uV to normalize to SNR")

  n_snr_bins = 20
  snr_edges = np.linspace(1.0, 9, n_snr_bins + 1)
  snr_centers = 0.5 * (snr_edges[1:] + snr_edges[:-1])

  all_dat = dict()

  reference_trigger = "trig_1Hz"
  pre_trigger = "trig_10kHz"
  standard_triggers = [reference_trigger, pre_trigger]
  
  all_models = dict()

  for filename in model_list:
    name = LoadModel(filename, all_models, device=device)
    all_models[name]["type"] = "Transformer"



  for filename in file_list:
    print(filename)

    basename = os.path.basename(filename)
    flavor = basename.split("_")[2]
    current = basename.split("_")[3]
    lgE = float(basename.split("_")[4][:-6])

    #############
    ## Init dicts
    #############
    if not lgE in all_dat.keys():
        all_dat[lgE] = dict()

    if not flavor in all_dat[lgE].keys():
        all_dat[lgE][flavor] = dict()

    if not current in all_dat[lgE][flavor].keys():
        all_dat[lgE][flavor][current] = dict()

        all_dat[lgE][flavor][current]["volume"] = 0
        all_dat[lgE][flavor][current]["n_tot"] = 0
        all_dat[lgE][flavor][current]["weight_total"] = 0

        for trig_name in standard_triggers:
            all_dat[lgE][flavor][current][trig_name] = dict()
            all_dat[lgE][flavor][current][trig_name]["weight"] = 0
            all_dat[lgE][flavor][current][trig_name]["snr_trig"] = np.zeros((2, n_snr_bins))

    ## Read in data
    file_dat = np.load(filename)
    print("\tLoaded!", file_dat["wvf"].shape)

    all_dat[lgE][flavor][current]["volume"] = file_dat["volume"]
    all_dat[lgE][flavor][current]["n_tot"] += file_dat["n_tot"]
    all_dat[lgE][flavor][current]["weight_total"] += np.sum(file_dat["weight"])

    # Take the largest value of all antennas
    snr_values = np.max(file_dat["snr"], axis=-1)

    ## Calculate the normal triggers
    for std_trig_name in standard_triggers:
        trig_mask = file_dat[std_trig_name].astype(bool)
        all_dat[lgE][flavor][current][std_trig_name]["weight"] += np.sum(file_dat["weight"][trig_mask])

        passing_snrs = snr_values[trig_mask]
        for snr in passing_snrs:
            if snr > snr_edges[-1] or snr < snr_edges[0]:
                continue
            i_pass = np.argmin(np.abs(snr - snr_centers))
            all_dat[lgE][flavor][current][std_trig_name]["snr_trig"][0, i_pass] += np.sum(file_dat["weight"][trig_mask])

        for snr in snr_values[trig_mask]:
            if snr > snr_edges[-1] or snr < snr_edges[0]:
                continue
            i_all = np.argmin(np.abs(snr - snr_centers))
            all_dat[lgE][flavor][current][std_trig_name]["snr_trig"][1, i_all] += np.sum(file_dat["weight"])

    print("\tConverting to tensor")
    waveforms = torch.Tensor(file_dat["wvf"].swapaxes(1, 2) / rms_noise).to(device)

    ## Join the labels across channels
    signal_labels = file_dat["label"]
    signal_labels = np.max(signal_labels, axis=1)
    for i in range(len(signal_labels)):
        ones = np.where(signal_labels[i] > 0)[0]
        if len(ones):
            signal_labels[i, min(ones) : max(ones)] = 1
    signal_labels = torch.Tensor(signal_labels).to(device)

    trigger_times = file_dat["trig_time"]

    for ml_trig_name in all_models.keys():
        print(f"\t{ml_trig_name}")

        if all_models[ml_trig_name]["type"] == "Transformer":

            # Calculate "good" pre-trig events
            pre_trig = file_dat[pre_trigger].astype(bool)

            triggers, pct_pass = get_transformer_triggers(
                waveforms, trigger_times, all_models[ml_trig_name], data_config, pre_trig=np.argwhere(pre_trig).squeeze()
            )

            n_pre_trig = int(sum(pre_trig))
            n_cnn_trig = int(sum(triggers))
            n_ref_trig = int(sum(file_dat[standard_triggers[0]].astype(bool)))
            n_or_trig = int(sum(np.bitwise_or(triggers.astype(bool), file_dat[standard_triggers[0]].astype(bool))))
            print(
                f"\t  N_pre: {n_pre_trig}, N_cnn: {n_cnn_trig}, N_ref: {n_ref_trig}, N_or: {n_or_trig}, %det {n_cnn_trig / n_pre_trig:0.2f}, % improve {n_or_trig / n_ref_trig:0.2f}"
            )

            triggers = np.bitwise_and(triggers.astype(bool), pre_trig)

        else:
            print("Type", all_models[ml_trig_name]["type"], "is unknown")
            assert False

        ## Perform an OR with the reference trigger
        trig_mask = np.bitwise_or(triggers.astype(bool), file_dat[standard_triggers[0]].astype(bool)).astype(bool)
        #all_dat[lgE][flavor][current][ml_trig_name]["weight"] += np.sum(file_dat["weight"][trig_mask])

        passing_snrs = snr_values[trig_mask]
        for snr in passing_snrs:
            if snr > snr_edges[-1] or snr < snr_edges[0]:
                continue
            i_pass = np.argmin(np.abs(snr - snr_centers))
            #all_dat[lgE][flavor][current][ml_trig_name]["snr_trig"][0, i_pass] += np.sum(file_dat["weight"][trig_mask])

        for snr in snr_values[trig_mask]:
            if snr > snr_edges[-1] or snr < snr_edges[0]:
                continue
            i_all = np.argmin(np.abs(snr - snr_centers))
           # all_dat[lgE][flavor][current][ml_trig_name]["snr_trig"][1, i_all] += np.sum(file_dat["weight"])


  avg_veff = dict()
  for trig_name in standard_triggers + list(all_models.keys()):
      avg_veff[trig_name] = []

  lgEs = []

  for lgE in all_dat.keys():
      lgEs.append(lgE)

      for trig_name in standard_triggers + list(all_models.keys()):
          avg_veff[trig_name].append(0.0)

      for flavor in all_dat[lgE].keys():
          for current in all_dat[lgE][flavor].keys():

              for trig_name in standard_triggers + list(all_models.keys()):
                  avg_veff[trig_name][-1] += (
                      all_dat[lgE][flavor][current][trig_name]["weight"] / all_dat[lgE][flavor][current]["n_tot"]
                  )
    
  colors = qualitative_colors(len(standard_triggers))
  markers = itertools.cycle(("s", "P", "o", "^", ">", "X"))
  linestyles = itertools.cycle(("-", "--", ":", "dashdot", (0, (3, 5, 1, 5))))

  nrows = 1
  ncols = 1
  fig, ax = plt.subplots(
      ncols=ncols, nrows=nrows, figsize=(ncols * 8 * 0.7, nrows * 5 * 0.7), gridspec_kw={"wspace": 0.2, "hspace": 0.2}
  )

  avg_snr_vals = dict()
  for trig_name in standard_triggers + list(all_models.keys()):
      avg_snr_vals[trig_name] = np.zeros_like(snr_centers)

  for lgE in all_dat.keys():
      for flavor in all_dat[lgE].keys():
          for current in all_dat[lgE][flavor].keys():

              for trig_name in standard_triggers + list(all_models.keys()):
                  avg_snr_vals[trig_name] += all_dat[lgE][flavor][current][trig_name]["snr_trig"][0]


  for i, trig_name in enumerate(standard_triggers):
      linestyle = next(linestyles)
      ax.step(
          snr_centers,
          avg_snr_vals[trig_name],
          where="mid",
          label=trig_name.replace("_", " ").replace("trig", "Standard trig"),
          c=colors[i],
          linestyle=linestyle,
      )

  ax.set_xlabel("SNR")
  ax.set_ylabel("Weighted Counts (arb)")
  ax.set_yscale("log")
  ax.set_xlim(0, max(snr_edges))
  ax.legend(prop={"size": "x-small"})

  filename = save_path + f"QuickVeffSNR.png"
  print("Saving", filename)
  fig.savefig(filename, bbox_inches="tight")
  plt.close()


  colors = qualitative_colors(len(standard_triggers))
  markers = itertools.cycle(("s", "P", "o", "^", ">", "X"))
  linestyles = itertools.cycle(("-", "--", ":", "dashdot", (0, (3, 5, 1, 5))))

  nrows = 1
  ncols = 1
  fig, ax = plt.subplots(
      ncols=ncols, nrows=nrows, figsize=(ncols * 8 * 0.7, nrows * 5 * 0.7), gridspec_kw={"wspace": 0.2, "hspace": 0.2}
  )


  for i, name in enumerate(standard_triggers + list(all_models.keys())):
      marker = next(markers)
      linestyle = next(linestyles)

      avg = np.array(avg_veff[name]) / np.array(avg_veff[reference_trigger])


      print(lgEs)
      print(avg)

      ax.plot(
          lgEs,
          avg,
          label=name.replace("_", " "),
          color=colors[i],
          marker=marker,
          linestyle=linestyle,
      )

  ymin, ymax = ax.get_ylim()
  ax.set_ylim(ymin=0.9, ymax=ymax)
  ax.legend(prop={"size": 6})
  ax.set_xlabel(r"lg(E$_{\nu}$ / eV)")
  ax.set_ylabel(r"V$_{\rm eff}$ / (" + reference_trigger.replace("_", " ") + ")")
  ax.tick_params(axis="both", which="both", direction="in")
  ax.yaxis.set_ticks_position("both")
  ax.xaxis.set_ticks_position("both")


  filename = save_path + "QuickVeffRatio.png"
  print("Saving", filename)
  fig.savefig(filename, bbox_inches="tight")