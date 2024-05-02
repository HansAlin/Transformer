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
import time

from models.models import ModelWrapper, get_n_params, build_encoder_transformer, load_model, ResidualConnection, MultiHeadAttentionBlock, InputEmbeddings, PositionalEncoding, FinalBlock, FeedForwardBlock, CnnInputEmbeddings, BatchNormalization, LayerNormalization, ViTEmbeddings
from dataHandler.datahandler import get_model_config, get_chunked_data, save_data, get_model_path, get_data
import plots.plots as pp
CODE_DIR_1  ='/home/acoleman/software/NuRadioMC/'
sys.path.append(CODE_DIR_1)
CODE_DIR_4 = '/home/halin/Master/nuradio-analysis/'
sys.path.append(CODE_DIR_4)

from analysis_tools.Filters import GetRMSNoise
from NuRadioReco.utilities import units
from analysis_tools import data_locations


   

def test_model(model, test_loader, device, config, plot_attention=False, extra_identifier='', noise_rejection_rate=1e4, precision=torch.float32):
  """
  This function tests the model on the test set
  and returns the accuracy and true positive rate
  true negative rate, false positive rate and false negative rate
  Arg:
    model : the trained model
    test_loader : the test data loader
    device : the device to use
    config: config file for the model
    plot_attention: whether to plot the attention scores or not
    extra_identifier: an extra identifier for the plot
    noise_rejection_rate: the noise rejection rate, typical 1e4
  Return:
    y_pred_data, accuracy, efficiency, precision,threshold
  """
  if 'transformer' in config:
     config = config['transformer']

  data_type = config['architecture'].get('data_type', 'chunked')

  model.to(device).to(precision)
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

  noise_events = 0
  signal_events = 0
  max_singal_events = 1
  max_noise_events = 1

  with torch.no_grad():

    for istep in tqdm(range(len(test_loader)), disable=True):

      x_test, y_test = test_loader.__getitem__(istep)
      if plot_attention:
        for i in range(0, len(x_test)):
          
          x = x_test[i].unsqueeze(0)
          y_true = y_test[i]
          if data_type == 'chunked':
            y_true = y_true.max()
          x, y_true = x.to(device), y_true.to(device)
          if noise_events >= max_noise_events and signal_events >= max_singal_events:
              break
          if y_true == 0 and noise_events >= max_noise_events:
              continue
          if y_true == 1 and signal_events >= max_singal_events:
              continue
          if y_true == 0:
            noise_events += 1
          else:
            signal_events += 1

          y_hat = model(x)
          if data_type == 'chunked':
            x = x.permute(0, 2, 1)

          pp.plot_attention_scores(
              model,
              x,
              y_true,
              extra_identifier=extra_identifier,
              save_path=config['basic']['model_path'] + 'plot/'
          )
          #input(f"Noise {noise_events}, Signals {signal_events}. Press Enter to continue...")

           
      if data_type == 'chunked':
        y_test = y_test.max(dim=1)[0]
      x_test, y_test = x_test.to(device).to(precision), y_test.to(device).to(precision)
      y_test = y_test.squeeze() 
      outputs = model(x_test)
      # if config['training']['loss_function'] == 'BCEWithLogits':
      #   outputs = torch.sigmoid(outputs)
      y_pred.append(outputs.cpu().detach().numpy())
      y.append(y_test.cpu().detach().numpy()) 

    threshold, accuracy, efficiency, precision, recall, F1 = validate(np.concatenate(y), np.concatenate(y_pred), noise_rejection=noise_rejection_rate)

    print(f"Accuracy: {accuracy:>}, Efficiency: {efficiency}, Precision: {precision}, Recall: {recall}, F1: {F1}")

  y_pred = np.concatenate(y_pred)
  y = np.concatenate(y)
  if data_type == 'chunked':
    y_pred = y_pred.flatten()
  y_pred_data = pd.DataFrame({'y_pred': y_pred, 'y': y})

  return y_pred_data, accuracy, efficiency, precision, threshold

def test_efficieny(model, test_loader, config, identifyer, device):
    if 'transformer' in config:
      config = config['transformer']
    data_type = config['architecture']['data_type']
    threshold, _ = get_threshold(config=config, text=identifyer)

    y = []
    y_pred = []

    model.to(device)

    with torch.no_grad():
      
      for istep in tqdm(range(len(test_loader)), disable=True):

        x_test, y_test = test_loader.__getitem__(istep)

        if data_type == 'chunked':
          y_test = y_test.max(dim=1)[0]
        x_test, y_test = x_test.to(device), y_test.to(device)
        y_test = y_test.squeeze() 
        outputs = model(x_test)

        y_pred.append(outputs.cpu().detach().numpy())
        y.append(y_test.cpu().detach().numpy()) 

      y = np.concatenate(y)
      y_pred = np.concatenate(y_pred)

      if threshold is None:
        return 0
      
      TP = np.sum(np.logical_and(y == 1, y_pred > threshold))


      efficiency = TP / np.count_nonzero(y) if np.count_nonzero(y) != 0 else 0

    return efficiency

      
     

def validate(y, y_pred, noise_rejection=1e4):
  """
  This function validates the model on the data and returns the accuracy, 
  efficiency, precision, recall and F1 score and the threshold at which the
  noise rejection is achieved

  Args:
    y: the true values
    y_pred: the predicted values
    noise_rejection: the noise rejection rate, typical 1e4
    
  Returns:
    threshold, accuracy, efficiency, precision, recall, F1
    
    """
  y = y.flatten()
  y_pred = y_pred.flatten()
  count = 0

  totalt_number_of_signals = np.count_nonzero(y)
  total_number_of_noise = len(y) - totalt_number_of_signals


  for threshold in np.linspace(min(y_pred), max(y_pred), 1000):

    TP = np.sum(np.logical_and(y == 1, y_pred > threshold))
    TN = np.sum(np.logical_and(y == 0, y_pred < threshold))
    FP = np.sum(np.logical_and(y == 0, y_pred > threshold))
    FN = np.sum(np.logical_and(y == 1, y_pred < threshold))

    if (TN + FP) != 0:
      noise_rejection_rate = FP / (TN + FP)
      
  
      
      if noise_rejection_rate < 1/noise_rejection:
        accuracy = (TP + TN) / len(y)
        efficiency = TP / np.count_nonzero(y) if np.count_nonzero(y) != 0 else 0
        precision = TP / (TP + FP) if TP + FP != 0 else 0 
        recall = TP / (TP + FN) if TP + FN != 0 else 0
        F1 = 2 * (precision * recall) / (precision + recall) if precision + recall != 0 else 0

        print(f"Noise events: {total_number_of_noise:>10}, False signals: {FP:>5}, Noise rejection rate: {noise_rejection_rate:>5.4f}, TP: {TP:>5}, TN: {TN:>5}, FP: {FP:>5}, FN: {FN:>5}")
        return threshold, accuracy, efficiency, precision, recall, F1
    count += 1
  return None, 0, 0, 0, 0, 0
 
def validate_2(y, y_pred, noise_rejection=1e4):


  totalt_number_of_signals = np.count_nonzero(y)
  total_number_of_noise = len(y) - totalt_number_of_signals

  pred_noise = y_pred[y == 0]

  line = np.linspace(0,1,len(pred_noise))
  y_pred_sorted = np.sort(pred_noise)
  threshold_index = np.where(line > (1 - 1/noise_rejection))[0][0]
  threshold = y_pred_sorted[threshold_index]

  
  TP = np.sum(np.logical_and(y == 1, y_pred > threshold))
  TN = np.sum(np.logical_and(y == 0, y_pred < threshold))
  FP = np.sum(np.logical_and(y == 0, y_pred > threshold))
  FN = np.sum(np.logical_and(y == 1, y_pred < threshold))
  accuracy = (TP + TN) / len(y)
  efficiency = TP / np.count_nonzero(y) if np.count_nonzero(y) != 0 else 0
  precision = TP / (TP + FP) if TP + FP != 0 else 0
  recall = TP / (TP + FN) if TP + FN != 0 else 0
  F1 = 2 * (precision * recall) / (precision + recall) if precision + recall != 0 else 0

  print(f"Noise events: {total_number_of_noise:>10}, False signals: {FP:>5}, Noise rejection rate: {(FP / (TN + FP)):>5.4f}, TP: {TP:>5}, TN: {TN:>5}, FP: {FP:>5}, FN: {FN:>5}")


  return threshold, accuracy, efficiency, precision, recall, F1
    

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
        return 0


def get_energy(device_number):
    gpu_info = get_gpu_info()
    if gpu_info is None:
        return 0
    try:
        energy = gpu_info[device_number][-1]
        energy = float(energy.split(' ')[0])
        return energy
    except Exception as e:
        print(f"Error: {e}")
        return 0
    




def get_results(model_num, device=0):

    config = get_model_config(model_num=model_num)
    model = build_encoder_transformer(config)
    torch.cuda.set_device(device)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_loader, val_loader, test_loader = get_data(batch_size=config['training']['batch_size'], seq_len=config['architecture']['seq_len'], subset=False)
    del train_loader
    del val_loader
    y_pred_data, accuracy, efficiency, precision = test_model(model, test_loader, device, config)
    save_data(config=config, y_pred_data=y_pred_data)
    return accuracy, efficiency, precision

def find_key_in_dict(nested_dict, target_key):
    for key, value in nested_dict.items():
        if key == target_key:
            return value
        elif isinstance(value, dict):
            result = find_key_in_dict(value, target_key)
            if result is not None:
                return result
    return None

def get_transformer_triggers(waveforms, trigger_times, model_name, pre_trig):
  
  config = model_name['config']
  triggers = np.zeros((len(waveforms)))
  
  target_length = find_key_in_dict(config, 'seq_len')
  n_ant = find_key_in_dict(config, 'n_ant')
  data_config = model_name['data_config']
  sampling_rate = data_config['sampling']['rate'] 
  upsampling = data_config['training']['upsampling']
  frac_into_waveform = data_config['training']['start_frac']
  model = model_name['model']
  threshold = model_name['threshold']
  sigmoid = model_name['sigmoid']

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
              if find_key_in_dict(config, 'data_type') == 'trigger':
                x = x.transpose(1, 2)
              elif find_key_in_dict(config, 'data_type') == 'chunked':
                 x = x  
              elif find_key_in_dict(config, 'data_type') == 'phased':
                 x = x.transpose(1, 2)
                 x = x[:, :, :n_ant]   
              yhat = model(x)
              if sigmoid:
                yhat = torch.sigmoid(yhat)
              # if config['results']['TRESH_AT_10KNRF'] > yhat.cpu().squeeze() and config['basic']['model_num'] == 213:
              #   print(yhat.cpu().squeeze())
              #   pass
                 
              triggers[i] = yhat.cpu().squeeze() > threshold
              pct_pass += 1 * triggers[i]
          except Exception as e:
              print("Exception: ", str(e))
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

def test_threshold(model_num, treshold=None):
  """
    This function tests the threshold of the model based on the results, y_pred and y
    that are saved in the model folder and commes from the resluts from the test set
    after the training.

    Args:
      model_num: the model number
      treshold: the treshold to test the model with (optional)

    Returns:
      None
  """
  config = get_model_config(model_num=model_num, type_of_file='yaml')
  if 'transformer' in config:
    config = config['transformer']
  model_folder = config['basic']['model_path']
  data_path = model_folder + 'y_pred_data.pkl'
  y_pred_data = pd.read_pickle(data_path)
  print(y_pred_data.head())
  y_pred = y_pred
  y = y
  y_pred = y_pred.to_numpy()
  y = y.to_numpy()
  if treshold is not None:
    y_pred = np.where(y_pred > treshold, 1, 0)
  else:
    y_pred = np.where(y_pred > config['results']['TRESH_AT_10KNRF'], 1, 0)
  

  print(f"Accuracy: {validate(y, y_pred, metric='Accuracy')}")
  print(f"Efficiency: {validate(y, y_pred, metric='Efficiency')}")
  print(f"Precission: {validate(y, y_pred, metric='Precision')}")
  print(f"Treshold: {config['results']['TRESH_AT_10KNRF']}")
  true_negative = np.sum(np.logical_and(y == 0, y_pred == 0)) 
  false_positive = np.sum(np.logical_and(y == 0, y_pred == 1))
  false_negative = np.sum(np.logical_and(y == 1, y_pred == 0))
  true_positive = np.sum(np.logical_and(y == 1, y_pred == 1))
  total_neg = np.count_nonzero(y == 0)
  total_pos = np.count_nonzero(y == 1)
  print(f"True Negative: {true_negative} total: {total_neg}, ratio: {true_negative / total_neg}")
  print(f"False Positive: {false_positive}")
  print(f"False Negative: {false_negative}")
  print(f"True Positive: {true_positive} total: {total_pos}, ratio {true_positive / total_pos}")
  print(f"Total: {len(y)}")


def noise_rejection(model_number, model_type='final', cuda_device=0, verbose=False, ):
  """
    This function tests the noise rejection rate of the model. It loads the noise from the
    noise generation and tests the model on the noise. It then calculates the noise rejection rate.

    Arguments:
      model_number: the model number (int)
      model_type: the type of model to test, either 'final' or 'early_stop'
      cuda_device: the cuda device to use
      verbose: whether to print the results or not

    Returns:
      None
  """
  
  noise_path = '/home/halin/Master/nuradio-analysis/noise-generation/high-low-trigger/data/fLow_0.08-fhigh_0.23-rate_0.5/SNR_3.421-Vrms_5.52-mult_2-fLow_0.08-fhigh_0.23-rate_0.5-File_0000.npy'
  noise_path =     '/mnt/md0/data/trigger-development/rno-g/noise/high-low/fLow_0.08-fhigh_0.23-rate_0.5/prod_2023.03.24/SNR_3.421-Vrms_5.52-mult_2-fLow_0.08-fhigh_0.23-rate_0.5-File_0000.npy'
  noise = np.load(noise_path)   
  print(noise.shape)
  ##### Normalize the noise
  noise_mean = np.mean(noise)
  noise_std = np.std(noise)

  rms_noise_2 = GetRMSNoise(float(0.08), float(0.23), 0.5, 300 * units.kelvin)
  # noise = (noise - noise_mean) / noise_std
  rms_noise = 5.5239165283398165e-06

  noise = noise / rms_noise
  new_std = np.std(noise)
  print(f"New std: {new_std}")
  noise = torch.Tensor(noise)
  noise = noise.permute(0,2,1)

  config = get_model_config(model_num=model_number, type_of_file='yaml')

  if 'transformer' in config:
    config = config['transformer']


  seq_len = config['architecture']['seq_len']  
  noise = noise[:100000, 250:seq_len+250, :]

  if verbose:
    print(f"Mean: {noise_mean}, std: {noise_std}")
    pp.plot_examples(noise.cpu().detach().numpy(), save_path='/home/halin/Master/Transformer/figures/test_1.png')
    print(noise.shape)

  device = torch.device(f"cuda:{cuda_device}")
 


  model = load_model(config, text=model_type, verbose=False)
  model.to(device)
  chunks = torch.split(noise, 64)

  outputs = []

 
  try:
    threshold, sigmoid = get_threshold(config, text=model_type, verbose=False)
  except:
    print("No threshold found in model state")
    threshold = config['results']['TRESH_AT_10KNRF']
    sigmoid = True


  print(f"Model number: {model_number}, epoch type: {model_type}, threshold: {threshold}, sigmoid: {sigmoid}")  


  with torch.no_grad():
    for chunk in chunks:
      chunk_noise = chunk.to(device)
      output = model(chunk_noise)
      self_attention_score = 1
      if sigmoid:
        output = torch.sigmoid(output)
      output = output.cpu().detach().numpy()
      outputs.append(output)
  
  outputs = np.concatenate(outputs)
  outputs = outputs.ravel()


  
  interpret_signal = np.where(outputs > threshold, 1, 0)
  print(f"Number of noise events interpreted as signal: {np.sum(interpret_signal)}")
  print(f"Noise rejection rate: {len(outputs)/np.sum(interpret_signal)}")
  return outputs, threshold 
  
def get_threshold(config, text='final', verbose=False):
    if config['architecture']['data_type'] == 'chunked':
        path = config['basic']['model_path'] + '/config_' + str(config['basic']['model_num']) + f'_{text}_skipsize_206.npz'
        npzfile = np.load(path)
        threshold = float(npzfile['one_hz_threshold'])
        sigmoid = False
        return threshold, sigmoid
    else:
      try:
          model_path = get_model_path(config, text=text)
          states = torch.load(model_path)
          try:
            threshold = states['model_state_dict']['threshold']
          except:
            threshold = states['threshold']  
          sigmoid = False
          if verbose:
              print(f"Model path: {model_path}, Threshold: {threshold}, Sigmoid: {sigmoid}")
          return threshold, sigmoid
      except:
          if 'transformer' in config:
              config = config['transformer']
          threshold = config['results']['TRESH_AT_10KNRF']  
          sigmoid = True
          if verbose:
              print(f"Model path: {model_path}, Threshold: {threshold}, Sigmoid: {sigmoid}")
          return threshold, sigmoid

def find_best_model(config, device, save_path='', test=True):
   
    plot_attention = False

    best_efficiency = 0
    best_accuracy = 0
    best_precision = 0
    best_model_epoch = None
    best_threshold = None
    best_y_pred_data = None

    df = pd.DataFrame([], columns= ['Epoch', 'Accuracy', 'Precission', 'Efficiency', 'Threshold'])

    train_loader, val_loader, test_loader = get_data(config, subset=test)

    del train_loader, val_loader

    num_epochs = config['transformer']['results']['current_epoch'] 

    data_type = config['transformer']['architecture']['data_type']

    if data_type == 'chunked':
      noise_rejection = config['sampling']['rate']*1e9/config['input_length']
    else:
      noise_rejection = 10000

    for model_epoch in range(1, num_epochs+ 1):  
      model_path = get_model_path(config, f'{model_epoch}')
      model = load_model(config=config, text=f'{model_epoch}')

      y_pred_data, accuracy , efficiency, precision, threshold = test_model(model=model, 
                                                                  test_loader=test_loader,
                                                                  device=device, 
                                                                  config=config, 
                                                                  extra_identifier=f'_{model_epoch}',
                                                                  plot_attention=plot_attention,
                                                                  noise_rejection_rate=noise_rejection,)  
      temp_df = pd.DataFrame([[model_epoch, accuracy, precision, efficiency, threshold]],
                            columns= ['Epoch', 'Accuracy', 'Precission', 'Efficiency', 'Threshold'])
      df = pd.concat([df, temp_df], ignore_index=True)
      state_dict = torch.load(model_path)

      try:
          state_dict['model_state_dict']['threshold'] = threshold
      except:
          state_dict['threshold'] = threshold
      if not test:    
        torch.save(state_dict, model_path)

      if efficiency >= best_efficiency:
        best_efficiency = efficiency
        best_accuracy = accuracy
        best_precision = precision
        best_model_epoch = model_epoch
        best_threshold = threshold
        best_y_pred_data = y_pred_data  


      del model

    config['transformer']['results']['Accuracy'] = float(best_accuracy)
    config['transformer']['results']['Efficiency'] = float(best_efficiency)
    config['transformer']['results']['Precission'] = float(best_precision)
    config['transformer']['results']['best_epoch'] = int(best_model_epoch)
    config['transformer']['results']['best_threshold'] = float(best_threshold)

    print(f"Test efficiency for best model: {config['transformer']['results']['Efficiency']:.4f}")

    y = best_y_pred_data['y']
    y_pred = best_y_pred_data['y_pred']

    pp.histogram(y_pred, y, config, text='_best', threshold=best_threshold, save_path=save_path)
    nr_area, nse, threshold = pp.plot_performance_curve([y_pred], [y], [config], curve='nr', x_lim=[0,1], bins=1000, text='best', save_path=save_path)
    config['transformer']['results']['nr_area'] = float(nr_area)
    config['transformer']['results']['NSE_AT_10KNRF'] = float(nse)
    roc_area, nse, threshold = pp.plot_performance_curve([y_pred], [y], [config], curve='roc', bins=1000, text='best', save_path=save_path)
    config['transformer']['results']['roc_area'] = float(roc_area)
    config['transformer']['results']['NSE_AT_10KROC'] = float(nse)
    config['transformer']['results']['TRESH_AT_10KNRF'] = float(threshold)
    pp.plot_results(config['transformer']['basic']['model_num'], config['transformer'])


    x_batch, y_batch = test_loader.__getitem__(0)
    x = x_batch.cpu().detach().numpy()
    y = y_batch.cpu().detach().numpy()
      
    pp.plot_examples(x, y, config=config['transformer'], save_path=save_path)
    pp.plot_performance(config['transformer'], device, x_batch=x_batch, y_batch=y_batch, lim_value=best_threshold, save_path=save_path)
    if not test:
      save_data(config, df, y_pred_data)
      save_data(config, df, y_pred_data, text='best')
