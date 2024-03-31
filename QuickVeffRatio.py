#!/bin/env python

# python QuickVeffRatio.py /home/acoleman/data/rno-g/signal-generation/data/npy-files/veff/fLow_0.08-fhigh_0.23-rate_0.5/CDF_0.7/VeffData_nu_*.npy
# python QuickVeffRatio.py /home/acoleman/data/rno-g/signal-generation/data/npy-files/veff/fLow_0.096-fhigh_0.22-rate_0.5/CDF_1.0/VeffData_nu_mu_?c_1?.00*

import argparse
import itertools
import matplotlib
from matplotlib import pyplot as plt
import matplotlib.style as style
import numpy as np
import os
#from plotting_tools import qualitative_colors
import torch
import sys
import glob
import matplotlib.cm as cm
import time
import pandas as pd
 
CODE_DIR_1  ='/home/halin/Master/Transformer/'
sys.path.append(CODE_DIR_1)
CODE_DIR_1  ='/home/acoleman/software/NuRadioMC/'
sys.path.append(CODE_DIR_1)
CODE_DIR_2 = '/home/acoleman/work/rno-g/'
sys.path.append(CODE_DIR_2)

from NuRadioReco.utilities import units

from analysis_tools import tot
from analysis_tools.config import GetConfig
from analysis_tools.Filters import GetRMSNoise
from analysis_tools.model_loaders import LoadModelFromConfig

from models.models import build_encoder_transformer, load_model
from model_configs.config import get_config
from dataHandler.datahandler import get_model_config, get_model_path
from evaluate.evaluate import get_transformer_triggers, get_threshold

ABS_PATH_HERE = '/home/halin/Master/Transformer/'
plt.rcParams["font.family"] = "serif"
plt.rcParams["mathtext.fontset"] = "dejavuserif"

parser = argparse.ArgumentParser()

# parser.add_argument("--models", type=int, nargs='+', default=[0], help="List of models to run")
# args = parser.parse_args()
# h [108, 116, 117, 118, 119], [14, 16, 17, ]
# N [120,121,122,123,124], [111, 108, 112, 113, 114,]
# d_model [ 99, 100, 101, 102, 103, 104]
# d_ff [19, 20, 21,] [105, 101, 106, 107, 108, 109, 110]
# final_type [105, 121,] [126, 128]
# normalization [125,127]
# pos_enc_type [116, 128, 129]


test =  False
chunked = False

# if chunked:
# transformer_models = {301: 'best',
#                     #   302: 'best'
#                       }
# else:
transformer_models = {  
                        # 302: 'best',
                    #     201:'final', 
                    #   230:'early_stop.pth',
                    #   231:'231_5.pth',
                    #   232:'232_10.pth', 
                    #   233:'233_37.pth',
                    #   234:'234_16.pth',
                    #     235: '235_52.pth',  
                    # '231_best' : '231_5.pth',
                    # '231_worst' : '231_1.pth',
                    # '231_last' : '231_35.pth',
                    # '234_best' : '234_16.pth',
                    # '234_worst' : '234_48.pth',
                    # '234_last' : '234_100.pth',
                    # '235_best' : '235_52.pth',
                    # '235_worst' : '235_20.pth',
                    # '235_last' : '235_100.pth',
'240_worst' : '240_55',
'240_best' : '240_37',
'240_last' : '240_100.pth',
'241_worst' : '241_96',
'241_best' : '241_36',
'241_last' : '241_100.pth',
'242_worst' : '242_91',
'242_best' : '242_12',
'242_last' : '242_100.pth',
                     }
model_dict = {}
for model_num in transformer_models.keys():
    model_dict[str(model_num)] = { 
                       'count': 0, 
                       'time': 0,
                       'total_signals': 0,
                       'predicted_signals': 0}

best_model_dict = dict()

def qualitative_colors(length, darkening_factor=0.6):
    colors = [cm.Set3(i) for i in np.linspace(0, 1, length)]
    darker_colors = [(r*darkening_factor, g*darkening_factor, b*darkening_factor, a) for r, g, b, a in colors]
    return darker_colors


device = 2
torch.cuda.set_device(device)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}, name of GPU: {torch.cuda.get_device_name(device=device)}")
  

if chunked:
    data_path = '/mnt/md0/acoleman/rno-g/signal-generation/data/npy-files/veff/fLow_0.08-fhigh_0.23-rate_0.5/CDF_0.7/'
    file_list=glob.glob(os.path.join(data_path, "*.npz"))
else:    
    data_path = '/home/acoleman/data/rno-g/signal-generation/data/npy-files/veff/fLow_0.08-fhigh_0.23-rate_0.5/CDF_0.7/'
    file_list = glob.glob(data_path+'VeffData_nu_*.npz')

print(file_list)
# answer  =input("Continue? y/n: ")
# if answer == 'n':
#     sys.exit()


if test:
    file_list = file_list[:3]

sampling_string = data_path.split("/")[-3]
band_flow = float(sampling_string.split("-")[0].split("_")[1])
band_fhigh = float(sampling_string.split("-")[1].split("_")[1])
sampling_rate = float(sampling_string.split("-")[2].split("_")[1])
rms_noise = GetRMSNoise(band_flow, band_fhigh, sampling_rate, 300 * units.kelvin)
print(f"Scaling all values by 1 / {rms_noise / (1e-6 * units.volt):0.4f} uV to normalize to SNR")


def GetRnnTriggers(waveforms, labels, model):
    """
    waveforms: (N x 4 x time) should already be on "device"
    labels: (N x time) should already be on "device"
    model: dict with the model info and the config
    """
    predictions = torch.empty((labels.shape[0], labels.shape[1]))
    triggers = np.zeros((len(predictions)))

    with torch.no_grad():
        h = model["model"].init_hidden(1).to(device)
        for i in range(len(waveforms)):
            yhat, h = model["model"](waveforms[i].unsqueeze(0), h.detach())
            predictions[i] = yhat.cpu().squeeze()

    threshold = model["config"]["trigger"]["tot_thresh"]
    tot_bins = model["config"]["trigger"]["tot_bins"]

    for i in range(len(predictions)):
        n_sig, tp, n_noise, fp = tot.CalculateTriggerTorch(threshold, tot_bins, predictions[i], labels[i], 50, 50, 50)
        triggers[i] = tp > 0

    return triggers


def GetCnnTriggers(waveforms, trigger_times, model, pre_trig):
    """
    waveforms: (N x 4 x time) should already be on "device"
    trigger_times: (N) time in ns of each trigger to know where to cut
    model: dict with the model info and the config
    pre_trig: list of indexes where the pre-triggers are
    """
    triggers = np.zeros((len(waveforms)))

    target_length = model["config"]["input_length"]
    sampling_rate = model["config"]["sampling"]["rate"]
    upsampling = model["config"]["training"]["upsampling"]
    frac_into_waveform = model["config"]["training"]["start_frac"]

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
                yhat = model["model"](this_wvf[cut_low_bin:cut_high_bin].swapaxes(0, 1).unsqueeze(0))
                triggers[i] = yhat.cpu().squeeze() > model["config"]["trigger"]["threshold"] * 0.95
                pct_pass += 1 * triggers[i]
            except Exception as e:
                print("Yhat failed for ", this_wvf[cut_low_bin:cut_high_bin].swapaxes(0, 1).unsqueeze(0).shape)
                print(trig_bin, cut_low_bin, cut_high_bin, current_length)
                continue

    pct_pass /= len(pre_trig)
    return triggers, pct_pass



n_snr_bins = 20
snr_edges = np.linspace(1.0, 9, n_snr_bins + 1)
snr_centers = 0.5 * (snr_edges[1:] + snr_edges[:-1])

all_dat = dict()

####################################
######### Load up all the models
####################################
reference_trigger = "trig_1Hz"
pre_trigger = "trig_10kHz"
standard_triggers = [reference_trigger, pre_trigger]
rnn_model_filenames = [
    # "../trigger-dev/data/configs/fLow_0.08-fhigh_0.23-rate_0.5/hid5_lay1.yaml",
    #"../trigger-dev/data/configs/fLow_0.08-fhigh_0.23-rate_0.5/hid5_lay3.yaml",
    # "../trigger-dev/data/configs/fLow_0.08-fhigh_0.23-rate_0.5/hid10_lay5.yaml",
]
cnn_model_filenames = [

    #"../trigger-dev/data/configs/fLow_0.096-fhigh_0.22-rate_0.5/len64_filt5_kern_5_str3_mpkern2.yaml",
    #"../trigger-dev/data/configs/fLow_0.096-fhigh_0.22-rate_0.5/len80_filt5_kern_3_str1_mpkern2.yaml",
    #"../trigger-dev/data/configs/fLow_0.096-fhigh_0.22-rate_0.5/len128_filt9_9_9_9_kern3_3_3_3_str1_1_1_1_mpkern2.yaml",
    # "../trigger-dev/data/configs/fLow_0.096-fhigh_0.22-rate_0.5/len80_filt5_kern_5_str3_mpkern2.yaml",
    # "../trigger-dev/data/configs/fLow_0.096-fhigh_0.22-rate_0.5/len100_filt5_kern_5_str3_mpkern2.yaml",
    # "../trigger-dev/data/configs/fLow_0.096-fhigh_0.22-rate_0.5/len128_filt5_kern_5_str3_mpkern2.yaml",
    # "../trigger-dev/data/configs/fLow_0.096-fhigh_0.22-rate_0.5/len200_filt5_kern_5_str3_mpkern2.yaml",
    # "../trigger-dev/data/configs/fLow_0.096-fhigh_0.22-rate_0.5/len256_filt5_kern_5_str3_mpkern2.yaml",
    # "../trigger-dev/data/configs/fLow_0.096-fhigh_0.22-rate_0.5/len300_filt5_kern_5_str3_mpkern2.yaml",
    # "../trigger-dev/data/configs/fLow_0.096-fhigh_0.22-rate_0.5/len400_filt5_kern_5_str3_mpkern2.yaml",
    # "../trigger-dev/data/configs/fLow_0.096-fhigh_0.22-rate_0.5/len128_filt5_kern_5_str3_mpkern2.yaml",

    # "../trigger-dev/data/configs/fLow_0.096-fhigh_0.22-rate_0.5/len64_filt5_kern_5_str3_mpkern2.yaml",
    # "../trigger-dev/data/configs/fLow_0.096-fhigh_0.22-rate_0.5/len80_filt5_kern_3_str1_mpkern2.yaml",
    # "../trigger-dev/data/configs/fLow_0.096-fhigh_0.22-rate_0.5/len128_filt9_9_9_9_kern3_3_3_3_str1_1_1_1_mpkern2.yaml",
    # "../trigger-dev/data/configs/fLow_0.096-fhigh_0.22-rate_0.5/len65_filt5_kern_5_str3_mpkern2.yaml",
    # "../trigger-dev/data/configs/fLow_0.096-fhigh_0.22-rate_0.5/len81_filt5_kern_3_str1_mpkern2.yaml",
    # "../trigger-dev/data/configs/fLow_0.096-fhigh_0.22-rate_0.5/len129_filt9_9_9_9_kern3_3_3_3_str1_1_1_1_mpkern2.yaml",

    # "../trigger-dev/data/configs/fLow_0.096-fhigh_0.22-rate_0.5/len400_filt5_kern_5_str3_mpkern2.yaml",
    # "../trigger-dev/data/configs/fLow_0.08-fhigh_0.23-rate_0.5/len128_filt5_kern_5_str3_mpkern2.yaml",
    # "../trigger-dev/data/configs/fLow_0.08-fhigh_0.23-rate_0.5/len128_filt5_4_kern_5_3_str2_1_mpkern2.yaml",
]
chunked_model_filenames = []

all_models = dict()


def LoadModel(filename, model_list):
    config = GetConfig(filename)
    name = config["name"]
    model_list[name] = dict()
    model_list[name]["config"] = config
    model_list[name]["model"] = LoadModelFromConfig(config, device)
    model_list[name]["model"].to(device)
    model_list[name]["model"].eval()
    return name

def LoadTransformerModel(model_name, model_list, text):
    model_num = int(text.split('_')[0])
    config = get_model_config(model_num=model_num, type_of_file='yaml', )

    name = model_name
    # try:
    #     name = str(config['basic']["model_num"])
    # except:
    #     name = str(config['transformer']["basic"]["model_num"])

    threshold, sigmoid = get_threshold(config['transformer'], text=text, verbose=False)
    #threshold, sigmoid = 3, False

    print(f"Threshold for model {name}: {threshold}, sigmoid: {sigmoid}")
    if threshold == 0:
        assert False, "Model not trained"
    model_list[name] = dict()
    model_list[name]["config"] = config
    model_list[name]['data_config'] = GetConfig('/home/halin/Master/Transformer/data_config.yaml')
    model_list[name]['threshold'] = threshold
    model_list[name]['sigmoid'] = sigmoid
    
    model = load_model(config, text=text)

    model_list[name]["model"] = model
    model_list[name]["model"].to(device)
    model_list[name]["model"].eval()
    return name


for filename in cnn_model_filenames:
    name = LoadModel(filename, all_models)
    all_models[name]["type"] = "CNN"

for filename in rnn_model_filenames:
    name = LoadModel(filename, all_models)
    all_models[name]["type"] = "RNN"

for filename in chunked_model_filenames:
    name = LoadModel(filename, all_models)
    all_models[name]["type"] = "Chunk"

for model_name, model_type in transformer_models.items():
    name = LoadTransformerModel(model_name, all_models, text=model_type)
    all_models[name]["type"] = "Transformer"
    


####################################
####################################

for filename in file_list:
    print(filename)

    best_pred = 0

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

        for trig_name in standard_triggers + list(all_models.keys()):
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
    best_model = None
    for ml_trig_name in all_models.keys():
        print(f"\t{ml_trig_name}")
        if all_models[ml_trig_name]["type"] == "RNN":
            triggers = GetRnnTriggers(waveforms, signal_labels, all_models[ml_trig_name])
            n_rnn_trig = int(sum(triggers))
            n_ref_trig = int(sum(file_dat[standard_triggers[0]].astype(bool)))
            n_or_trig = int(sum(np.bitwise_or(triggers.astype(bool), file_dat[standard_triggers[0]].astype(bool))))
            print(f"\t  N_rnn: {n_rnn_trig}, N_ref: {n_ref_trig}, N_or: {n_or_trig}, % improve {n_or_trig / n_ref_trig:0.2f}")

        elif all_models[ml_trig_name]["type"] == "CNN":

            # Calculate "good" pre-trig events
            pre_trig = file_dat[pre_trigger].astype(bool)

            triggers, pct_pass = GetCnnTriggers(
                waveforms, trigger_times, all_models[ml_trig_name], pre_trig=np.argwhere(pre_trig).squeeze()
            )

            n_pre_trig = int(sum(pre_trig))
            n_cnn_trig = int(sum(triggers))
            n_ref_trig = int(sum(file_dat[standard_triggers[0]].astype(bool)))
            n_or_trig = int(sum(np.bitwise_or(triggers.astype(bool), file_dat[standard_triggers[0]].astype(bool))))
            print(
                f"\t  N_pre: {n_pre_trig}, N_cnn: {n_cnn_trig}, N_ref: {n_ref_trig}, N_or: {n_or_trig}, %det {n_cnn_trig / n_pre_trig:0.2f}, % improve {n_or_trig / n_ref_trig:0.2f}"
            )

            triggers = np.bitwise_and(triggers.astype(bool), pre_trig)

        elif all_models[ml_trig_name]["type"] == "Transformer":
                        # Calculate "good" pre-trig events
            test_variable = file_dat[pre_trigger]
            pre_trig = file_dat[pre_trigger].astype(bool)
            start_time = time.time()
            triggers, pct_pass = get_transformer_triggers(
                waveforms, trigger_times, all_models[ml_trig_name], pre_trig=np.argwhere(pre_trig).squeeze()
            )
            elapsed_time = time.time() - start_time
            
            n_pre_trig = int(sum(pre_trig)) # Number of true positive events from pre-trigger 'trig_10kHz'
            n_transform_trig = int(sum(triggers)) # Number of true positive events from transformer
            n_ref_trig = int(sum(file_dat[standard_triggers[0]].astype(bool))) # Number of true positive events from pre-trigger 'trig_1Hz'
            n_or_trig = int(sum(np.bitwise_or(triggers.astype(bool), file_dat[standard_triggers[0]].astype(bool))))
            print(
                f"\t  N_pre: {n_pre_trig}, N_trans: {n_transform_trig}, N_ref: {n_ref_trig}, N_or: {n_or_trig}, %det {n_transform_trig / n_pre_trig:0.2f}, % improve {n_or_trig / n_ref_trig:0.2f}, time: {elapsed_time:0.2f}"
            )
            model_dict[ml_trig_name]['predicted_signals'] += n_transform_trig
            model_dict[ml_trig_name]['total_signals'] += n_pre_trig
            model_dict[ml_trig_name]['time'] += elapsed_time
            if n_transform_trig > best_pred:
                best_pred = n_transform_trig
                best_model = ml_trig_name
                

            triggers = np.bitwise_and(triggers.astype(bool), pre_trig)

        elif all_models[ml_trig_name]["type"] == "Chunk":
            triggers = GetChunkTriggers(waveforms, signal_labels, all_models[ml_trig_name])

        else:
            print("Type", all_models[ml_trig_name]["type"], "is unknown")
            assert False

        ## Perform an OR with the reference trigger
        # print(f"filedat: {file_dat.files}") 
        # for file in file_dat.files:
        #     print(f"{file}: {file_dat[file].shape}")
        # true_true = np.sum(np.logical_and(triggers,                                    file_dat[standard_triggers[1]]))
        # true_false = np.sum(np.logical_and(triggers,                    np.logical_not(file_dat[standard_triggers[1]])))
        # false_true = np.sum(np.logical_and(np.logical_not(triggers),                   file_dat[standard_triggers[1]])) 
        # false_false = np.sum(np.logical_and(np.logical_not(triggers),   np.logical_not(file_dat[standard_triggers[1]])))

        # print(f"True True: {true_true}, True False: {true_false}, False True: {false_true}, False False: {false_false}")
        # labels = np.any(file_dat['label'], axis=(1,2))
        # signals = np.sum(labels)  

        trig_mask = np.bitwise_or(triggers.astype(bool), file_dat[standard_triggers[0]].astype(bool)).astype(bool)
        all_dat[lgE][flavor][current][ml_trig_name]["weight"] += np.sum(file_dat["weight"][trig_mask])

        passing_snrs = snr_values[trig_mask]
        for snr in passing_snrs:
            if snr > snr_edges[-1] or snr < snr_edges[0]:
                continue
            i_pass = np.argmin(np.abs(snr - snr_centers))
            all_dat[lgE][flavor][current][ml_trig_name]["snr_trig"][0, i_pass] += np.sum(file_dat["weight"][trig_mask])

        for snr in snr_values[trig_mask]:
            if snr > snr_edges[-1] or snr < snr_edges[0]:
                continue
            i_all = np.argmin(np.abs(snr - snr_centers))
            all_dat[lgE][flavor][current][ml_trig_name]["snr_trig"][1, i_all] += np.sum(file_dat["weight"])
    model_dict[best_model]['count'] += 1

        

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
print("Results:")
model_num_str = ''
for model_num in model_dict.keys():
    model_num_str += f'{model_num}_'
    print(f"Model number: {model_num}, count: {model_dict[model_num]['count']:>5}, time: {model_dict[model_num]['time']:>7.0f}, total signals: {model_dict[model_num]['total_signals']:>7}, predicted signals: {model_dict[model_num]['predicted_signals']:>7}, Fraction: {model_dict[model_num]['predicted_signals']/model_dict[model_num]['total_signals']:>0.3f}")                
pd.DataFrame(model_dict).to_pickle(f'/home/halin/Master/Transformer/Test/data/veff_model_{model_num_str}dict.pkl')
if test:
    plot = input("Plot? y/n: ")
    if plot == 'n':
        sys.exit()

colors = qualitative_colors(len(standard_triggers) + len(list(all_models.keys())))
markers = itertools.cycle(("s", "P", "o", "^", ">", "X"))
linestyles = itertools.cycle(("-", "--", ":", "dashdot", (0, (3, 5, 1, 5))))

nrows = 1
ncols = 1
fig, ax = plt.subplots(
    ncols=ncols, nrows=nrows, figsize=(ncols * 12 * 0.7, nrows * 8 * 0.7), gridspec_kw={"wspace": 0.2, "hspace": 0.2}
)

avg_snr_vals = dict()
for trig_name in standard_triggers + list(all_models.keys()):
    avg_snr_vals[trig_name] = np.zeros_like(snr_centers)

for lgE in all_dat.keys():
    for flavor in all_dat[lgE].keys():
        for current in all_dat[lgE][flavor].keys():

            for trig_name in standard_triggers + list(all_models.keys()):
                avg_snr_vals[trig_name] += all_dat[lgE][flavor][current][trig_name]["snr_trig"][0]


for i, trig_name in enumerate(standard_triggers + list(all_models.keys())):
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
plot_name = ''
for model_num in transformer_models:
    plot_name += f'_{model_num}'
filename = os.path.join(ABS_PATH_HERE, "figures/", f"EfficiencyVsSNR{plot_name}.png")
print("Saving", filename)
fig.savefig(filename, bbox_inches="tight")
plt.close()


colors = qualitative_colors(len(standard_triggers) + len(list(all_models.keys())))
markers = itertools.cycle(("s", "P", "o", "^", ">", "X"))
linestyles = itertools.cycle(("-", "--", ":", "dashdot", (0, (3, 5, 1, 5))))

nrows = 1
ncols = 1
fig, ax = plt.subplots(
    ncols=ncols, nrows=nrows, figsize=(ncols * 12 * 0.7, nrows * 8 * 0.7), gridspec_kw={"wspace": 0.2, "hspace": 0.2}
)

sorting_index = np.argsort(lgEs)
lgEs = np.array(lgEs)[sorting_index]

for i, name in enumerate(standard_triggers + list(all_models.keys())):
    marker = next(markers)
    linestyle = next(linestyles)

    avg = np.array(avg_veff[name]) / np.array(avg_veff[reference_trigger])


    avg = np.array(avg)[sorting_index]

    print(lgEs)
    print(avg)

    ax.plot(
        lgEs,
        avg,
        label=name.replace("_", " "),
        color=colors[i],
        marker=marker,
        #linestyle=linestyle,
    )
style.use('seaborn-colorblind')
ymin, ymax = ax.get_ylim()
ax.set_ylim(0.9, ymax)
ax.legend(prop={"size": 6})
ax.set_xlabel(r"lg(E$_{\nu}$ / eV)")
ax.set_ylabel(r"V$_{\rm eff}$ / (" + reference_trigger.replace("_", " ") + ")")
ax.tick_params(axis="both", which="both", direction="in")
ax.yaxis.set_ticks_position("both")
ax.xaxis.set_ticks_position("both")


filename = os.path.join(ABS_PATH_HERE, "figures/", f"QuickVeffRatio{plot_name}.png")
print("Saving", filename)
fig.savefig(filename, bbox_inches="tight")
plt.close()
