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
from dataHandler.datahandler import get_model_config, get_model_path, get_value
from evaluate.evaluate import get_transformer_triggers, get_threshold

ABS_PATH_HERE = '/home/halin/Master/Transformer/'
plt.rcParams["font.family"] = "serif"
plt.rcParams["mathtext.fontset"] = "dejavuserif"


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--models", type=int, nargs='+', required=True, help="List of models to run")
    parser.add_argument("--save_path", default='', help="Path to save the plot")
    parser.add_argument("--cuda_device", type=int, default=0, help="Which cuda device to use")
    args = parser.parse_args()
    return args


def qualitative_colors(length, darkening_factor=0.6):
    colors = [cm.Set3(i) for i in np.linspace(0, 1, length)]
    darker_colors = [(r*darkening_factor, g*darkening_factor, b*darkening_factor, a) for r, g, b, a in colors]
    return darker_colors


def LoadModel(filename, model_list, device):
    config = GetConfig(filename)
    name = config["name"]
    model_list[name] = dict()
    model_list[name]["config"] = config
    model_list[name]["model"] = LoadModelFromConfig(config, device)
    model_list[name]["model"].to(device)
    model_list[name]["model"].eval()
    return name

def LoadTransformerModel(model_name, model_list, text, device):
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
    data_type = get_value(config, 'data_type' )
    model_list[name]['data_config'] = GetConfig(f'/home/halin/Master/Transformer/{data_type}_data_config.yaml')
    model_list[name]['data_config']['input_length'] = get_value(config, 'seq_len')
    model_list[name]['data_config']['n_ant'] = get_value(config, 'n_ant')
    model_list[name]['data_config']['training']['batch_size'] = get_value(config, 'batch_size')
    model_list[name]['threshold'] = threshold
    model_list[name]['sigmoid'] = sigmoid
    
    model = load_model(config, text=text)

    model_list[name]["model"] = model
    model_list[name]["model"].to(device)
    model_list[name]["model"].eval()
    return name

def veff(models, device, save_path=None, test=False): 
    """ Calculate the effective volume for a given model or models. 
        The 1 Hz triger is used as a reference trigger and is equal to one. 
        All the other triggers are compared to this trigger.

        Args:
            models (list, int, dict): List of models to run, or a single model number. 
                                        If a single model number is given, the best epoch is used. 
                                        If a dictionary is given, the keys are the model numbers and the values are the model epoch.
            device (int): Which cuda device to use
            save_path (str): Path to save the plot
            test (bool): If True, only two files are loaded
    
    """
    if type(models) == list or type(models) == int:

        if type(models) == int:
            config = get_model_config(model_num=models)
            antenna_type = get_value(config, 'antenna_type')
            if save_path is None:
                save_path = get_value(config, 'model_path') + 'plot/'
            
            models = [models]
        else:
            if save_path is None:
                save_path = '/home/halin/Master/Transformer/figures/veff/'


        transformer_models = {}

        for model_num in models:
            config = get_model_config(model_num=model_num)
            antenna_type = get_value(config, 'antenna_type')

            data_type = get_value(config, 'data_type')
            best_epoch = get_value(config, 'best_epoch')


            transformer_models[f'{model_num}_best'] = f'{model_num}_{best_epoch}.pth'
        print(f"Models: {transformer_models}")

    elif type(models) == dict:
        antenna_type = 'LPDA'
        transformer_models = models
        print(f"Models: {transformer_models}")


    extra_identifier = ''

    model_dict = {}
    for model_num in transformer_models.keys():
        model_dict[str(model_num)] = { 
                        'count': 0, 
                        'time': 0,
                        'total_signals': 0,
                        'predicted_signals': 0}

    best_model_dict = dict()




    torch.cuda.set_device(device)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}, name of GPU: {torch.cuda.get_device_name(device=device)}")
    

    
    if antenna_type == 'LPDA':    
        #data_path = '/home/acoleman/data/rno-g/signal-generation/data/npy-files/veff/fLow_0.08-fhigh_0.23-rate_0.5/CDF_0.7/'
        data_path = '/mnt/md0/data/trigger-development/rno-g/veff/fLow_0.08-fhigh_0.23-rate_0.5/prod_2023.11.27/CDF_0.7/'
        file_list = glob.glob(data_path+'VeffData_nu_*.npz')
    elif data_type == 'phased':
        data_path = '/mnt/md0/data/trigger-development/rno-g/veff/fLow_0.096-fhigh_0.22-rate_0.5/prod_2024.04.12/CDF_1.0/'
        file_list = glob.glob(data_path+'VeffData_nu_*.npz')


    if test:
        file_list = file_list[:2]

    for file in file_list:
        print(file)


    sampling_string = data_path.split("/")[-4]
    band_flow = float(sampling_string.split("-")[0].split("_")[1])
    band_fhigh = float(sampling_string.split("-")[1].split("_")[1])
    sampling_rate = float(sampling_string.split("-")[2].split("_")[1])
    rms_noise = GetRMSNoise(band_flow, band_fhigh, sampling_rate, 300 * units.kelvin)
    print(f"Scaling all values by 1 / {rms_noise / (1e-6 * units.volt):0.4f} uV to normalize to SNR")


    n_snr_bins = 20
    snr_edges = np.linspace(1.0, 9, n_snr_bins + 1)
    snr_centers = 0.5 * (snr_edges[1:] + snr_edges[:-1])

    all_dat = dict()

    ####################################
    ######### Load up all the models
    ####################################
    reference_trigger = "trig_1Hz"
    pre_trigger = "trig_10kHz"
    standard_triggers = [reference_trigger, pre_trigger, 'trig_2sigma', 'trig_3sigma']
    rnn_model_filenames = []
    cnn_model_filenames = []
    chunked_model_filenames = []

    all_models = dict()





    for filename in cnn_model_filenames:
        name = LoadModel(filename, all_models, device)
        all_models[name]["type"] = "CNN"

    for filename in rnn_model_filenames:
        name = LoadModel(filename, all_models, device)
        all_models[name]["type"] = "RNN"

    for filename in chunked_model_filenames:
        name = LoadModel(filename, all_models, device)
        all_models[name]["type"] = "Chunk"

    for model_name, model_type in transformer_models.items():
        name = LoadTransformerModel(model_name, all_models, text=model_type, device=device)
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

        updated_standard_triggers = []
        for standard_trigger in standard_triggers:
            for key in file_dat.keys():
                if standard_trigger in key:
                    updated_standard_triggers.append(standard_trigger)

        standard_triggers = updated_standard_triggers        

        ## Calculate the normal triggers
        for std_trig_name in standard_triggers:
            trig_mask = file_dat[std_trig_name].astype(bool)
            detected_pre_trig_events = np.sum(file_dat["weight"][trig_mask])
            print(f"\t{std_trig_name}: {detected_pre_trig_events} events")  
            all_dat[lgE][flavor][current][std_trig_name]["weight"] += detected_pre_trig_events

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
            if all_models[ml_trig_name]["type"] == "Transformer":
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
        if best_model is not None:        
            model_dict[best_model]['count'] += 1

            
        flavor_ratio = {"e": 1 / 3.0, "mu": 1 / 3.0, "tau": 1 / 3.0}
        interaction_weight = {"cc": 0.7064, "nc": 1.0 - 0.7064}

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
                            flavor_ratio[flavor]
                            * interaction_weight[current]
                            * (all_dat[lgE][flavor][current][trig_name]["weight"] / all_dat[lgE][flavor][current]["n_tot"])
                        )

    print("Results:")
    model_num_str = ''
    for model_num in model_dict.keys():
        model_num_str += f'{model_num}_'
        print(f"Model number: {model_num}, count: {model_dict[model_num]['count']:>5}, time: {model_dict[model_num]['time']:>7.0f}, total signals: {model_dict[model_num]['total_signals']:>7}, predicted signals: {model_dict[model_num]['predicted_signals']:>7}, Fraction: {model_dict[model_num]['predicted_signals']/model_dict[model_num]['total_signals']:>0.3f}")      

    pd.DataFrame(model_dict).to_pickle(save_path + f'veff_model_{model_num_str}dict.pkl')

    # if test:
    #     plot = input("Plot? y/n: ")
    #     if plot == 'n':
    #         sys.exit()

    colors = qualitative_colors(len(standard_triggers) + len(list(all_models.keys())))
    markers = itertools.cycle(("s", "P", "o", "^", ">", "X"))
    linestyles = itertools.cycle(("-", "--", ":", "dashdot", (0, (3, 5, 1, 5))))

    nrows = 1
    ncols = 1
    fig, ax = plt.subplots(
        ncols=ncols, nrows=nrows, figsize=(ncols * 15 * 0.7, nrows * 10 * 0.7), gridspec_kw={"wspace": 0.2, "hspace": 0.2}
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

    plot_name += f"{extra_identifier}"  
   
    filename = os.path.join(save_path, f"EfficiencyVsSNR{plot_name}.png")
    
    print("Saving", filename)
    fig.savefig(filename, bbox_inches="tight")
    plt.close()


    colors = qualitative_colors(len(standard_triggers) + len(list(all_models.keys())))
    markers = itertools.cycle(("s", "P", "o", "^", ">", "X"))
    linestyles = itertools.cycle(("-", "--", ":", "dashdot", (0, (3, 5, 1, 5))))

    nrows = 1
    ncols = 1
    fig, ax = plt.subplots(
        ncols=ncols, nrows=nrows, figsize=(ncols * 15 * 0.7, nrows * 10 * 0.7), gridspec_kw={"wspace": 0.2, "hspace": 0.2}
    )

    sorting_index = np.argsort(lgEs)
    lgEs = np.array(lgEs)[sorting_index]
    npz_file = {}
    npz_file['lgEs'] = lgEs

    for i, name in enumerate(standard_triggers + list(all_models.keys())):
        marker = next(markers)
        linestyle = next(linestyles)

        avg = np.array(avg_veff[name]) / np.array(avg_veff[reference_trigger])


        avg = np.array(avg)[sorting_index]
        npz_file[name] = avg

        # print(lgEs)
        # print(avg)

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

    
    filename = os.path.join(save_path, f"QuickVeffRatio{plot_name}.png")
 
    print("Saving", filename)
    np.savez(filename.replace('.png', '.npz'), **npz_file)
    fig.savefig(filename, bbox_inches="tight")
    plt.close()

for model_num in [400]:
    veff(models=model_num, device=2, test=False, save_path='/home/halin/Master/Transformer/figures/veff/')