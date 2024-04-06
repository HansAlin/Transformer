import os
from os.path import dirname, abspath, join
import sys
import unittest
import pickle
import numpy as np
# Find code directory relative to our directory
CODE_DIR_1  ='/home/halin/Master/Transformer/'
sys.path.append(CODE_DIR_1)
# type(sys.path)
# for path in sys.path:
#    print(path)
from models.models import InputEmbeddings, LayerNormalization, FinalBlock, build_encoder_transformer
import dataHandler.datahandler as dd
# from training.train import test_model
import plots.plots as pp
import torch
import subprocess
import pandas as pd
import time
import matplotlib.pyplot as plt
import re

def condense_sequence(match):
    numbers = list(map(int, match.group(0).split('_')))
    return f'{numbers[0]}-{numbers[-1]}'

# path = os.getcwd()

# data_path = path + '/Test/data/test_1000_data.npy'
# x_train, x_test, x_val, y_train, y_val, y_test = get_data(path=data_path)

# train_loader, val_loader, test_loader, number_of_noise, number_of_signals = prepare_data(x_train, x_val, x_test, y_train, y_val, y_test, 32)

# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")    
# model_number = 992
# with open(f"/mnt/md0/halin/Models/model_{model_number}/config.txt", 'rb') as f:
#     config = pickle.load(f)
# path = config['basic']['model_path']
# save_path = f'/home/halin/Master/Transformer/Test/ModelsResults/test/model_{model_number}_noise_reduction.png'
# df = pd.read_pickle(path + 'y_pred_data.pkl')
# noise_reduction_factor([df['y_pred']], [df['y']], [config], save_path=save_path, x_lim=[0,1])
#histogram(df['y_pred'], df['y'], config,)

# model = build_encoder_transformer(config)

# blocks = [ 'final_binary_block']
# for block in blocks:
#     plot_weights(model, config, block=block, quiet=True)


# ###################################################################
# #  Plot collections of noise reduction factors or roc             #
# ###################################################################
# models_path = '/mnt/md0/halin/Models/'
# models = [19, 131]
# curve = 'nr'
# parameter = 'N'
# bins = 10000

# str_models = '_'.join(map(str, models))
# #save_path = f"/mnt/md0/halin/Models/collections/{str_models}_{curve}.png"
# save_path = f'/home/halin/Master/Transformer/Test/presentation/model_{str_models}_{curve}.png'


# hyper_parameters = find_hyperparameters(model_number=models, 
#                                         parameter=parameter,
#                                         models_path=models_path)
# labels = {'hyper_parameters': hyper_parameters, 'name': f'Model num.,  {parameter}'}

# plot_collections(models, 
#                  labels, 
#                  save_path=save_path, 
#                  models_path=models_path,
#                  x_lim=[0.8,1],
#                  curve=curve,
#                  bins=bins)
# df = collect_config_to_df(model_numbers=models, save_path='/home/halin/Master/Transformer/Test/ModelsResults/collections/', save=False)
# df = df.sort_values('NSE_AT_10KNRF', ascending=False)
# keys = ['model_num', 'd_model', 'd_ff', 'N', 'h', 'input_param','encoder_param', 'final_param', 'NSE_AT_10KNRF', 'training_time']
# for key in keys:
#     if key not in df.columns:
#         df[key] = np.nan
# print(df[keys])

###################################################################
# Plot tabel of hyperparameters                                   #
###################################################################
# # models_path = '/mnt/md0/halin/Models/'

# models = [118,119]
# str_models = '_'.join(map(str, models))
# sort = True
# if sort:
#     save_path = f'/home/halin/Master/Transformer/Test/presentation/model_{str_models}_table_sorted.png'
#     df = collect_config_to_df(model_numbers=models, save_path=save_path)
#     df = df.sort_values('NSE_AT_10KNRF', ascending=False)
# else:
#     save_path = f'/home/halin/Master/Transformer/Test/presentation/model_{str_models}_table.png'
#     df = collect_config_to_df(model_numbers=models, save_path=save_path)
# pattern = r'(\d+(_\d+)+)'

# save_path = re.sub(pattern, condense_sequence, save_path)
# keys = ['model_num', 'pos_enc_type','d_model', 'd_ff', 'N', 'h', "num_param", 'NSE_AT_10KNRF', 'MACs' ]
# plot_table(df, keys, save_path=save_path, print_common_values=True)

###################################################################
# Plot performance of a model                                     #
###################################################################
# num = 4
# plot_performance(model_num=num, lim_value=0.1, save_path=f'/home/halin/Master/Transformer/Test/ModelsResults/test/model_{num}_')


###################################################################
# Plot single curves                                                     #
###################################################################
# model_nums = [123]
# window_pred = False
# bins = 10000
# curve = 'nr'
# save_path = f'/home/halin/Master/Transformer/Test/ModelsResults/collections/test_{curve}_threshold.png'
# for model_num in model_nums:
#     config_path = models_path + f'model_{model_num}/config.txt'
#     with open(config_path, 'rb') as f:
#         config = pickle.load(f)
#     y_data_path = models_path + f'model_{model_num}/y_pred_data.pkl'    
#     with open(y_data_path, 'rb') as f:
#         y_data = pickle.load(f)
#     y_pred = np.asarray(y_data['y_pred'])    
#     y = np.asarray(y_data['y'])
#     area, nse, threshold = plot_performance_curve(ys=[y], 
#                         y_preds=[y_pred], 
#                         configs=[config], 
#                         save_path=save_path, 
#                         labels=None,
#                         x_lim=0.8,
#                         bins=bins,
#                         curve='roc',
#                         log_bins=False,
#                         reject_noise=1e4)
#     print(f'Model number: {model_num} Area under curve: {area:.4f}, NSE: {nse:.4f}, Threshold: {threshold:.4f}')


###################################################################
# Plot multi curves                                               #
###################################################################
# model_nums = [123]
# window_pred = False
# bins = 100
# save_path = '/home/halin/Master/Transformer/Test/ModelsResults/collections/test_threshold.png'
# labels = None
# for model_num in model_nums:
#     plot_collections([model_num], 
#                     labels, 
#                     save_path=save_path, 
#                     models_path=models_path,
#                     x_lim=[0,1],
#                     window_pred=window_pred,
#                     curve='nr',
#                     bins=bins,
#                     log_bins=True, 
#                     reject_noise=1e4)

    # plot_collections([model_num], 
    #                 labels, 
    #                 save_path=save_path, 
    #                 models_path=models_path,
    #                 x_lim=[0,1],
    #                 window_pred=window_pred,
    #                 curve='roc',
    #                 bins=bins,
    #                 log_bins=True)

#######################################################################
# Plot results from training                                          #
# #######################################################################
# model_num = 129
# config = get_model_config(model_num=model_num)
# plot_results(model_num, config)

#######################################################################
# Plot dataframe                            #
# #######################################################################

# def try_convert_to_int(value):
#     try:
#         return int(value), True
#     except ValueError:
#         return value, False

# model_nums = [240,241,242]
# extra_identifier = '_test_of_noise_content_3'
# for model_num in model_nums:
#     df = pd.read_pickle(f'/home/halin/Master/Transformer/Test/data/epoch_data_model_{model_num}{extra_identifier}.pkl')

#     epochs = df['Epoch'].values
#     efficiencies = df['Efficiency'].values
#     thresholds = df['Threshold'].values


#     epoch_ints = []
#     epoch_strs = []


#     efficiency_int = []
#     efficiency_str = []

#     thresholds_int = []
#     thresholds_str = []

#     for epoch, efficiency, threshold in zip(epochs, efficiencies, thresholds):

#         value, is_int = try_convert_to_int(epoch)

#         if is_int:
#             epoch_ints.append(value)
#             efficiency_int.append(efficiency)
#             thresholds_int.append(threshold)
#         else:
#             epoch_strs.append(epoch)
#             efficiency_str.append(efficiency)
#             thresholds_str.append(threshold)

#     epoch_strs = []
#     count = 0
#     for value in efficiency_str:
#         try:
#             index = epoch_ints.index(value)
#             # print(f"The value {value} is at index {index} in the list.")
#             epoch_strs.append(value + 1)
#         except ValueError:
#             # print(f"The value {value} is not in the list.")
#             epoch_strs.append(count + 1)
#             count += 1
    
#     epoch_ints = np.array(epoch_ints)
#     sorta = np.argsort(epoch_ints)
#     epoch_ints = epoch_ints[sorta]
#     efficiency_int = np.array(efficiency_int)[sorta]
#     thresholds_int = np.array(thresholds_int)[sorta]

#     # for i, epoch in enumerate(epoch_ints):
#     #     if model_num == 240 and epoch == 37 or model_num == 241 and epoch == 36 or model_num == 242 and epoch == 12:
#     #         print(f"Model {model_num}, Epoch: {epoch}, Efficiency: {efficiency_int[i]:.4f}, Threshold: {thresholds_int[i]:.4f}")

#     int_plot = False
#     if len(epoch_strs) > 0:
#         plt.plot(epoch_ints, efficiency_int, label=f'Model {model_num}')
#         plt.ylim(0.0,1.0)
#         int_plot = True
#     if len(epoch_strs) > 0 and not int_plot:
#         plt.scatter(epoch_strs, efficiency_str)    
   
#     plt.xlabel('Epoch')
#     plt.ylabel('Efficiency')
#     plt.legend()
#     plt.grid()
#     plt.show()
#     plt.savefig(f'/home/halin/Master/Transformer/figures/efficiency/efficiency_model_{model_num}{extra_identifier}.png')
#     plt.close()
#     best_index = df['Efficiency'].idxmax()
#     worst_index = df['Efficiency'].idxmin()
#     alt_worst_index = np.where(efficiency_int == np.min(efficiency_int))[0][0] + 1
#     last_index = epoch_ints[-1]
#     last_efficency = efficiency_int[-1].item()
#     print(f"Model number: {model_num:>10}")
#     print(f"  Best efficiency: {df['Efficiency'][best_index]:>5.4f} and threshold: {df['Threshold'][best_index]:>5.4f} at epoch {df['Epoch'][best_index]:>5}, ")
#     print(f"  Worst efficiency: {df['Efficiency'][worst_index]:>5.4f} and threshold: {df['Threshold'][worst_index]:>5.4f} at epoch {df['Epoch'][worst_index]:>5}, ")
#     print(f" Last efficiency: {last_efficency:>5.4f} and threshold: {thresholds[-1]:>5.4f} at epoch {last_index:>5}, ")
#     # print(f"'{model_num}_worst' : '{model_num}_{df['Epoch'][worst_index]}',")
#     # print(f"'{model_num}_best' : '{model_num}_{df['Epoch'][best_index]}',")
#     # print(f"'{model_num}_last' : '{model_num}_{last_index}.pth',")


########################################################################
# Plot results                                                          #
########################################################################

# model_num = 234
# config = dd.get_model_config(model_num)
# plot_path = f'/home/halin/Master/Transformer/figures/loss/'
# pp.plot_results(model_number=model_num, path=plot_path, config=config['transformer'])


########################################################################
# Plot veff
########################################################################
veff_files = [
    '/home/halin/Master/nuradio-analysis/plots/fLow_0.08-fhigh_0.23-rate_0.5/config_310/veff_n_lges_040_0_average_rolls_0.npz',
     '/home/halin/Master/nuradio-analysis/plots/fLow_0.08-fhigh_0.23-rate_0.5/config_311/veff_n_lges_085_0_average_rolls_0.npz',
    '/home/halin/Master/nuradio-analysis/plots/fLow_0.08-fhigh_0.23-rate_0.5/config_312/veff_n_lges_017_0_average_rolls_0.npz'

]
models = [241,245,246]
veff_files = []
for model in models:
    veff_files.append(f'/home/halin/Master/Transformer/figures/QuickVeffRatio_{model}_best.npz')

pp.plot_veff(veff_files)

########################################################################
# Test roc nr curve
########################################################################

# model_nums = [240,241,242,243,244,245]
# y_preds = []
# ys = []
# configs = []
# labels = {'hyper_parameters': ['' for _ in range(len(model_nums))],
#           'name': 'Model number'}
# curve_type = 'nr'

# for model_num in model_nums:
#     configs.append(dd.get_model_config(model_num, type_of_file='yaml'))
#     y_data_path = f'/mnt/md0/halin/Models/model_{model_num}/y_pred_data.pkl'
#     with open(y_data_path, 'rb') as f:
#         y_data = pickle.load(f)
#     y_preds.append(np.asarray(y_data['y_pred']))
#     ys.append(np.asarray(y_data['y']))
# path_str = '_'.join(map(str, model_nums))    
# save_path = f'/home/halin/Master/Transformer/figures/roc/{curve_type}_{path_str}.png'
# area, nse, threshold = pp.plot_performance_curve(ys=ys, 
#                     y_preds=y_preds, 
#                     configs=configs, 
#                     save_path=save_path, 
#                     labels=labels,
#                     x_lim=0.8,
#                     bins=1000,
#                     curve=curve_type,
#                     log_bins=False,
#                     reject_noise=1e4)


########################################################################
# test plot examples
########################################################################
# batch_size = 32
# seq_len = 128
# n_ant = 4
# y = np.random.choice([0,1], size=batch_size)
# shape = (batch_size, seq_len, n_ant)
# x = np.random.randn(*shape)
# pp.plot_examples(x,y)