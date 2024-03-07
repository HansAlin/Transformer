import torch
import sys
from itertools import zip_longest

CODE_DIR_1  ='/home/halin/Master/Transformer/'
sys.path.append(CODE_DIR_1)
CODE_DIR_1  ='/home/acoleman/software/NuRadioMC/'
sys.path.append(CODE_DIR_1)
CODE_DIR_2 = '/home/acoleman/work/rno-g/'
sys.path.append(CODE_DIR_2)

from evaluate.evaluate import test_model, get_results, count_parameters, get_MMac,  get_quick_veff_ratio, test_threshold, noise_rejection
from models.models import build_encoder_transformer, load_model, get_FLOPs
from model_configs.config import get_config
import dataHandler.datahandler as dd
import plots.plots as pp
from analysis_tools.config import GetConfig

#################################################################################
#  Get new values for a model                                                   #
#################################################################################
# model_number = 201
# config = get_model_config(model_num=model_number, type_of_file='yaml')
# model = load_model(config, text='early_stop', verbose=True)
# # print()


# train_data, val_data, test_data = get_trigger_data(config=config,
#                                                    subset=False)
# del train_data
# del val_data

# y_pred_data, accuracy, efficiency, precission = test_model(model=model,
#                                                 test_loader=test_data,
#                                                 device=0,
#                                                 config=config['transformer'],)

# nr_area, nse, threshold = plot_performance_curve([y_pred_data['y_pred'].values], [y_pred_data['y'].values], [config], curve='nr', x_lim=[0,1], bins=10000, log_bins=False, reject_noise=1e4)
# print(f'NRF: {nr_area}')
# print(f'NSE: {nse}')
# print(f'Threshold: {threshold}')
# config['transformer']['results']['nr_area'] = float(nr_area)
# config['transformer']['results']['NSE_AT_10KNRF'] = float(nse)
# roc_area, nse, threshold = plot_performance_curve([y_pred_data['y_pred']], [y_pred_data['y']], [config], curve='roc', bins=10000, reject_noise=1e4)
# config['transformer']['results']['roc_area'] = float(roc_area)
# config['transformer']['results']['NSE_AT_10KROC'] = float(nse)
# config['transformer']['results']['TRESH_AT_10KNRF'] = float(threshold)
# save_data(config=config, y_pred_data=y_pred_data)

#################################################################################
# Test getting attention scores                                                 #
#################################################################################
# model_number = 130
# config = get_model_config(model_num=model_number, type_of_file='yaml')
# model = TransformerModel(config)
# model_path = get_model_path(config, text='')
# state = torch.load(model_path)
# state_dict = state['model_state_dict']
# ignore_keys = [
#             #  'encoder.layers.0.self_attention_block.relative_positional_k.embeddings_table', 
#             #    'encoder.layers.0.self_attention_block.relative_positional_v.embeddings_table',
#             #    'encoder.layers.1.self_attention_block.relative_positional_k.embeddings_table',
#             #    'encoder.layers.1.self_attention_block.relative_positional_v.embeddings_table'
#                ]
# name_mapping = {
#     'encoder.layers.0.residual_connections.0.norm.alpha': 'encoder.layers.0.residual_connection_1.norm.alpha',
#     'encoder.layers.0.residual_connections.0.norm.bias':  'encoder.layers.0.residual_connection_1.norm.bias',
#     'encoder.layers.0.residual_connections.1.norm.alpha': 'encoder.layers.0.residual_connection_2.norm.alpha',
#     'encoder.layers.0.residual_connections.1.norm.bias':  'encoder.layers.0.residual_connection_2.norm.bias',
#     'encoder.layers.1.residual_connections.0.norm.alpha': 'encoder.layers.1.residual_connection_1.norm.alpha',
#     'encoder.layers.1.residual_connections.0.norm.bias':  'encoder.layers.1.residual_connection_1.norm.bias',
#     'encoder.layers.1.residual_connections.1.norm.alpha': 'encoder.layers.1.residual_connection_2.norm.alpha',
#     'encoder.layers.1.residual_connections.1.norm.bias':  'encoder.layers.1.residual_connection_2.norm.bias',
#     # Add more mappings here
# }

# state_dict = {name_mapping.get(k, k): v for k, v in state_dict.items() if k not in ignore_keys}

# model.load_state_dict(state_dict)
# x = torch.rand(1, 127, 4)
# plot_attention_scores(model, x, save_path='/home/halin/Master/Transformer/Test/presentation/attention_scores.png')


#################################################################################
# Compare early stop and last saved model
#################################################################################
# models = [201, 202, 203, 204, 205, 206, 207, 208, 209, 210]
# print('{:<15} {:<10} {:<10}'.format('Model number', 'NSE early', 'NSE final'))
# for model in models:
#     final_config = get_model_config(model_num=model, type_of_file='yaml')
#     ealry_stop_config = get_model_config(model_num=model, type_of_file='yaml', sufix='_early_stop')
#     final_nse = final_config['results']['NSE_AT_10KNRF']
#     early_nse = ealry_stop_config['results']['NSE_AT_10KNRF']
#     print(f'{model:<15} {early_nse:<10.6f} {final_nse:<10.6f}')

#################################################################################
# Test thresholds                                                               #   
# #################################################################################
# model_num =201
# test_threshold(model_num=model_num, treshold=None)

#################################################################################
# Test load weights                                                             #
#################################################################################

# model_num = 213
# config = get_model_config(model_num=model_num, type_of_file='yaml')
# model = load_model(config, text='early_stop', verbose=True)

#################################################################################
# Test noise rejection                                                          #
#################################################################################
# models = [201, 202, 213]
# values = []
# for model_num in models:
#     value = noise_rejection(model_number=model_num, verbose=True, model_type='early_stop')
#     values.append(value)

# pp.plot_hist(values=values, 
#              names=models, 
#              bins=100, 
#              log=False,
#              xlim=0.1,
#              ylim=4)
#################################################################################
# Test get_FLOPs                                                                #
#################################################################################
# model_num = 213
# verbose = False
# config = get_model_config(model_num=model_num, type_of_file='yaml')
# model = load_model(config, text='final', verbose=verbose)
# flops = get_FLOPs(model=model, config=config, verbose=verbose)
# MACs = get_MMac(model=model, config=config['transformer'])
# print(f'FLOPs: {flops}')
# print(f'MACs: {MACs}')

#################################################################################
# Test attention scores
#################################################################################
#model_num

#################################################################################
# Test perfomance
#################################################################################
model_nums = [213,213,213,213]
texts = ['early_stop', 'early_stop', 'final', 'final']
text2s = ['no positional encoding', 'positional encoding', 'no positional encoding', 'positional encoding']
data_config = dd.get_data_config()
train_data, val_data, test_data = dd.get_trigger_data(config=data_config,
                                                   subset=False)
del train_data
del val_data

models = []
textss = []
text2ss = []
AOCs = []
NSEs = []
thresholds = []


for model_num, text, text2 in zip_longest(model_nums, texts, text2s):
    config = dd.get_model_config(model_num=model_num, type_of_file='yaml')
    if text2 == 'no positional encoding':
        config['transformer']['architecture']['pos_enc_type'] = 'None'
    model = load_model(config, text=text, verbose=False)


    y_pred_data, accuracy, efficiency, precission = test_model(model=model,
                                                    test_loader=test_data,
                                                    device=0,
                                                    config=config['transformer'],)

    AOC, nse, threshold =  pp.plot_performance_curve([y_pred_data['y_pred']], 
                                                     [y_pred_data['y']], 
                                                     [config], 
                                                     curve='nr', 
                                                     x_lim=[0.8,1], 
                                                     bins=10000, 
                                                     log_bins=False, 
                                                     reject_noise=1e4,
                                                     save_path=f'/home/halin/Master/Transformer/figures/performance_{text}_{text2}_',
                                                     text= f'{text} {text2}',
                                                     )

    pp.histogram(y_pred_data['y_pred'], 
                y=y_pred_data['y'], 
                config=config['transformer'],
                bins=100, 
                save_path=f'/home/halin/Master/Transformer/figures/performance_{text}_{text2}_',
                text= f'{text} {text2}',
                )
    models.append(model_num)
    textss.append(text)
    text2ss.append(text2)
    AOCs.append(AOC)
    NSEs.append(nse)
    thresholds.append(threshold)
    
print(f"{'Model number':<15} {'Which model':<15} {'Pos. type':<25} {'AOC':<15} {'NSE':<15} {'Threshold':<15}")
for model_num, text, text2, AOC, nse, threshold in zip_longest(model_nums, texts, text2s, AOCs, NSEs, thresholds):    
    print(f'{model_num:<15} {text:<15} {text2:<25} {AOC:<15.6f} {nse:<15.6f} {threshold:<15.6f}')