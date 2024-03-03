import torch
import sys
from itertools import zip_longest

CODE_DIR_1  ='/home/halin/Master/Transformer/'
sys.path.append(CODE_DIR_1)
CODE_DIR_1  ='/home/acoleman/software/NuRadioMC/'
sys.path.append(CODE_DIR_1)
CODE_DIR_2 = '/home/acoleman/work/rno-g/'
sys.path.append(CODE_DIR_2)

from evaluate.evaluate import test_model, get_results, count_parameters, get_MMac,  get_quick_veff_ratio, test_threshold
from models.models import build_encoder_transformer, load_model
from model_configs.config import get_config
from dataHandler.datahandler import get_model_config, get_model_path, save_data, get_trigger_data
from plots.plots import histogram, plot_performance_curve, plot_performance, plot_collections, plot_table, plot_attention_scores
from analysis_tools.config import GetConfig

#################################################################################
#  Get new values for a model                                                   #
#################################################################################
# model_number = 211
# config = get_model_config(model_num=model_number, type_of_file='yaml')
# model = load_model(config, text='early_stop')
# print()

# macs, params = get_MMac(model, config)
# results = count_parameters(model, verbose=False)
# config['num of parameters']['MACs'] = macs
# config['num of parameters']['num_param'] = results['total_param'] #
# config['num of parameters']['encoder_param'] = results['encoder_param'] # 
# config['num of parameters']['input_param'] = results['src_embed_param'] # 
# config['num of parameters']['final_param'] = results['final_param'] # 
# config['num of parameters']['pos_param'] = results['buf_param'] 
# print(f'MACs: {macs}')
# print(f'Total params: {results["total_param"]}')
# print(f'Encoder params: {results["encoder_param"]}')
# print(f'Input params: {results["src_embed_param"]}')
# print(f'Final params: {results["final_param"]}')
# print(f'Positional params: {results["buf_param"]}')

# save_data(config=config)

# train_data, val_data, test_data = get_trigger_data(seq_len=config['architecture']['seq_len'],
#                                                           batch_size=config['training']['batch_size'], 
#                                                           subset=False, save_test_set=False)
# del train_data
# del val_data

# y_pred_data, accuracy, efficiency, precission = test_model(model=model,
#                                                 test_loader=test_data,
#                                                 device=0,
#                                                 config=config,)
# config['results']['Accuracy'] = float(accuracy)
# config['results']['Efficiency'] = float(efficiency)
# config['results']['Precission'] = float(precission)
# save_path = 'Test/ModelsResults/test/'
# histogram(y_pred_data['y_pred'], y_pred_data['y'], config, text=text)
# nr_area, nse, threshold = plot_performance_curve([y_pred_data['y_pred']], [y_pred_data['y']], [config], curve='nr', x_lim=[0,1], bins=10000, text=text)
# config['results']['nr_area'] = float(nr_area)
# config['results']['NSE_AT_10KNRF'] = float(nse)
# roc_area, nse, threshold = plot_performance_curve([y_pred_data['y_pred']], [y_pred_data['y']], [config], curve='roc', bins=10000, text=text)
# config['results']['roc_area'] = float(roc_area)
# config['results']['NSE_AT_10KROC'] = float(nse)
# config['results']['TRESH_AT_10KNRF'] = float(threshold)
# save_data(config=config, y_pred_data=y_pred_data, text=text)

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
# test_threshold(model_num=model_num)

#################################################################################
# Test load weights                                                             #
#################################################################################

model_num = 214
config = get_model_config(model_num=model_num, type_of_file='yaml')
model = load_model(config, text='early_stop', verbose=True)