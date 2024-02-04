import torch
import sys
from itertools import zip_longest

CODE_DIR_1  ='/home/halin/Master/Transformer/'
sys.path.append(CODE_DIR_1)
CODE_DIR_1  ='/home/acoleman/software/NuRadioMC/'
sys.path.append(CODE_DIR_1)
CODE_DIR_2 = '/home/acoleman/work/rno-g/'
sys.path.append(CODE_DIR_2)

from evaluate.evaluate import test_model, get_results, count_parameters, get_MMac,  get_quick_veff_ratio
from models.models import build_encoder_transformer
from model_configs.config import get_config
from dataHandler.datahandler import get_model_config, get_model_path, get_data_binary_class, save_data
from plots.plots import histogram, plot_performance_curve, plot_performance, plot_collections, plot_table
from analysis_tools.config import GetConfig

#################################################################################
#  Get new values for a model                                                   #
#################################################################################
model_number = 131
config = get_model_config(model_num=model_number, type_of_file='yaml')

model = build_encoder_transformer(config)
model_path = get_model_path(config, text='early_stop')

print(f'Preloading model {model_path}')
state = torch.load(model_path)
model_keys = list(model.state_dict().keys())
model_shapes = [str(value.shape) for value in model.state_dict().values()]


state_dict = state['model_state_dict']
ignore_keys = ['hej',
            #  'encoder.layers.0.self_attention_block.relative_positional_k.embeddings_table', 
            #    'encoder.layers.0.self_attention_block.relative_positional_v.embeddings_table',
            #    'encoder.layers.1.self_attention_block.relative_positional_k.embeddings_table',
            #    'encoder.layers.1.self_attention_block.relative_positional_v.embeddings_table'
               ]
name_mapping = {
    'encoder.layers.0.residual_connections.0.norm.alpha': 'encoder.layers.0.residual_connection_1.norm.alpha',
    'encoder.layers.0.residual_connections.0.norm.bias':  'encoder.layers.0.residual_connection_1.norm.bias',
    'encoder.layers.0.residual_connections.1.norm.alpha': 'encoder.layers.0.residual_connection_2.norm.alpha',
    'encoder.layers.0.residual_connections.1.norm.bias':  'encoder.layers.0.residual_connection_2.norm.bias',
    'encoder.layers.1.residual_connections.0.norm.alpha': 'encoder.layers.1.residual_connection_1.norm.alpha',
    'encoder.layers.1.residual_connections.0.norm.bias':  'encoder.layers.1.residual_connection_1.norm.bias',
    'encoder.layers.1.residual_connections.1.norm.alpha': 'encoder.layers.1.residual_connection_2.norm.alpha',
    'encoder.layers.1.residual_connections.1.norm.bias':  'encoder.layers.1.residual_connection_2.norm.bias',
    # Add more mappings here
}

state_dict = {name_mapping.get(k, k): v for k, v in state_dict.items() if k not in ignore_keys}


loaded_keys = list(state_dict.keys())
loaded_shapes = [str(value.shape) for value in state_dict.values()]

print("Model's state_dict keys and shapes vs Loaded state_dict keys and shapes:")
for model_key, model_shape, loaded_key, loaded_shape in zip_longest(model_keys, model_shapes, loaded_keys, loaded_shapes):
    print(f"{str(model_key):<55} {str(model_shape):<22} vs {str(loaded_key):<55} {str(loaded_shape):<22}")

model.load_state_dict(state_dict)

train_data, val_data, test_data = get_data_binary_class(seq_len=config['architecture']['seq_len'],
                                                          batch_size=config['training']['batch_size'], 
                                                          subset=False, save_test_set=False)
del train_data
del val_data

y_pred_data, accuracy, efficiency, precission = test_model(model=model,
                                                test_loader=test_data,
                                                device=0,
                                                config=config,)
config['results']['Accuracy'] = float(accuracy)
config['results']['Efficiency'] = float(efficiency)
config['results']['Precission'] = float(precission)
histogram(y_pred_data['y_pred'], y_pred_data['y'], config)
nr_area, nse, threshold = plot_performance_curve([y_pred_data['y_pred']], [y_pred_data['y']], [config], curve='nr', x_lim=[0,1], bins=10000)
config['results']['nr_area'] = float(nr_area)
config['results']['NSE_AT_10KNRF'] = float(nse)
roc_area, nse, threshold = plot_performance_curve([y_pred_data['y_pred']], [y_pred_data['y']], [config], curve='roc', bins=10000)
config['results']['roc_area'] = float(roc_area)
config['results']['NSE_AT_10KROC'] = float(nse)
config['results']['TRESH_AT_10KNRF'] = float(threshold)
save_data(config=config, y_pred_data=y_pred_data)

