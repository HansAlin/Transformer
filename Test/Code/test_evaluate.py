import torch
import sys

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
model_number = 128
config = get_model_config(model_num=model_number)

model = build_encoder_transformer(config)
model_path = get_model_path(config)

print(f'Preloading model {model_path}')
state = torch.load(model_path)
model.load_state_dict(state['model_state_dict'])

train_data, val_data, test_data = get_data_binary_class(seq_len=config['seq_len'],
                                                          batch_size=config['batch_size'], 
                                                          subset=False, save_test_set=False)
del train_data
del val_data

y_pred_data, accuracy, efficiency, precission = test_model(model=model,
                                                test_loader=test_data,
                                                device=0,
                                                config=config,)
config['Accuracy'] = accuracy
config['Efficiency'] = efficiency
config['Precission'] = precission
histogram(y_pred_data['y_pred'], y_pred_data['y'], config)
nr_area, nse, threshold = plot_performance_curve([y_pred_data['y_pred']], [y_pred_data['y']], [config], curve='nr', x_lim=[0,1], bins=10000)
config['nr_area'] = nr_area
config['NSE_AT_10KNRF'] = nse
roc_area, nse, threshold = plot_performance_curve([y_pred_data['y_pred']], [y_pred_data['y']], [config], curve='roc', bins=10000)
config['roc_area'] = roc_area
config['NSE_AT_10KROC'] = nse
config['TRESH_AT_10KNRF'] = threshold
save_data(config=config, y_pred_data=y_pred_data)

