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
from dataHandler.datahandler import get_model_config, get_model_path

from analysis_tools.config import GetConfig


config = get_model_config(model_num=123)

model = build_encoder_transformer(config)
model_path = get_model_path(config)

print(f'Preloading model {model_path}')
state = torch.load(model_path)
model.load_state_dict(state['model_state_dict'])

get_quick_veff_ratio(model_list=[123],
                     data_config=GetConfig('/home/halin/Master/Transformer/data_config.yaml'),
                     device=0,
                     save_path='/home/halin/Master/Transformer/Test/ModelsResults/')


