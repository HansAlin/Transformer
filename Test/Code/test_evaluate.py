import torch
import sys

CODE_DIR_1  ='/home/halin/Master/Transformer/'
sys.path.append(CODE_DIR_1)

from evaluate.evaluate import test_model, get_results, count_parameters, get_MMac
from models.models import build_encoder_transformer
from model_configs.config import get_config
from dataHandler.datahandler import get_model_config


config = get_model_config(model_num=108)

model = build_encoder_transformer(config)
input_data = torch.ones((config['batch_size'], config['seq_len'], 4))
mac = get_MMac(model, batch_size=32,  seq_len=config['seq_len'], channels=4)
print(f"MACs: {mac}")