import torch
import sys

CODE_DIR_1  ='/home/halin/Master/Transformer/'
sys.path.append(CODE_DIR_1)

from evaluate.evaluate import test_model
from models.models import get_n_params, build_encoder_transformer
from dataHandler.datahandler import get_model_config, get_data, save_data

def get_results(model_num, device=0):

    config = get_model_config(model_num=model_num)
    model = build_encoder_transformer(config)
    torch.cuda.set_device(device)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_loader, val_loader, test_loader = get_data(batch_size=config['batch_size'], seq_len=config['seq_len'], subset=False)
    del train_loader
    del val_loader
    y_pred_data, accuracy, efficiency, precission = test_model(model, test_loader, device, config)
    save_data(config=config, y_pred_data=y_pred_data)

get_results(114)    