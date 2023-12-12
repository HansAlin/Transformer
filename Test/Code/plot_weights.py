import torch
import pickle
from os.path import dirname, abspath, join
import sys
import matplotlib.pyplot as plt
import numpy as np

THIS_DIR = dirname(__file__)
CODE_DIR = abspath(join(THIS_DIR, '..', ''))
sys.path.append(CODE_DIR)

from models.models_1 import build_encoder_transformer

def get_weights(model_number, layer_name, quiet=True):
    dic_path = f'Test/ModelsResults/model_{model_number}/config.txt'
    with open(dic_path, 'rb') as f:
        config = pickle.load(f)
    print(f"Model name: {config['model_name']}")    
    model = build_encoder_transformer(config,
                                      embed_size=config['embed_size'], 
                                      seq_len=config['seq_len'], 
                                      d_model=config['d_model'], 
                                      N=config['N'], 
                                      h=config['h'], 
                                      dropout=config['dropout'],
                                      omega=config['omega'])
    model_pth_path = config["model_path"] + f"saved_model/model_{model_number}.pth"
    print(f'Preloading model {model_pth_path}')
    state = torch.load(model_pth_path)
    model.load_state_dict(state['model_state_dict'])

    for name, param in model.named_parameters():
        if 'weight' in name:
            if not quiet:
                print(f'Layer: {name}, Shape: {param.shape}')
                print(param)
            if name == layer_name:
                print(f'Layer: {name}, Shape: {param.shape}')
              
                return param.detach().numpy()


def plot_weights(weight, model_number, layer_name, plot_type='hist', title='', x_albel='', y_label=''):
    save_path = f'Test/ModelsResults/model_{model_number}/plot/{layer_name}_{model_number}.png'
    one_dim = weight.shape[1]
    print(one_dim)
    if one_dim == 1:
        if plot_type == 'hist':
            weights = np.ones_like(weight)/len(weight)
            plt.hist(weight, weights=weights, bins=50)
            plt.xlabel(x_albel)
            plt.ylabel(y_label)
            plt.title(title)
        elif plot_type == 'curve':
            plt.plot(np.arange(len(weight)), weight) 
            plt.xlabel(x_albel)
            plt.ylabel(y_label)
            plt.title(title)   
        plt.savefig(save_path)
        plt.clf()


for model_number in range(996,1000):

    layer_name = 'src_embed.embedding.weight' 
    plot_type = 'hist'   
    weight = get_weights(model_number=model_number,
                layer_name=layer_name)
    plot_weights(weight,model_number, layer_name, 
                 plot_type=plot_type,
                 title="Embedding linear layer",
                 x_albel='d_model')
    