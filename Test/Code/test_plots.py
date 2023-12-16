import os
from os.path import dirname, abspath, join
import sys
import unittest
import pickle

# Find code directory relative to our directory
THIS_DIR = dirname(__file__)
CODE_DIR = abspath(join(THIS_DIR, '..', ''))
sys.path.append(CODE_DIR)

from models.models_1 import TimeInputEmbeddings, LayerNormalization, FinalBinaryBlock, build_encoder_transformer
from dataHandler.datahandler import get_data, prepare_data
from training.train import test_model
import torch
import subprocess


path = os.getcwd()
subprocess.run(["python3 -m tensorboard.main --logdir=Test/ModelsResults/model_997/trainingdata"])
data_path = path + '/Test/data/test_1000_data.npy'
x_train, x_test, x_val, y_train, y_val, y_test = get_data(path='')

train_loader, val_loader, test_loader, number_of_noise, number_of_signals = prepare_data(x_train, x_val, x_test, y_train, y_val, y_test, 32)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")    
model_number = 998
with open(f"Test/ModelsResults/model_{model_number}/config.txt", 'rb') as f:
    config = pickle.load(f)

model = build_encoder_transformer(config,
                                      embed_size=config['embed_size'], 
                                      seq_len=config['seq_len'], 
                                      d_model=config['d_model'], 
                                      N=config['N'], 
                                      h=config['h'], 
                                      dropout=config['dropout'],
                                      omega=config['omega'])

test_model(model=model,
           test_loader=test_loader,
           device=device,
           config=config)
