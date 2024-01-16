import os
from os.path import dirname, abspath, join
import sys
import unittest
import numpy as np
# Find code directory relative to our directory
CODE_DIR_1  ='/home/halin/Master/Transformer'
sys.path.append(CODE_DIR_1)

# from models.models import InputEmbeddings, LayerNormalization, FinalBlock, MultiHeadAttentionBlock, LearnablePositionalEncoding
from models.models import InputEmbeddings, LayerNormalization, FinalBlock, MultiHeadAttentionBlock, LearnablePositionalEncoding, build_encoder_transformer
from dataHandler.datahandler import get_data, prepare_data
from model_configs.config import get_config
import torch


# PATH = '/home/halin/Master/Transformer/Test/data/test_100_data.npy'

# x_train, x_val, x_test, y_train, y_val, y_test = get_data(path=PATH)

# train_loader, val_loader, test_loader = prepare_data(x_train, x_val, x_test, y_train, y_val, y_test, batch_size=32, multi_channel=True)    

# train_loader, val_loader, test_loader = get_data(batch_size=32, seq_len=256, subset=True)

# input_data = None
# for i in train_loader:
#     input_data = i[0]
#     break
config  = {'model_name': "Attention is all you need",
            'model_type': "base_encoder",
              'model':None,
              'inherit_model': None, # The model to inherit from
              'embed_type': 'basic', # Posible options: 'relu_drop', 'gelu_drop', 'basic'
              'by_pass': False, # If channels are passed separatly through the model
              'pos_enc_type':'Sinusoidal', # Posible options: 'Sinusoidal', 'Relative', 'None', 'Learnable'
              'final_type': 'maxpool', # Posible options: 'basic', 'slim'
              'loss_function': 'BCEWithLogits', # Posible options: 'BCE', 'BCEWithLogits'
              'model_num': None,
              'seq_len': 256,
              'd_model': 128, # Have to be dividable by h
              'd_ff': 64,
              'N': 2,
              'h': 16,
              'output_size': 1,
              'dropout': 0.1,
              'num_epochs': None,
              'batch_size': None,
              "learning_rate": 1e-3,
              "decreas_factor": 0.5,
              "num_parms":0,
              'MACs':0,
              "data_path":'',
              "current_epoch":0,
              "global_epoch":0,
              "model_path":'',
              "test_acc":0,
              "early_stop":7,
              "omega": 10000,
              "trained_noise":0,
              "trained_signal":0,
              "data_type": "classic", # Possible options: 'classic', 'chunked'
              "n_ant":4,
              "metric":'Efficiency', # Posible options: 'Accuracy', 'Efficiency', 'Precision'
              "Accuracy":0,
              "Efficiency":0,
              "Precission":0,
              'nr_area':0,
              'roc_area':0,
              "trained": False,
              'power':0,
              'training_time':0,
              'energy':0,
         
            }

model = build_encoder_transformer(config)
input_data = torch.ones((32, 256, 4))
output = model.encode(input_data, src_mask=None)

# time_input_model = InputEmbeddings(d_model=512, channels=4, dropout=0, activation='none' )
# print(f"Shape input time embedding: {input_data.shape}")
# output = time_input_model(input_data)
# print(f"Shape output time embedding: {output.shape}")

# learn_pos_layer = LearnablePositionalEncoding(d_model=512)
# print(f"Shape input learnable position: {output.shape}")
# output = learn_pos_layer(output)
# print(f"Shape output learnable position: {output.shape}")

# multi_head = MultiHeadAttentionBlock(d_model=512, h=8, dropout=0.1, max_relative_position=10, relative_positional_encoding=True)
# print(f"Shape input to multi head: {output.shape}")
# output = multi_head(output, output, output, mask=None)
# print(f"Shape output from multi head: {output.shape}")

# final_layer = FinalBlock(d_model=512, seq_len=256, out_put_size=1, forward_type='new')
# final_output = final_layer(output)
# print(f"Shape input: {final_output.shape}")
