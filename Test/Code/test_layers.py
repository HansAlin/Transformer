import os
from os.path import dirname, abspath, join
import sys
import unittest
import numpy as np
# Find code directory relative to our directory
CODE_DIR_1  ='/home/halin/Master/Transformer'
sys.path.append(CODE_DIR_1)

# from models.models import InputEmbeddings, LayerNormalization, FinalBlock, MultiHeadAttentionBlock, LearnablePositionalEncoding
from models.models import InputEmbeddings, LayerNormalization, FinalBlock, MultiHeadAttentionBlock, LearnablePositionalEncoding, build_encoder_transformer, ModelWrapper, BatchNormalization
from dataHandler.datahandler import get_data, prepare_data, get_data_binary_class, get_model_config
from model_configs.config import get_config
from evaluate.evaluate import get_MMac, test_model
import torch


# PATH = '/home/halin/Master/Transformer/Test/data/test_100_data.npy'

# x_train, x_val, x_test, y_train, y_val, y_test = get_data(path=PATH)

# train_loader, val_loader, test_loader = prepare_data(x_train, x_val, x_test, y_train, y_val, y_test, batch_size=32, multi_channel=True)    


# model_number = 19
# config = get_model_config(model_num=model_number)
# model_path = config['model_path'] + 'saved_model' + f'/model_{config["model_num"]}.pth'
# model = build_encoder_transformer(config)
# print(f'Preloading model {model_path}')
# state = torch.load(model_path)
# model.load_state_dict(state['model_state_dict'])
# train_loader, val_loader, test_loader = get_data_binary_class(seq_len=config['seq_len'],
#                                                               batch_size=config['batch_size'],)

# y_pred_data, accuracy, efficiency, precission = test_model(model=model, 
#            test_loader=test_loader,
#            device=0,
#            config=config,)
# print(f"Accuracy: {accuracy}, Efficiency: {efficiency}, Precission: {precission}")

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

d_model = 64
batch_size = 32
seq_len = 128
input_data = torch.randn(batch_size, seq_len, d_model)

# normal = BatchNormalization(features=d_model)
# output = normal(input_data)
# print(f"Shape output: {output.shape}")

learnable = LearnablePositionalEncoding(d_model=d_model)
output = learnable(input_data)
print(f"Shape output: {output.shape}")
